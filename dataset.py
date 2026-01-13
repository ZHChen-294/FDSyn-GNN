import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy import io
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


@dataclass
class DirectPaths:
    roi_signal_dir: str
    excel_path: str
    cache_root: str


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_mat_timeseries(mat_path: str, key: str = "ROISignals") -> np.ndarray:
    d = io.loadmat(mat_path)
    if key not in d:
        raise KeyError(f"Key '{key}' not found in {mat_path}. Keys: {list(d.keys())}")
    x = d[key]
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        raise ValueError(f"ROISignals must be a 2D array, got {type(x)} with shape {getattr(x, 'shape', None)}")
    return x


def _zscore_timeseries(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def _roi_slice(roi: str) -> slice:
    roi_upper = str(roi).strip()
    if roi_upper in {"AAL", "aal"}:
        return slice(0, 116)
    if roi_upper in {"Harvard", "harvard"}:
        return slice(116, 228)
    if roi_upper in {"Craddock", "craddock"}:
        return slice(228, 428)
    if roi_upper in {"AHC", "ahc"}:
        return slice(0, 428)
    raise ValueError(f"Unsupported roi='{roi}'. Expected one of: AAL, Harvard, Craddock, AHC")


class DatasetDIRECT(torch.utils.data.Dataset):
    def __init__(
        self,
        sourcedir: str,
        roi: str,
        k_fold: Union[int, str, None] = 5,
        dynamic_length: Optional[int] = None,
        target_feature: str = "DX_GROUP",
        smoothing_fwhm: Optional[float] = None,
        paths: Optional[DirectPaths] = None,
        num_samples: int = -1,
        mat_key: str = "ROISignals",
        seed: int = 0,
    ):
        super().__init__()

        self.sourcedir = sourcedir
        self.roi = roi
        self.dynamic_length = dynamic_length
        self.target_feature = target_feature
        self.mat_key = mat_key
        self.seed = seed

        if paths is None:
            paths = DirectPaths(
                roi_signal_dir="",
                excel_path="",
                cache_root=sourcedir,
            )
        self.paths = paths

        self.cache_dir = os.path.join(self.paths.cache_root, f"MDD_{roi}_FC")
        _safe_mkdir(self.cache_dir)

        self.ts_cache_path = os.path.join(self.cache_dir, f"{roi}.pkl")
        self.participants_path = os.path.join(self.cache_dir, "participants.tsv")

        self.timeseries_dict, behavioral_df = self._load_or_build_cache()

        self.full_subject_list = list(self.timeseries_dict.keys())
        if 0 < num_samples < len(self.full_subject_list):
            rng = np.random.default_rng(seed)
            self.full_subject_list = rng.choice(self.full_subject_list, size=num_samples, replace=False).tolist()

        if target_feature not in behavioral_df.columns:
            raise KeyError(f"target_feature='{target_feature}' not found in behavioral_df columns: {list(behavioral_df.columns)}")

        self.behavioral_dict = behavioral_df[target_feature].to_dict()
        self.behavioral_dict = {str(k): (0 if v == -1 else v) for k, v in self.behavioral_dict.items()}

        self.full_subject_list = [sid for sid in self.full_subject_list if sid in self.behavioral_dict]
        if len(self.full_subject_list) == 0:
            raise ValueError("No subjects left after aligning timeseries with labels.")

        self.full_label_list = [self.behavioral_dict[sid] for sid in self.full_subject_list]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.full_label_list)
        self.num_classes = len(self.label_encoder.classes_)

        any_ts = next(iter(self.timeseries_dict.values()))
        self.num_timepoints, self.num_nodes = any_ts.shape

        self.folds, self.k_fold = self._build_folds(k_fold)
        self.k = None
        self.train = None
        self.subject_list = self.full_subject_list

    def _load_or_build_cache(self) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        if os.path.isfile(self.ts_cache_path) and os.path.isfile(self.participants_path):
            with open(self.ts_cache_path, "rb") as f:
                timeseries_dict = pickle.load(f)
            behavioral_df = pd.read_csv(self.participants_path, sep="\t").set_index("SubID")
            timeseries_dict = {str(k): v for k, v in timeseries_dict.items()}
            behavioral_df.index = behavioral_df.index.astype(str)
            return timeseries_dict, behavioral_df

        df = pd.read_excel(self.paths.excel_path)
        if "SubID" not in df.columns:
            raise KeyError(f"'SubID' column not found in excel: {self.paths.excel_path}")

        df["SubID"] = df["SubID"].astype(str)
        df = df.set_index("SubID")
        df.index = df.index.astype(str)

        sl = _roi_slice(self.roi)
        timeseries_dict: Dict[str, np.ndarray] = {}

        sub_id_list = df.index.to_list()
        for sid in tqdm(sub_id_list, ncols=60, desc="Loading ROISignals"):
            mat_path = os.path.join(self.paths.roi_signal_dir, f"ROISignals_{sid}.mat")
            if not os.path.isfile(mat_path):
                continue
            ts = _load_mat_timeseries(mat_path, key=self.mat_key)
            if ts.shape[1] < sl.stop:
                continue
            ts = ts[:, sl]
            timeseries_dict[sid] = ts

        if len(timeseries_dict) == 0:
            raise ValueError("No timeseries loaded. Check roi_signal_dir, excel_path, roi, and mat_key.")

        with open(self.ts_cache_path, "wb") as f:
            pickle.dump(timeseries_dict, f)

        df.to_csv(self.participants_path, sep="\t")
        return timeseries_dict, df

    def _build_folds(self, k_fold: Union[int, str, None]):
        if k_fold is None:
            return [], None

        if isinstance(k_fold, str):
            behavioral_df = pd.read_csv(self.participants_path, sep="\t").set_index("SubID")
            behavioral_df.index = behavioral_df.index.astype(str)
            if k_fold not in behavioral_df.columns:
                raise KeyError(f"k_fold='{k_fold}' not found in participants.tsv columns: {list(behavioral_df.columns)}")

            folds = sorted([str(x) for x in behavioral_df[k_fold].dropna().unique().tolist()])
            fold_map: Dict[str, Tuple[List[int], List[int]]] = {}

            idx_map = {sid: i for i, sid in enumerate(self.full_subject_list)}
            for fold in folds:
                test_sids = behavioral_df.index[behavioral_df[k_fold].astype(str) == fold].tolist()
                train_sids = behavioral_df.index[behavioral_df[k_fold].astype(str) != fold].tolist()
                train_idx = [idx_map[sid] for sid in train_sids if sid in idx_map]
                test_idx = [idx_map[sid] for sid in test_sids if sid in idx_map]
                fold_map[fold] = (train_idx, test_idx)
            return folds, fold_map

        k_fold = int(k_fold)
        folds = list(range(k_fold))
        if k_fold <= 1:
            return folds, None

        skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=self.seed)
        return folds, skf

    def __len__(self) -> int:
        return len(self.subject_list) if self.k is not None else len(self.full_subject_list)

    def set_fold(self, fold: Union[int, str], train: bool = True) -> None:
        if not self.k_fold:
            self.k = fold
            self.train = train
            self.subject_list = self.full_subject_list
            return

        self.k = fold
        if isinstance(fold, int):
            splits = list(self.k_fold.split(self.full_subject_list, self.full_label_list))
            train_idx, test_idx = splits[fold]
        else:
            train_idx, test_idx = self.k_fold[str(fold)]

        if train:
            rng = np.random.default_rng(self.seed)
            train_idx = list(train_idx)
            rng.shuffle(train_idx)
            self.subject_list = [self.full_subject_list[i] for i in train_idx]
            self.train = True
        else:
            self.subject_list = [self.full_subject_list[i] for i in test_idx]
            self.train = False

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        subject = self.subject_list[idx]
        ts = self.timeseries_dict[subject]
        ts = _zscore_timeseries(ts)

        if self.dynamic_length is not None:
            if ts.shape[0] < self.dynamic_length:
                raise ValueError(f"Subject {subject}: T={ts.shape[0]} < dynamic_length={self.dynamic_length}")
            if ts.shape[0] == self.dynamic_length:
                start = 0
            else:
                start = np.random.randint(0, ts.shape[0] - self.dynamic_length + 1)
            ts = ts[start : start + self.dynamic_length]

        y_raw = self.behavioral_dict[subject]
        y = int(self.label_encoder.transform([y_raw])[0])

        return {
            "id": subject,
            "timeseries": torch.tensor(ts, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
        }

    def get_class_counts(self) -> List[int]:
        y = self.label_encoder.transform(self.full_label_list)
        counts = np.bincount(y, minlength=self.num_classes)
        return counts.tolist()
