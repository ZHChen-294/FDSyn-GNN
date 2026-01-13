import os
import csv
import numpy as np
from sklearn import metrics


def _safe_concat(x_list, default_shape=None):
    if len(x_list) == 0:
        if default_shape is None:
            return np.array([])
        return np.zeros(default_shape)
    if len(x_list) == 1:
        return np.asarray(x_list[0])
    return np.concatenate([np.asarray(x) for x in x_list])


class BaseLogger:
    def __init__(self, k_fold=None, num_classes=None, with_id=False):
        self.k_fold = k_fold
        self.num_classes = num_classes
        self.with_id = with_id
        self.initialize(k=None)

    def __call__(self, **kwargs):
        if len(kwargs) == 0:
            return self.get()
        return self.add(**kwargs)

    def _initialize_metric_dict(self):
        d = {"pred": [], "true": [], "prob": []}
        if self.with_id:
            d["id"] = []
        return d

    def _print_metric(self, metric: dict):
        spacer = len(max(metric, key=len)) if metric else 0
        for key, value in metric.items():
            print(f"> {key:{spacer+1}}: {value}")

    def initialize(self, k=None):
        if self.k_fold is None:
            self.samples = self._initialize_metric_dict()
            return
        if k is None:
            self.samples = {fold: self._initialize_metric_dict() for fold in self.k_fold}
        else:
            self.samples[k] = self._initialize_metric_dict()

    def add(self, k=None, **kwargs):
        if self.k_fold is None:
            for name, value in kwargs.items():
                self.samples[name].append(value)
            return
        if k not in self.k_fold:
            raise ValueError(f"k must be one of {self.k_fold}, got {k}")
        for name, value in kwargs.items():
            self.samples[k][name].append(value)

    def _get_one(self, store: dict):
        out = {}
        if self.with_id:
            out["id"] = _safe_concat(store["id"])
        out["true"] = _safe_concat(store["true"])
        out["pred"] = _safe_concat(store["pred"])
        out["prob"] = _safe_concat(store["prob"])
        return out

    def get(self, k=None, initialize=False):
        if self.k_fold is None:
            out = self._get_one(self.samples)
        else:
            if k is None:
                out = {"true": {}, "pred": {}, "prob": {}}
                if self.with_id:
                    out["id"] = {}
                for fold in self.k_fold:
                    one = self._get_one(self.samples[fold])
                    if self.with_id:
                        out["id"][fold] = one["id"]
                    out["true"][fold] = one["true"]
                    out["pred"][fold] = one["pred"]
                    out["prob"][fold] = one["prob"]
            else:
                out = self._get_one(self.samples[k])

        if initialize:
            self.initialize(k)
        return out

    def evaluate(self, k=None, initialize=False, option="mean", print=True):
        samples = self.get(k)

        if self.num_classes == 1:
            if self.k_fold is not None and k is None:
                agg = np.mean if option == "mean" else np.std if option == "std" else None
                if agg is None:
                    raise ValueError("option must be 'mean' or 'std'")
                explained_var = agg([metrics.explained_variance_score(samples["true"][f], samples["pred"][f]) for f in self.k_fold])
                r2 = agg([metrics.r2_score(samples["true"][f], samples["pred"][f]) for f in self.k_fold])
                mse = agg([metrics.mean_squared_error(samples["true"][f], samples["pred"][f]) for f in self.k_fold])
            else:
                explained_var = metrics.explained_variance_score(samples["true"], samples["pred"])
                r2 = metrics.r2_score(samples["true"], samples["pred"])
                mse = metrics.mean_squared_error(samples["true"], samples["pred"])

            if initialize:
                self.initialize(k)
            metric = {"explained_var": explained_var, "r2": r2, "mse": mse}
            if print:
                self._print_metric(metric)
            return metric

        if self.num_classes and self.num_classes > 1:
            avg = "binary" if self.num_classes == 2 else "micro"

            def _auc(true, prob):
                if self.num_classes == 2:
                    return metrics.roc_auc_score(true, prob[:, 1])
                return metrics.roc_auc_score(true, prob, average="macro", multi_class="ovr")

            if self.k_fold is not None and k is None:
                agg = np.mean if option == "mean" else np.std if option == "std" else None
                if agg is None:
                    raise ValueError("option must be 'mean' or 'std'")
                accuracy = agg([metrics.accuracy_score(samples["true"][f], samples["pred"][f]) for f in self.k_fold])
                precision = agg([metrics.precision_score(samples["true"][f], samples["pred"][f], average=avg, zero_division=0) for f in self.k_fold])
                recall = agg([metrics.recall_score(samples["true"][f], samples["pred"][f], average=avg, zero_division=0) for f in self.k_fold])
                f1 = agg([metrics.f1_score(samples["true"][f], samples["pred"][f], average=avg, zero_division=0) for f in self.k_fold])
                roc_auc = agg([_auc(samples["true"][f], samples["prob"][f]) for f in self.k_fold])
            else:
                accuracy = metrics.accuracy_score(samples["true"], samples["pred"])
                precision = metrics.precision_score(samples["true"], samples["pred"], average=avg, zero_division=0)
                recall = metrics.recall_score(samples["true"], samples["pred"], average=avg, zero_division=0)
                f1 = metrics.f1_score(samples["true"], samples["pred"], average=avg, zero_division=0)
                roc_auc = _auc(samples["true"], samples["prob"])

            if initialize:
                self.initialize(k)
            metric = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc}
            if print:
                self._print_metric(metric)
            return metric

        raise ValueError("num_classes must be >= 1")

    def to_csv(self, targetdir, k=None, initialize=False, print=True):
        os.makedirs(targetdir, exist_ok=True)
        metric_dict = self.evaluate(k=k, initialize=initialize, print=print)

        path = os.path.join(targetdir, "metric.csv")
        append = os.path.isfile(path)

        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if not append:
                writer.writerow(["fold"] + list(metric_dict.keys()))
            writer.writerow([str(k)] + [str(v) for v in metric_dict.values()])

            if k is None and self.k_fold is not None:
                std_dict = self.evaluate(k=None, initialize=False, option="std", print=False)
                writer.writerow(["std"] + [str(v) for v in std_dict.values()])


class Logger(BaseLogger):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__(k_fold=k_fold, num_classes=num_classes, with_id=False)


class Logger_eva(BaseLogger):
    def __init__(self, k_fold=None, num_classes=None):
        super().__init__(k_fold=k_fold, num_classes=num_classes, with_id=True)
