import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from einops import repeat
from torch.utils.tensorboard import SummaryWriter

import util
from model.FDSyn_GNN import FDSyn_GNN
from dataset import DatasetDIRECT


def step(model, criterion, v, a, t, label, clip_grad=0.0, device="cpu", optimizer=None, scheduler=None):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    out = model(v.to(device), a.to(device), t.to(device))
    logit = out[0] if isinstance(out, (tuple, list)) else out
    loss = criterion(logit, label.to(device))

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        if clip_grad > 0.0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    return logit, loss


def make_ce_weight_from_counts(class_counts, device):
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("class_counts must be a 1D non-empty list/array.")
    if torch.any(counts <= 0):
        raise ValueError("All class_counts must be > 0.")
    w = 1.0 / counts
    w = w / w.mean()
    return w


def build_dataset(argv):
    return DatasetDIRECT(
        argv.sourcedir,
        roi=argv.roi,
        k_fold=argv.k_fold,
        dynamic_length=getattr(argv, "dynamic_length", None),
        target_feature=getattr(argv, "target_feature", None),
        smoothing_fwhm=getattr(argv, "fwhm", None),
    )


def build_dataloader(dataset, argv, shuffle=False, batch_size=None):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=argv.minibatch_size if batch_size is None else batch_size,
        shuffle=shuffle,
        num_workers=argv.num_workers,
        pin_memory=True,
    )


def compute_class_counts(dataset):
    counts = np.zeros(dataset.num_classes, dtype=np.int64)
    for i in range(len(dataset)):
        y = dataset[i]["label"]
        if torch.is_tensor(y):
            y = int(y.item())
        else:
            y = int(y)
        counts[y] += 1
    return counts.tolist()


def train(argv):
    os.makedirs(os.path.join(argv.targetdir, "model"), exist_ok=True)
    os.makedirs(os.path.join(argv.targetdir, "summary"), exist_ok=True)

    torch.manual_seed(argv.seed)
    np.random.seed(argv.seed)
    random.seed(argv.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(argv.seed)

    dataset = build_dataset(argv)

    ckpt_path = os.path.join(argv.targetdir, "checkpoint.pth")
    if os.path.isfile(ckpt_path):
        print("resuming checkpoint experiment")
        checkpoint = torch.load(ckpt_path, map_location=device)
    else:
        checkpoint = {"fold": 0, "epoch": 0, "model": None, "optimizer": None, "scheduler": None}

    for k_index, k in enumerate(dataset.folds):
        if checkpoint["fold"]:
            if k_index < dataset.folds.index(checkpoint["fold"]):
                continue

        os.makedirs(os.path.join(argv.targetdir, "model", str(k)), exist_ok=True)

        dataset.set_fold(k, train=True)
        train_loader = build_dataloader(dataset, argv, shuffle=True)
        train_counts = compute_class_counts(dataset)

        dataset.set_fold(k, train=False)
        val_loader = build_dataloader(dataset, argv, shuffle=False)

        model = FDSyn_GNN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        ).to(device)

        if checkpoint["model"] is not None:
            model.load_state_dict(checkpoint["model"])

        if dataset.num_classes > 1:
            weight = make_ce_weight_from_counts(train_counts, device=device)
            criterion_train = torch.nn.CrossEntropyLoss(weight=weight)
            criterion_eval = torch.nn.CrossEntropyLoss()
        else:
            criterion_train = torch.nn.MSELoss()
            criterion_eval = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=argv.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=argv.max_lr,
            epochs=argv.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.2,
            div_factor=argv.max_lr / argv.lr,
            final_div_factor=1000,
        )

        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])

        summary_writer = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "train"))
        summary_writer_val = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "val"))
        logger = util.logger.Logger(dataset.folds, dataset.num_classes)

        for epoch in range(checkpoint["epoch"], argv.num_epochs):
            logger.initialize(k)
            dataset.set_fold(k, train=True)

            loss_accumulate = 0.0

            for i, x in enumerate(tqdm(train_loader, ncols=60, desc=f"k:{k} e:{epoch}")):
                a = util.bold.process_fc(x["timeseries"])
                if i == 0:
                    v = repeat(torch.eye(dataset.num_nodes), "n1 n2 -> b n1 n2", b=argv.minibatch_size)
                if len(a) < argv.minibatch_size:
                    v = v[: len(a)]

                t = x["timeseries"]
                label = x["label"]

                logit, loss = step(
                    model=model,
                    criterion=criterion_train,
                    v=v,
                    a=a,
                    t=t,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit

                loss_accumulate += float(loss.detach().cpu().numpy())
                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                summary_writer.add_scalar("lr", scheduler.get_last_lr()[0], i + epoch * len(train_loader))

            samples = logger.get(k)
            metrics = logger.evaluate(k)

            summary_writer.add_scalar("loss", loss_accumulate / max(1, len(train_loader)), epoch)
            if dataset.num_classes > 1 and "prob" in samples and samples["prob"].ndim == 2 and samples["prob"].shape[1] > 1:
                summary_writer.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1], epoch)
            for key, value in metrics.items():
                if key != "fold":
                    summary_writer.add_scalar(key, value, epoch)
            summary_writer.flush()

            torch.save(
                {
                    "fold": k,
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt_path,
            )

            if argv.validate:
                print("validating. not for testing purposes")
                logger.initialize(k)
                dataset.set_fold(k, train=False)

                val_loss_accumulate = 0.0

                for i, x in enumerate(val_loader):
                    with torch.no_grad():
                        a = util.bold.process_fc(x["timeseries"])
                        if i == 0:
                            v = repeat(torch.eye(dataset.num_nodes), "n1 n2 -> b n1 n2", b=argv.minibatch_size)
                        if len(a) < argv.minibatch_size:
                            v = v[: len(a)]

                        t = x["timeseries"]
                        label = x["label"]

                        logit, loss = step(
                            model=model,
                            criterion=criterion_eval,
                            v=v,
                            a=a,
                            t=t,
                            label=label,
                            clip_grad=argv.clip_grad,
                            device=device,
                            optimizer=None,
                            scheduler=None,
                        )

                    pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                    prob = logit.softmax(1) if dataset.num_classes > 1 else logit
                    logger.add(
                        k=k,
                        pred=pred.detach().cpu().numpy(),
                        true=label.detach().cpu().numpy(),
                        prob=prob.detach().cpu().numpy(),
                    )
                    val_loss_accumulate += float(loss.detach().cpu().numpy())

                samples = logger.get(k)
                metrics = logger.evaluate(k)

                summary_writer_val.add_scalar("loss", val_loss_accumulate / max(1, len(val_loader)), epoch)
                if dataset.num_classes > 1 and "prob" in samples and samples["prob"].ndim == 2 and samples["prob"].shape[1] > 1:
                    summary_writer_val.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1], epoch)
                for key, value in metrics.items():
                    if key != "fold":
                        summary_writer_val.add_scalar(key, value, epoch)
                summary_writer_val.flush()

        torch.save(model.state_dict(), os.path.join(argv.targetdir, "model", str(k), "model.pth"))
        checkpoint.update({"epoch": 0, "model": None, "optimizer": None, "scheduler": None})

        summary_writer.close()
        summary_writer_val.close()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


def test(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = build_dataset(argv)
    dataloader = build_dataloader(dataset, argv, shuffle=False, batch_size=1)
    logger = util.logger.Logger(dataset.folds, dataset.num_classes)

    if dataset.num_classes > 1:
        criterion_eval = torch.nn.CrossEntropyLoss()
    else:
        criterion_eval = torch.nn.MSELoss()

    for k in dataset.folds:
        model = FDSyn_GNN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        ).to(device)

        model.load_state_dict(torch.load(os.path.join(argv.targetdir, "model", str(k), "model.pth"), map_location=device))

        summary_writer = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "test"))

        logger.initialize(k)
        dataset.set_fold(k, train=False)

        loss_accumulate = 0.0

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f"k:{k}")):
            with torch.no_grad():
                a = util.bold.process_fc(x["timeseries"])
                if i == 0:
                    v = repeat(torch.eye(dataset.num_nodes), "n1 n2 -> b n1 n2", b=argv.minibatch_size)
                if len(a) < argv.minibatch_size:
                    v = v[: len(a)]

                t = x["timeseries"]
                label = x["label"]

                logit, loss = step(
                    model=model,
                    criterion=criterion_eval,
                    v=v,
                    a=a,
                    t=t,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit

                logger.add(
                    k=k,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                loss_accumulate += float(loss.detach().cpu().numpy())

        samples = logger.get(k)
        metrics = logger.evaluate(k)

        summary_writer.add_scalar("loss", loss_accumulate / max(1, len(dataloader)))
        if dataset.num_classes > 1 and "prob" in samples and samples["prob"].ndim == 2 and samples["prob"].shape[1] > 1:
            summary_writer.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1])
        for key, value in metrics.items():
            if key != "fold":
                summary_writer.add_scalar(key, value)
        summary_writer.flush()

        logger.to_csv(argv.targetdir, k, print=False)
        summary_writer.close()

    logger.to_csv(argv.targetdir)
    logger.evaluate(print=False)
    torch.save(logger.get(), os.path.join(argv.targetdir, "samples.pkl"))


def evaluation(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = build_dataset(argv)
    dataloader = build_dataloader(dataset, argv, shuffle=False, batch_size=1)
    logger = util.logger.Logger_eva(dataset.folds, dataset.num_classes)

    if dataset.num_classes > 1:
        criterion_eval = torch.nn.CrossEntropyLoss()
    else:
        criterion_eval = torch.nn.MSELoss()

    all_id, all_probs, all_labels = [], [], []

    for k in dataset.folds:
        model = FDSyn_GNN(
            input_dim=dataset.num_nodes,
            hidden_dim=argv.hidden_dim,
            num_classes=dataset.num_classes,
            num_heads=argv.num_heads,
            num_layers=argv.num_layers,
            sparsity=argv.sparsity,
            dropout=argv.dropout,
            cls_token=argv.cls_token,
            readout=argv.readout,
        ).to(device)

        model.load_state_dict(torch.load(os.path.join(argv.targetdir, "model", str(k), "model.pth"), map_location=device))

        summary_writer = SummaryWriter(os.path.join(argv.targetdir, "summary", str(k), "test"))

        logger.initialize(k)
        dataset.set_fold(k, train=False)

        loss_accumulate = 0.0

        for i, x in enumerate(tqdm(dataloader, ncols=60, desc=f"k:{k}")):
            with torch.no_grad():
                a = util.bold.process_fc(x["timeseries"])
                if i == 0:
                    v = repeat(torch.eye(dataset.num_nodes), "n1 n2 -> b n1 n2", b=argv.minibatch_size)
                if len(a) < argv.minibatch_size:
                    v = v[: len(a)]

                sub_id = x["id"]
                t = x["timeseries"]
                label = x["label"]

                logit, loss = step(
                    model=model,
                    criterion=criterion_eval,
                    v=v,
                    a=a,
                    t=t,
                    label=label,
                    clip_grad=argv.clip_grad,
                    device=device,
                    optimizer=None,
                    scheduler=None,
                )

                pred = logit.argmax(1) if dataset.num_classes > 1 else logit
                prob = logit.softmax(1) if dataset.num_classes > 1 else logit

                logger.add(
                    k=k,
                    id=sub_id,
                    pred=pred.detach().cpu().numpy(),
                    true=label.detach().cpu().numpy(),
                    prob=prob.detach().cpu().numpy(),
                )
                loss_accumulate += float(loss.detach().cpu().numpy())

        samples = logger.get(k)

        all_id.extend(samples["id"])
        all_labels.extend(samples["true"])
        all_probs.extend(samples["prob"])

        metrics = logger.evaluate(k)

        summary_writer.add_scalar("loss", loss_accumulate / max(1, len(dataloader)))
        if dataset.num_classes > 1 and "prob" in samples and samples["prob"].ndim == 2 and samples["prob"].shape[1] > 1:
            summary_writer.add_pr_curve("precision-recall", samples["true"], samples["prob"][:, 1])
        for key, value in metrics.items():
            if key != "fold":
                summary_writer.add_scalar(key, value)
        summary_writer.flush()

        logger.to_csv(argv.targetdir, k, print=False)
        summary_writer.close()

    all_id = np.array(all_id)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)[:, 1]

    results_df = pd.DataFrame({"SubID": all_id, "Label": all_labels, argv.exp_name: all_probs})
    save_path = os.path.join(argv.savedir, "result.csv")

    if os.path.exists(save_path):
        existing_results_df = pd.read_csv(save_path)

        if not (existing_results_df["SubID"].values == results_df["SubID"].values).all():
            raise ValueError("The 'SubID' column in the existing file does not match the current SubID list.")
        if not (existing_results_df["Label"].values == results_df["Label"].values).all():
            raise ValueError("The 'Label' column in the existing file does not match the current labels.")

        if argv.exp_name in existing_results_df.columns:
            print(f"Column '{argv.exp_name}' already exists in {save_path}. Skipping adding this column.")
        else:
            existing_results_df[argv.exp_name] = all_probs
            existing_results_df.to_csv(save_path, index=False)
            print(f"Added predictions for model '{argv.exp_name}' to {save_path}")
    else:
        results_df.to_csv(save_path, index=False)
        print(f"Created new result file: {save_path}")

    logger.to_csv(argv.targetdir)
    logger.evaluate(print=False)
    torch.save(logger.get(), os.path.join(argv.targetdir, "samples.pkl"))
