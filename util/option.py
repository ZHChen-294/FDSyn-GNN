import os
import csv
import argparse


def parse(argv_list=None):
    parser = argparse.ArgumentParser(
        description="FDSyn-GNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--exp_name", type=str, default="FDSyn_GNN")
    parser.add_argument("-k", "--k_fold", type=int, default=5)
    parser.add_argument("-b", "--minibatch_size", type=int, default=4)

    parser.add_argument("-ds", "--sourcedir", type=str, default="")
    parser.add_argument("-dt", "--targetdir", type=str, default="")

    parser.add_argument("--dataset", type=str, default="mdd-rest")
    parser.add_argument("--target_feature", type=str, default="Dx")
    parser.add_argument("--roi", type=str, default="aal")
    parser.add_argument("--fwhm", type=float, default=None)

    parser.add_argument("--dynamic_length", type=int, default=140)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--reg_lambda", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=30)

    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--finetune_epochs", type=int, default=5)

    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--sparsity", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--readout", type=str, default="sero", choices=["garo", "sero", "mean"])
    parser.add_argument("--cls_token", type=str, default="sum", choices=["sum", "mean", "param"])


    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--evaluation", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=-1)

    argv = parser.parse_args(argv_list)

    argv.savedir = argv.targetdir
    argv.targetdir = os.path.join(argv.targetdir, argv.exp_name)
    os.makedirs(argv.targetdir, exist_ok=True)

    argv_path = os.path.join(argv.targetdir, "argv.csv")
    file_exists = os.path.isfile(argv_path)
    with open(argv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["key", "value"])
        for k, v in vars(argv).items():
            writer.writerow([k, v])

    return argv

