import random
import numpy as np
import torch

import util
from experiment import train, test, evaluation


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def main() -> None:
    argv = util.option.parse()

    seed = getattr(argv, "seed", 0)
    set_seed(seed)

    if not any([getattr(argv, "train", False), getattr(argv, "test", False), getattr(argv, "evaluation", False)]):
        argv.train = True
        argv.test = True
        argv.evaluation = True

    if getattr(argv, "train", False):
        train(argv)
    if getattr(argv, "test", False):
        test(argv)
    if getattr(argv, "evaluation", False):
        evaluation(argv)


if __name__ == "__main__":
    main()
