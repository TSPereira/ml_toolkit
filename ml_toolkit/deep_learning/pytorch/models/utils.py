from contextlib import contextmanager
from typing import Any
import torch.nn as nn


def check_non_linearity(item: Any) -> bool:
    """Evaluates whether "item" is a torch module

    :param item: variable to check
    :return: boolean
    """
    return isinstance(item(), nn.Module) if item is not None else True


@contextmanager
def evaluating(net: nn.Module) -> None:
    """Temporarily switch to evaluation mode.

    :param net: Network to evaluate
    :return: None
    """

    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()

    return
