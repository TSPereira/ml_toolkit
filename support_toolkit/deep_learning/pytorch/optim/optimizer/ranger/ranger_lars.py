from ..lookahead import make_lookahead_optimizer
from ..ralamb import Ralamb


def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
    """
    RAdam + LARS + LookAHead

    Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
    RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20

    :param params: net parameters
    :param alpha: lookahead alpha parameter
    :param k: lookahead k parameter
    :param args: args for RaLamb
    :param kwargs: kwargs for RaLamb
    :return:
    """
    return make_lookahead_optimizer(Ralamb(params, *args, **kwargs), alpha, k)
