# DISCLAIMER
# This package implements the Ranger Optimizer available in https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# as of 24/04/2020
#
# Ranger deep learning optimizer - RAdam + Lookahead + Gradient Centralization, combined into one optimizer.


from .ranger import Ranger
from .ranger_va import RangerVA
from .ranger_qh import RangerQH


__all__ = ['Ranger', 'RangerQH', 'RangerVA']
