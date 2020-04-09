from typing import Iterable


def duplicated(l: Iterable) -> set:
    """Find which items are duplicated in a list (does not keep order)

    :param Iterable l: Iterable with items to check
    :return set: set with items repeated (unique)
    """

    seen = set()
    return set(x for x in l if x in seen or seen.add(x))