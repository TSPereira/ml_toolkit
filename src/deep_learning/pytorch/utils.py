import os
from contextlib import suppress


def savefig(fig, name, save_path='.', fileformat='png'):
    if isinstance(fileformat, str):
        fileformat = [fileformat]
    for format_ in fileformat:
        with suppress(ValueError):
            fig.savefig(os.path.join(save_path, f'{name}.{format_}'), format=format_, dpi=600)
    return
