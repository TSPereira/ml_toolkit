import pkg_resources
import prettytable
import os
import argparse


def get_pkg_license(pkg):
    """Finds the license associated to a package

    :param pkg_resources.EggInfoDistribution pkg: package to look for
    :return string: License for package pkg
    """

    try:
        lines = pkg.get_metadata_lines('METADATA')
    except (FileNotFoundError, KeyError, OSError):
        lines = pkg.get_metadata_lines('PKG-INFO')

    for line in lines:
        if line.startswith('License:'):
            return line[9:]
    return '(Licence not found)'


def packages_and_licenses(path='.', save=False, show=True):
    """Lists all packages and respective licenses in the current environment

    :param string path: default: '.'. Destination to save the resulting file to. If none is passed, defaults to
    current location
    :param bool save: default: False. Whether to save the resulting table to a file or not.
    :param bool show: default: True. Whether to print the table to the terminal or console
    :return: None
    """

    t = prettytable.PrettyTable(['Package', 'License'])
    for pkg in sorted(pkg_resources.working_set, key=lambda x: str(x).lower()):
        t.add_row((str(pkg), get_pkg_license(pkg)))

    if show:
        print(t)

    if save:
        with open(os.path.join(path, 'licenses.txt'), 'w') as h:
            h.write(t.get_string())

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ns', '--no-show', help='Show table on terminal', action='store_true')
    parser.add_argument('-s', '--save', help='Save file to path', action='store_true')
    parser.add_argument('-d', '--dest', help='Destination to save file to. Defaults to current location', default='.',
                        type=str)

    args = parser.parse_args()
    packages_and_licenses(path=args.dest, save=args.save, show=(not args.no_show))
