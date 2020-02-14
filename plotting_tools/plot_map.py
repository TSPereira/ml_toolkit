import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
from cartopy.feature import BORDERS
from cartopy.io.img_tiles import *
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def _basemap_portugal(extent):
    fig = plt.figure(figsize=(8, 16))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(extents=extent, crs=ccrs.Geodetic())
    ax.coastlines(resolution='10m')
    BORDERS.scale = '10m'
    ax.add_feature(BORDERS)

    # ax.gridlines(color='.5')
    # ax.set_xticks(np.arange(-10, -5, 1), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(36, 43, 1), crs=ccrs.PlateCarree())

    # ax.xaxis.set_major_formatter(LongitudeFormatter())
    # ax.yaxis.set_major_formatter(LatitudeFormatter())
    return fig, ax


def _plot_clusters_in_map(data):

    # define extent of the map to plot
    extent = data.lng.min() - 0.05, data.lng.max() + 0.05, data.lat.min() - 0.01, data.lat.max() + 0.01

    url = 'http://map1c.vis.earthdata.nasa.gov/wmts-geo/wmts.cgi'
    layer = 'Landsat_WELD_CorrectedReflectance_TrueColor_Global_Annual'
    fig, ax = _basemap_portugal(extent)

    cmap = plt.get_cmap('gist_rainbow')
    scatter = ax.scatter(data.lng, data.lat, marker='o', c=data.labels, s=2, cmap=cmap, transform=ccrs.Geodetic())

    ax.add_wmts(url, layer)
    ax.set_extent(extents=extent, crs=ccrs.Geodetic())

    fig.colorbar(scatter)
    plt.show()

    return


def _get_random_close_coords(coords, radius=0.0005):
    """
    Create random points inside of a small radius circle in order to represent on the map overlapping points as several points
    close to each others.

    :param coords: tuple, coordinates of the original point
    :param radius: radius of the circle where generated points can be created
    :return:
    """

    lng, lat = coords
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, radius)

    x = lng + r*np.cos(theta)
    y = lat + r*np.sin(theta)

    return x, y


def get_map(zone, data):
    """

    :param zone: zone of the map to plot labels on. {'Lisbon', 'Portugal'}
    :param data: pandas DataFrame with coordinates and labels to plot
    :return:
    """

    if zone == 'Lisbon':
        data_plot = data[data['zip_code'].str[0] == '1']
    elif zone == 'Portugal':
        data_plot = data[(data.lng > -9.7559) & (data.lng < -6.1747) & (data.lat < 42.1587) & (data.lat > 36.9171)]
    else:
        raise KeyError('Only implemented to \'Lisbon\' and \'Portugal\'.')

    # # apply label filter in case only one label to be shown
    # data_plot = data_plot[data_plot.labels == 2]

    _plot_clusters_in_map(data_plot)

    return


def _get_labels(data):
    """
    Create pandas DataFrame with labels and coordinates
    :param data: DataFrame with labels and zip_codes
    :return: DataFrame with labels and coordinates
    """
    zip_code_coordinates_path = 'packages/closer/aux_data/zip_code_coordinates.csv'
    coords = pd.read_csv(zip_code_coordinates_path)

    data = data[['zip_code', 'labels']].dropna()
    data.labels = data.labels.astype(int)
    data = pd.merge(data, coords, how='left', left_on='zip_code', right_on='zip_code')

    data['lng'], data['lat'] = zip(*data[['lng', 'lat']].apply(_get_random_close_coords, axis=1))

    return data


if __name__ == '__main__':
    with pd.HDFStore('results/train/trained_databases.h5', 'r', 9, 'blosc:lz4hc', True) as f:
        trained_clients = f['trained_clients']

    contacts = trained_clients.loc['contacts', 'cluster']
    accounts = trained_clients.loc['accounts', 'cluster']
    d = contacts.get_data_with_labels()
    data = _get_labels(d)
    get_map('Portugal', data)
