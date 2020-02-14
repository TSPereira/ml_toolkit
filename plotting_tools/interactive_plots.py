from itertools import cycle
import plotly.graph_objs as go
import plotly.offline as po
from plotly.colors import n_colors, label_rgb
from pandas.core.frame import DataFrame


def scatter(data, mode='lines+markers', title='', x_axis_title='', y_axis_title='', auto_open=False,
            filename='./temp_scatter_plot.html'):
    """Produces a scatter plot saved as a html figure at project root

    :param pandas.DataFrame data: DataFrame of values to plot. Names of columns will be used as names of each series
    and index will be used as x axis
    :param string mode: default = 'lines+markers'. value to control mode to plot on. Any combination of "lines",
    "markers", "text" joined with a "+" OR "none".  examples: "lines", "markers", "lines+markers",
    "lines+markers+text", "none"
    :param string title: default = ''. Main title of the scatter plot
    :param string x_axis_title: default = ''. title for x_axis
    :param string y_axis_title: default = ''. title for y_axis
    :param bool auto_open: default = False. Parameter to control whether the plot should be shown automatically in
    browser
    :param string filename: path and filename where to save for
    :return: plotly figure dictionary
    """

    assert isinstance(data, DataFrame), 'Function "scatter" is currently only implemented for DataFrames.'

    data = [go.Scatter(x=data.index, y=data[col], mode=mode, name=col) for col in data.columns]
    layout = go.Layout(title=title, xaxis=dict(title=x_axis_title), yaxis=dict(title=y_axis_title))
    fig = dict(data=data, layout=layout)
    po.plot(fig, filename=filename, auto_open=auto_open,)
    return fig


def scatter_grouped(data, mode='lines+markers', title='', x_axis_title='', y_axis_title='', auto_open=False,
                    filename='./temp_scatter_plot.html', ):
    """
    This function plots dataframes with multiindex columns as grouped data. If more than one dataframe is passed in
    "data", then all have to have same columns of the first one, and each dataframe will be plotted with one
    different linestyle.
    Each column of the last level of the column MultiIndex will be plotted with a different color (same color for
    same column on all dataframes)
    Each column of the first level of the column MultiIndex will be plotted as a group (in legend) (same group for
    same column on all dataframes)

    :param tuple|list data: Collection of dataframe. Each dataframe info will be plotted with different dash type
                            Columns of the dataframes must be equal. If there is a MultiIndex on columns, plots will be
                            grouped together.
    :param str mode: Some possible values are 'lines' or 'markers'. Several values can be passed together using a
    '+'. For example: "lines+markers". Check Plotly documentation on Scatter modes for more info
    :param str title: Title to put on the plot layout
    :param str x_axis_title: Title for the x axis
    :param str y_axis_title: Title for the y axis
    :param bool auto_open: boolean to control whether the generated html should open automatically once created
    :param str filename: path to file to write. If no extension is given, '.html' will be added to the file path
    :return: Plotly figure dictionary
    """

    assert isinstance(data, (tuple, list)), 'Data must be a collection of Dataframes'
    assert all(isinstance(d, DataFrame) for d in data), 'Function "scatter" is currently only implemented for ' \
                                                        'DataFrames.'
    assert all(all(data[0].columns == d.columns) for d in data), 'All DataFrames must have same columns index'

    _colors = (label_rgb((255, 0, 0)), label_rgb((0, 0, 255)))
    _dashes = ['solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']

    unq_col_lwr = data[0].columns.get_level_values(-1).unique().to_list()
    colors = n_colors(*_colors, n_colors=len(unq_col_lwr), colortype='rgb') if len(unq_col_lwr) > 2 else _colors
    dashes = cycle(_dashes[:len(data)])

    data_trace = []
    for col in data[0].columns:
        trace = col[-1]
        color = colors[unq_col_lwr.index(trace)]
        legendgroup = None if len(col) == 1 else col[0]
        trace = trace if legendgroup is None else legendgroup + '_' + trace

        data_trace.extend([go.Scatter(x=d.index, y=d[col], mode=mode, name=trace, visible='legendonly',
                                      legendgroup=legendgroup, line=dict(color=color, dash=next(dashes)))
                           for d in data])

    layout = go.Layout(title=title, xaxis=dict(title=x_axis_title), yaxis=dict(title=y_axis_title))
    fig = dict(data=data_trace, layout=layout)
    po.plot(fig, filename=filename, auto_open=auto_open, )
    return fig
