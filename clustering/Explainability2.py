import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from itertools import product
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from ..clustering.plot_functions import take_outliers_graph


class Explainability2:

    def __init__(self, data, features_to_plot, labels_column, title='', save_path='', labels_by_plot=5, verbose=False):
        '''
        Constructor checks if input data is correct. Tries to fix problems.

        :param data:
        :param features_to_plot:
        :param labels_column:
        :param title:
        :param save_path:
        :param labels_by_plot:
        :param verbose:
        '''

        if type(data) is not pd.DataFrame:
            raise TypeError('Input "data" must be a pandas DataFrame!')

        self.data = data
        self.data[labels_column] = self.data[labels_column].astype('str')

        if type(features_to_plot) is not dict:
            raise TypeError('Input "features_to_plot" must be a dict!'
                            'Relevant keys are: "ordinal", "categorical" and "continuous".')

        for feature_type in ['ordinal', 'categorical', 'continuous']:
            if feature_type not in features_to_plot.keys():
                features_to_plot[feature_type] = []
            if type(features_to_plot[feature_type]) is not list:
                features_to_plot[feature_type] = list(str(features_to_plot[feature_type]))

        self.features_to_plot = features_to_plot

        for input in [labels_column, title, save_path]:
            if type(input) is not str:
                raise TypeError('Input "' + input + '"must be a string!')

        self.labels_column = labels_column
        self.title = title
        self.save_path = save_path

        self.labels_by_plot = labels_by_plot
        self.verbose = verbose
        self.number_of_clusters = len(self.data[labels_column].unique())

    def barplot_frequencies(self, feature, type_='category', relative=False):

        if type_ not in ['category', 'cluster']:
            raise ValueError('Either "category" or "cluster" for barplot_frequencies')

        unique_feature = sorted([x for x in self.data[feature].unique() if str(x) not in ['nan', 'None']])
        unique_labels = sorted([x for x in list(set(map(str, self.data[self.labels_column]))) if str(x) != 'nan'])

        # Create DataFrame with all possible combinations of features and labels
        df = pd.DataFrame([i for i in product(unique_feature, unique_labels)], columns=[feature, self.labels_column])

        # Count frequencies and append
        df2 = self.data.groupby([feature, self.labels_column]).size().reset_index()
        df2[self.labels_column] = df2[self.labels_column].astype('str')
        df2[feature] = df2[feature].astype(df[feature].dtype)
        df = df.merge(df2, how='left', on=[feature, self.labels_column])

        # Calculate relative frequencies
        if relative:
            # Count totals append
            if type_ == 'cluster':
                df2 = df.groupby([self.labels_column]).sum().reset_index()
                df = df.merge(df2, how='left', on=self.labels_column)
            if type_ == 'category':
                df2 = df.groupby([feature]).sum().reset_index()
                df = df.merge(df2, how='left', on=feature)
            df['_freq'] = df.iloc[:, -2] / df.iloc[:, -1]

        # Divide into multiple plots based on number of labels
        if self.labels_by_plot is None:
            number_of_plots = 1
        else:
            number_of_plots = int(np.ceil(len(unique_feature) / self.labels_by_plot))

        for i in range(number_of_plots):

            fig = plt.figure(figsize=(20, 10))

            if relative:
                fig.suptitle(self.title + ' | ' + 'Relative frequencies of [' + feature + '] by: ' + type_,
                             size='xx-large', weight='bold')
            else:
                fig.suptitle(self.title + ' | ' + 'Absolute frequencies of [' + feature + '] by: ' + type_,
                             size='xx-large', weight='bold')

            # Filter DataFrame if plot is split into multiple figures
            if self.labels_by_plot is None:
                df_subset = df
            else:
                filter_df = unique_feature[i * self.labels_by_plot: (i + 1) * self.labels_by_plot]
                df_subset = df[df[feature].isin(filter_df)]
            #
            frequencies_subset = df_subset.iloc[:, -1]

            clusters = ['Cluster ' + i for i in df_subset[self.labels_column].astype(str)]
            # This where the plot is plotted
            if type_ == 'cluster':
                with sns.color_palette("coolwarm", len(df_subset[feature].unique())):
                    g = sns.barplot(y=frequencies_subset, x=clusters, hue=feature, data=df_subset)
            if type_ == 'category':
                with sns.color_palette("colorblind", self.number_of_clusters):
                    g = sns.barplot(y=frequencies_subset, x=feature, hue=clusters, data=df_subset)

            # Some axis configurations
            if relative:
                g.set(ylim=(0, 1))
                g.set(ylabel='Frequency (%)')
            else:
                g.set(ylabel='Frequency')

            if type_ == 'cluster':
                g.set(xlabel='')
            if type_ == 'category':
                g.set(xlabel=feature)
                plt.xticks(rotation=20)

            # fig.show()

            # File name and save
            if relative:
                name_of_plot = 'relative_frequency_by_' + type_
            else:
                name_of_plot = 'absolute_frequency_by_' + type_

            if len(range(number_of_plots)) == 1:
                self._save_figure(name_of_plot=name_of_plot, fig=fig, feature=feature)
            else:
                self._save_figure(name_of_plot=name_of_plot, fig=fig, feature=feature, i=i)

        return

    def lineplot_frequencies(self, feature, relative=False):

        unique_feature = sorted([x for x in self.data[feature].unique() if str(x) not in ['nan', 'None']])
        unique_labels = sorted([x for x in list(set(map(str, self.data[self.labels_column]))) if str(x) != 'nan'])

        # Create DataFrame with all possible combinations of features and labels
        df = pd.DataFrame([i for i in product(unique_feature, unique_labels)], columns=[feature, self.labels_column])

        # Count frequencies and append
        df2 = self.data.groupby([feature, self.labels_column]).size().reset_index()
        df2[self.labels_column] = df2[self.labels_column].astype('str')
        df2[feature] = df2[feature].astype(df[feature].dtype)
        df = df.merge(df2, how='left', on=[feature, self.labels_column])

        # Calculate relative frequencies
        if relative:
            # Count totals append
            df2 = df.groupby([feature]).sum().reset_index()
            df = df.merge(df2, how='left', on=feature)
            df['_freq'] = df.iloc[:, -2] / df.iloc[:, -1]

        # Divide into multiple plots based on number of labels

        fig = plt.figure(figsize=(20, 10))

        if relative:
            fig.suptitle(self.title + ' | ' + 'Relative frequencies of [' + feature + ']',
                         size='xx-large', weight='bold')
        else:
            fig.suptitle(self.title + ' | ' + 'Absolute frequencies of [' + feature + ']',
                         size='xx-large', weight='bold')

        frequencies_subset = df.iloc[:, -1]

        # This where the plot is plotted
        with sns.color_palette("colorblind", self.number_of_clusters):
            g = sns.lineplot(y=frequencies_subset, x=[value.left for value in df[feature]],
                             hue=['Cluster ' + i for i in df[self.labels_column].astype(str)], linewidth=3)

        # Some axis configurations
        if relative:
            g.set(ylim=(0, 1))
            g.set(ylabel='Frequency (%)')
        else:
            g.set(ylabel='Frequency')

        g.set(xlabel=feature)
        plt.xticks(rotation=20)

        fig.show()

        # File name and save
        if relative:
            name_of_plot = 'continuous_relative_frequency'
        else:
            name_of_plot = 'continuous_absolute_frequency'

        self._save_figure(name_of_plot=name_of_plot, fig=fig, feature=feature)

        return

    def classic_explainability(self, feature):

        main_color_mpl = 'salmon'

        data = self.data.copy()
        unique_labels = sorted([x for x in list(set(map(str, data[self.labels_column]))) if str(x) != 'nan'])
        n_labels = len(unique_labels)
        # Calculate the number of plots and how to place them in each figure

        if np.sqrt(n_labels).is_integer():
            rows = cols = int(np.sqrt(n_labels))
        else:
            cols = int(np.ceil(np.sqrt(n_labels)))
            rows = int(n_labels // cols + (1 if n_labels % cols > 0 else 0))

        # Reduce the amount of chars to plot
        data[feature] = ['\n'.join(wrap(str(l), 20)) for l in data[feature]]

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(self.title + ' | ' + 'Frequency of ' + feature, size='xx-large', weight='bold')

        for i in range(n_labels):
            filter_df = (data[self.labels_column] == unique_labels[i])
            self.datalabel = data.loc[filter_df, :]
            population = self.datalabel.shape[0]

            ax = plt.subplot(rows, cols, i + 1)
            ax.text(0.7, 1.05, 'population=' + str(population), color='k', transform=ax.transAxes)

            val = pd.value_counts(self.datalabel[feature], sort=False, ascending=False, normalize=True) * 100

            # Select the top 10 categories in the feature that better represent the cluster
            index = val.sort_values(ascending=False)[:10].index.tolist()

            val[index].sort_index().plot(kind='bar', ax=ax, rot=45, color=main_color_mpl, alpha=0.5)

            for k, v in enumerate(val[index].sort_index()):
                ax.text(k, v + 10, str(round(v)), color='k', rotation=90, fontsize=8)

            ax.set_ylim(0, 115)
            ax.set_ylabel('%')
            ax.yaxis.set_label_coords(-0.02, 1)
            ax.xaxis.set_tick_params(labelsize=6)

        # Properties to rearrange the plots within each picture
        plt.subplots_adjust(top=0.9, bottom=0.106, left=0.036, right=0.966, hspace=0.545, wspace=0.198)

        self._save_figure(name_of_plot='classic_explainability', fig=fig, feature=feature)

        return

    def plot_all(self):

        if self.verbose:
            print('Performing explainability2 plots...')

        for feature in self.features_to_plot['continuous']:
            # Backup original feature
            self.data[feature + '_original'] = self.data[feature]
            # Make as many bins as possible (out of 100) to plot
            self.data[feature] = pd.qcut(self.data[feature + '_original'], q=100, duplicates='drop')

            # Absolute frequencies
            self.lineplot_frequencies(feature, relative=False)

            # Relative frequencies
            self.lineplot_frequencies(feature, relative=True)

            # Use original to obtain deciles
            self.data[feature] = pd.qcut(self.data[feature + '_original'], q=10, duplicates='drop')

        for feature in (self.features_to_plot['ordinal'] + self.features_to_plot['categorical'] +
                        self.features_to_plot['continuous']):

            # Absolute frequencies
            self.barplot_frequencies(feature)
            self.barplot_frequencies(feature, type_='cluster')

            # Relative frequencies
            self.barplot_frequencies(feature, relative=True)
            self.barplot_frequencies(feature, type_='cluster', relative=True)

            # Classic explainability
            self.classic_explainability(feature)

    def _save_figure(self, fig, feature, name_of_plot='', i=None):
        # Save the plots to the correct folder
        title = self.title
        if self.title != '' and self.title[-1] != '_':
            title += '_'
        if feature[-1] != '_':
            feature += '_'
        if i is not None:
            feature += str(i) + '_'
        if self.save_path != '':
            filename = title + feature + name_of_plot + '.png'
            filename = filename.replace('/', '_per_')
            fig.savefig(self.save_path + filename)
            if self.verbose:
                print('Printing: ' + self.save_path + filename)
            plt.close(fig)
        else:
            if self.verbose:
                print('Path to save pictures was not given. Will not save.')


def explainability(data, params, labels, cluster_name='', save_path=''):
    """
    Function to decode the used data and plot the features to provide explainability to the clusters

    :param data: encoded data used in the model (scipy.sparse.csr_matrix)
    :param params: dictionary with each feature encoder and location in the total array
    :param labels: array with the assigned cluster label to each record
    :param cluster_name: name of the cluster to print on the plots
    :param save_path: path to save pictures to
    :return: Plots
    """

    print('Performing explainability plots...')

    # Style and colors to use in plots
    # plt.style.use('ggplot')
    # plt.rcParams['figure.facecolor'] = 'w'
    main_color_sns = 'darkslategrey'
    main_color_mpl = 'salmon'

    # Find which variables were encoded as continuous, binary or multilabel
    cont_params = {enc.name: [enc.encoder, enc.pos, enc.gamma] for enc in params if
                   isinstance(enc.encoder, MinMaxScaler)}
    multi_params = {enc.name: [enc.encoder, enc.pos, enc.gamma] for enc in params if
                    isinstance(enc.encoder, MultiLabelBinarizer)}
    cat_params = {enc.name: [enc.encoder, enc.pos, enc.gamma] for enc in params
                  if enc.name not in list(cont_params.keys()) + list(multi_params.keys())}

    # cont_params = {key: param for key, param in params.items() if isinstance(param[0], MinMaxScaler)}
    # multi_params = {key: param for key, param in params.items() if isinstance(param[0], MultiLabelBinarizer)}
    # cat_params = {key: param for key, param in params.items()
    #               if key not in list(cont_params.keys()) + list(multi_params.keys())}

    # Calculate the number of plots and how to place them in each figure
    nr_labels = len(np.unique(labels))
    if np.sqrt(nr_labels).is_integer():
        rows = cols = int(np.sqrt(nr_labels))
    else:
        cols = int(np.ceil(np.sqrt(nr_labels)))
        rows = int(nr_labels // cols + (1 if nr_labels % cols > 0 else 0))

    # Create one figure for each feature
    for key, param in {**cont_params, **multi_params, **cat_params}.items():  # params.items():

        # Initialize figure settings
        fig = plt.figure(figsize=(20, 10))
        plt.grid(True)
        if key[:4] == 'log_':
            key_title = key[4:]
        else:
            key_title = key
        plt.suptitle(cluster_name + ' | ' + 'Frequency of ' + key_title, size='xx-large', weight='bold')

        _xmax = 1
        _xmin = 0
        for i in range(nr_labels):
            # Get data for the current cluster
            data_lab = data.tocsr()[labels == np.unique(labels)[i], :]

            # Get # population for current cluster
            pop = data_lab.shape[0]
            ax = plt.subplot(rows, cols, i + 1)
            ax.text(0.7, 1.05, 'population=' + str(pop), color='k', transform=ax.transAxes)

            # For continuous variables plot an histogram
            if key in cont_params.keys():
                values = data_lab[:, param[1][0]:param[1][1]].toarray() / param[2]
                values = param[0].inverse_transform(values).ravel()

                if key[:4] == 'log_':
                    values = np.expm1(values)

                _counts = pd.Series(values).value_counts(normalize=True).sort_index() * 100
                if _counts.shape[0] < 30:
                    ax.scatter(_counts.index, _counts.values, color='mediumblue')
                    for k, v in _counts.iteritems():
                        ax.text(k, v + 10, str(round(v)), color='k', rotation=90, fontsize=8)

                    ax.set_ylim(0, 115)
                    ax.set_ylabel('%')
                    ax.yaxis.set_label_coords(-0.02, 1)
                    ax.xaxis.set_tick_params(labelsize=6)
                    _range_x = _counts.index
                else:
                    val = take_outliers_graph(values)
                    sns.distplot(val, bins=50, ax=ax, color=main_color_sns, kde_kws={'color': 'red'})
                    _range_x = val

                plt.xticks(rotation=45)
                _xmin = np.min([_xmin, _range_x.min()])
                _xmax = np.max([_xmax, _range_x.max()])

            # For categorical variables plot bar plots with the count of the most frequent categories
            elif key in cat_params.keys():
                values = (data_lab[:, param[1][0]:param[1][1]].toarray() / param[2]).astype(int)
                values = param[0].inverse_transform(values).ravel()
                values = ['\n'.join(wrap(str(l), 20)) for l in values]
                val = pd.value_counts(pd.Series(values, name=key), sort=False, ascending=False, normalize=True) * 100
                index = val.sort_values(ascending=False)[:10].index.tolist()
                val[index].sort_index().plot(kind='bar', ax=ax, rot=45, color=main_color_mpl, alpha=0.5)

                for k, v in enumerate(val[index].sort_index()):
                    ax.text(k, v + 10, str(round(v)), color='k', rotation=90, fontsize=8)
                ax.set_ylim(0, 115)
                ax.set_ylabel('%')
                ax.yaxis.set_label_coords(-0.02, 1)
                ax.xaxis.set_tick_params(labelsize=6)

            # For Multi categorical variables plot most frequent labels according to their frequency
            elif key in multi_params.keys():
                values = (data_lab[:, param[1][0]:param[1][1]].toarray() / param[2]).astype(int)
                lbs = param[0].classes_

                # Get percentages of existence of each extra for the cluster
                per = values.sum(axis=0) * 100 / values.shape[0]
                val = pd.Series(per, name=key, index=lbs).sort_values(ascending=False)
                val[:30].plot(kind='bar', ax=ax, color=main_color_mpl, alpha=0.5)
                for k, v in enumerate(val[:30]):
                    ax.text(k - 0.1, v + 10, str(round(v)), color='k', rotation=90, fontsize=6)
                ax.set_ylim(0, 115)
                ax.set_ylabel('%')
                ax.yaxis.set_label_coords(-0.02, 1)

            ax.set_title('Cluster ' + str(np.unique(labels)[i]), pad=15, weight='semibold')

        # Properties to rearrange the plots within each picture
        plt.subplots_adjust(top=0.9, bottom=0.106, left=0.036, right=0.966, hspace=0.545, wspace=0.198)
        if key in cont_params.keys():
            for i in range(nr_labels):
                ax = plt.subplot(rows, cols, i + 1)
                ax.set_xlim(_xmin - 0.05 * (_xmax - _xmin) / _xmax, _xmax * 1.05)
        #
        #     plt.subplots_adjust(top=0.9, bottom=0.106, left=0.036, right=0.966, hspace=0.545, wspace=0.198)
        #
        # elif key in cat_params.keys():
        #     plt.subplots_adjust(top=0.9, bottom=0.106, left=0.036, right=0.966, hspace=0.545, wspace=0.198)
        #
        # elif key in multi_params.keys():
        #     plt.subplots_adjust(top=0.9, bottom=0.106, left=0.036, right=0.966, hspace=0.545, wspace=0.198)

        # Save the plots to the correct folder
        if save_path != '':
            fig.savefig(save_path + '/' + cluster_name + '_' + 'frequency_of_' +
                        key_title.replace('/', '_per_').replace(' |', '').replace(' ', '_') + '.png')
            plt.close(fig)
        else:
            print('Path to save pictures was not given. Will not save.')

    return


if __name__ == '__main__':
    import pickle
    import bz2

    with open('results/train/current/similarity_dot_tsne_2.pkl', 'rb') as f:
        source = pd.DataFrame(pickle.load(f))

    with open('results/train/current/contacts.bz2', 'rb') as f:
        contacts = pickle.load(bz2.BZ2File(f))['data_cleaned']

    source = source.merge(contacts, how='left', left_on='id', right_on='contact_id')

    source_sample = source
    features_to_explain = {'ordinal': ['age_bins', 'avg_monetary_bins'],
                           'categorical': [],
                           'continuous': ['avg_monetary']}

    labels_column = 'label'

    exp = Explainability2(source_sample, features_to_explain, labels_column, title='Barroca',
                          save_path='results/plots/', labels_by_plot=100, verbose=True)
    exp.plot_all()
