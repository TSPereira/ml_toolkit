import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from ml_toolkit.clustering import Cluster, Metrics


if __name__ == '__main__':
    df = pd.read_csv('s4.txt', header=None, sep=r'\s+')
    df.columns = ['x', 'y']
    x = MinMaxScaler().fit_transform(df.values)

    mdl = Cluster(method='kmeans', predictor=RandomForestClassifier(n_estimators=10), estimate_best_k=False,
                  method_eps=0.05, method_min_samples=100, method_n_clusters=15,
                  max_k=20, verbose=2).fit(x)
    # mdl.est.plot_metrics()
    mdl.plot(x, s=20, fontsize=5, show_areas=True)

    metrics_results = mdl.get_metrics(df.values)
    print(1)

# todo reshape as unittest
