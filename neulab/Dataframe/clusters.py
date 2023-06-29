
def graph_clusterize(data, metric='euclidean', threshold=None):

    from neulab.Vector.clusters import graph_clusterize
    import numpy as np

    df = data.copy()

    vectors = df.values.tolist()

    # apply the graph_clusterize function
    clusters = graph_clusterize(vectors, metric=metric, threshold=None, draw=False)

    return clusters