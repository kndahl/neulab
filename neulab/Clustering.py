from numpy import spacing
from neulab.Algorithms import manhattan_distance, euclidean_distance, max_metric
from neulab.Normalization import InterNormalization
import networkx as nx
import numpy as np
from neulab.Algorithms import euclidean_distance
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def CGraph(df, metric='euclid', r='std', rnd=3, draw=False, info=True):
    ''' Graph based clustering algorithm. 
    The algorithm itself determines the number of clusters. 
    Returns clusters linked in an array.'''

    # Determine round
    if rnd != 3:
        if type(rnd) == int and rnd > 0:
            ROUND = rnd
    else:
        ROUND = 3

    # Normalize
    for column in df:
        df[column] = InterNormalization(df[column])
    # Get distances
    distances = {}
    indx = df.index.values.tolist()
    for x in range(len(indx)-1):
        for i in range(len(indx)-1):
            if metric == 'manhattan':
                DIST = round(manhattan_distance(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'euclid':
                DIST = round(euclidean_distance(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'max':
                DIST = round(max_metric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            INDEX1 = df.index.values[x]
            INDEX2 = df.index.values[i+1]
            if INDEX1 == INDEX2: # Do not calculate dist between AA, BB, CC, DD etc.
                pass
            else:
                distances.update({str(INDEX1) + '|' + str(INDEX2): DIST})
    if info is True:
        print(f'Distances: {distances}')

    # Create list of edges
    edges = []
    for key, value in distances.items():
        key_split = key.split('|')
        edges.append([key_split[0], key_split[1], value])

    # Init Graph
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0],edge[1], weight = edge[2])
    # One more graph (remove edges)
    c = g.copy()

    # Calculate maen value of all distances (R)
    # TODO: Looks like R is not well calculated
    from neulab.Algorithms import std_deviation
    dist_list = list(distances.values())
    if r == 'std':
        R = round(std_deviation(vector=dist_list), ROUND)
    if r == 'mean':
        R = round(np.mean(vector=dist_list), ROUND)
    if r == 'median':
        R = round(np.median(vector=dist_list), ROUND)
    if info is True:
        print(f'R = {R}')

    # filter out all edges above threshold and grab id's
    long_edges = list(filter(lambda e: e[2] > R, (e for e in g.edges.data('weight'))))
    le_ids = list(e for e in long_edges)

    for node in le_ids:
        try:
            c.remove_edge(node[0], node[1])
        except nx.NetworkXError:
            pass

    if draw is True:
        # Draw Graph
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        ax = axes.flatten()
        ax[0].set_title('Graph')
        ax[1].set_title('Clusters')
        # Main
        edge_labels = dict( ((u, v), d["weight"]) for u, v, d in g.edges(data=True))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, node_size=600, node_color='pink', alpha=0.9,labels={node: node for node in g.nodes()}, ax=ax[0])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[0])

        # Cleared
        edge_labels = dict( ((u, v), d["weight"]) for u, v, d in c.edges(data=True))
        pos = nx.spring_layout(c)
        nx.draw(c, pos, node_size=600, node_color='pink', alpha=0.9,labels={node: node for node in c.nodes()}, ax=ax[1])
        nx.draw_networkx_edge_labels(c, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[1])

    # Get connected nodes (Clusters)
    import itertools
    all_connected_subgraphs = []
    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    for nb_nodes in range(2, c.number_of_nodes()):
        for SG in (c.subgraph(selected_nodes) for selected_nodes in itertools.combinations(c, nb_nodes)):
            if nx.is_connected(SG):
                # print(SG.nodes)
                all_connected_subgraphs.append(list(SG.nodes))
    if info is True:
        print(f'Found clusters: {all_connected_subgraphs}')

    return all_connected_subgraphs

def CGraphMST(df, clst_num, metric='euclid', rnd=3, draw=False, info=True):
    '''Graph based clustering algorithm. 
    The user himself sets the number of clusters. 
    Returns clusters linked in an array.'''

    # Determine round
    if rnd != 3:
        if type(rnd) == int and rnd > 0:
            ROUND = rnd
    else:
        ROUND = 3

    # Normalize
    for column in df:
        df[column] = InterNormalization(df[column])
    # Get distances
    distances = {}
    indx = df.index.values.tolist()
    for x in range(len(indx)-1):
        for i in range(len(indx)-1):
            if metric == 'manhattan':
                DIST = round(manhattan_distance(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'euclid':
                DIST = round(euclidean_distance(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'max':
                DIST = round(max_metric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            INDEX1 = df.index.values[x]
            INDEX2 = df.index.values[i+1]
            if INDEX1 == INDEX2: # Do not calculate dist between AA, BB, CC, DD etc.
                pass
            else:
                distances.update({str(INDEX1) + '|' + str(INDEX2): DIST})
    if info is True:
        print(f'Distances: {distances}')

    # Create list of edges
    edges = []
    for key, value in distances.items():
        key_split = key.split('|')
        edges.append([key_split[0], key_split[1], value])

    # Init Graph
    g = nx.Graph()
    for edge in edges:
        g.add_edge(edge[0],edge[1], weight = edge[2])

    st = nx.minimum_spanning_tree(g)
    spanning_tree = nx.minimum_spanning_tree(g).edges(data=True)
    if info is True:
        print(spanning_tree)

    # Sort tree by weight
    sorted_spanning_tree = sorted(spanning_tree,key= lambda x: x[2]['weight'],reverse=True)
    # Delete clst_num-1 edges with max weight
    edges_to_delete = sorted_spanning_tree[:clst_num-1]
    if info is True:
        print(f'Edges to delete: {edges_to_delete}')
    for elem in edges_to_delete:
        st.remove_edge(elem[0], elem[1])

    if draw is True:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))
        ax = axes.flatten()
        ax[0].set_title('Minimum Spanning Tree')
        ax[1].set_title('Clusters')

        edge_labels = dict( ((u, v), d["weight"]) for u, v, d in nx.minimum_spanning_tree(g).edges(data=True))
        pos = nx.spring_layout(nx.minimum_spanning_tree(g))
        nx.draw(nx.minimum_spanning_tree(g), pos, node_size=600, node_color='pink', alpha=0.9,labels={node: node for node in nx.minimum_spanning_tree(g).nodes()}, ax=ax[0])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[0])

        edge_labels = dict( ((u, v), d["weight"]) for u, v, d in st.edges(data=True))
        pos = nx.spring_layout(st)
        nx.draw(st, pos, node_size=600, node_color='pink', alpha=0.9,labels={node: node for node in st.nodes()}, ax=ax[1])
        nx.draw_networkx_edge_labels(st, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[1])

    # Get connected nodes (Clusters)
    import itertools
    all_connected_subgraphs = []
    # here we ask for all connected subgraphs that have at least 2 nodes AND have less nodes than the input graph
    for nb_nodes in range(2, st.number_of_nodes()):
        for SG in (st.subgraph(selected_nodes) for selected_nodes in itertools.combinations(st, nb_nodes)):
            if nx.is_connected(SG):
                all_connected_subgraphs.append(list(SG.nodes))
    if info is True:
        print(f'Found clusters: {all_connected_subgraphs}')

    return all_connected_subgraphs

class Forel:
    """
    A class representing FOREL clustering algorithm. 
    The algorithm itself determines the number of clusters. 

    Example:
        from neulab.Clustering import Forel
        import pandas as pd

        df = pd.read_csv('tests/csv/iris.csv').drop('Name', axis = 1)
        forel = Forel(data=df, verbose=True, scale = True, radius = 60)
        cluster = forel.get_clusters()

    """

    def __init__(self, data, radius=None, metric=euclidean_distance, scale=False, verbose=False):

        """
        Init function
        
        Parameter data: pandas.DataFrame with objects to cluster
        
        Parameter radius: search radius for local clusters
        Precondition: radius equals mean distance between objects divided by 2

        Parameter metric: metric function used to calculate distance between objects
        Precondition: neulab.Algorithms.euclidean_distance

        Parameter scale: If true all object parametres scaled in [0, 100]
        Precondition: False

        Parameter verbose: Show additional info
        Precondition: False

        """

        self.metric = metric
        self.scale = scale
        if self.scale:
            min_max_scaler = MinMaxScaler((0,100))
            self.data = pd.DataFrame(min_max_scaler.fit_transform(data), columns = data.columns)
        else:
            self.data = data.copy()
        self.verbose = verbose
        self.points = {}
    
        for index, row in self.data.iterrows():
            self.points.update({index: row.to_numpy()})
        # if self.verbose:
        #     print(f'Points: {self.points}')
        
        if radius is None:
            combs = combinations(list(self.points.values()), 2)
            dist = list(map(lambda x: self.metric(*x), list(combs)))
            self.radius = mean(dist)/2
            if verbose:
                print('R parameter was automaticly calculated.')
                print(f'R = {self.radius}')
        else:
            self.radius = radius

    def __dist(self, point_1, point_2):
        """
        Function for distance calculation between objects
        """
    
        return self.metric(point_1, point_2)

    def __in_cluster(self, center, point):
        """
        Returns true if object in cluster
        """
    
        return self.metric(center, point) <= self.radius 

    def __get_neighbors(self, p, points):
        """
        Function to find objects in a radius
        """
        neighbors = [point for point in points if self.__in_cluster(p, point)]
        return np.array(neighbors)

    def __get_centroid(self, points):
        """
        Function for center of mass calculation
        """
        return np.mean(points, axis=0)

    def __get_random_point(self, points):
        """
        Function for getting random object
        """
        random_index = np.random.choice(len(points), 1)[0]
        return points[random_index]

    def __remove_points(self, subset, points):
        """
        Function for objects list filtering
        """
        subset = [list(i) for i in subset]
        points = [p for p in points if list(p) not in subset]
        return points

    def __get_centroids(self, tol=1e-5):
        """
        Function with FOREL algorithm
        """
        self.centroids = []
        points = list(self.points.values())
        while len(points) != 0:
            current_point = self.__get_random_point(points)
            neighbors = self.__get_neighbors(current_point, points)
            centroid = self.__get_centroid(neighbors)
            while self.__dist(current_point, centroid) > tol:
                current_point = centroid
                neighbors = self.__get_neighbors(current_point, points)
                centroid = self.__get_centroid(neighbors)
            points = self.__remove_points(neighbors, points)
            self.centroids.append(current_point)

    def __cluster_mapping(self, point):
        """
        Function mapping point and cluster
        """
        for i in range(len(self.centroids)):
            if self.__in_cluster(self.centroids[i], point):
                return f"cluster {i+1}", self.centroids[i]

    def __detect_clusters(self):
        """
        Returns df with resulting clusters
        """
        
        df = self.data.copy()
        df['point'] = list(self.points.values()) 
        df['cluster'], df['cluster_center'] = zip(*df.point.apply(lambda x: self.__cluster_mapping(x)))
        return df

    def __visualise(self):
        """
        Function for clusters visualisation
        """
        pd.plotting.parallel_coordinates(self.__detect_clusters().drop(['point', 'cluster_center'], axis = 1), 'cluster')

    def get_clusters(self):
        self.__get_centroids()
        df = self.__detect_clusters()
        if self.verbose:
            self.__visualise()
        return df

def Kmeans(data, num_clusters):
    from sklearn.cluster import Kmeans

    cluster = Kmeans(n_clusters=num_clusters)
    model = cluster.fit(data)
    data['cluster'] = model.labels_
    return data