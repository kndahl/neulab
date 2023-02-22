import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def graph_clusterize(vectors, metric='euclidean', threshold=None, draw=False, figsize=(10, 10)):
    """
    Clusters input vectors using Graph algorithm in NetworkX.
    Parameters:
        - vectors: list of input vectors to be clustered
        - metric: distance metric to be used (default is Euclidean distance)
        - threshold: distance threshold for considering two vectors to be in the same cluster
        - draw: plot results if True
        - figsize: plot size
    Returns:
        - clusters: a list of clusters, where each cluster is a list of vector indices
    """
    # Compute pairwise distances between vectors
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if metric == 'euclidean':
                dist = np.linalg.norm(vectors[i] - vectors[j])
            elif metric == 'cosine':
                dist = 1 - np.dot(vectors[i], vectors[j])/(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]))
            else:
                raise ValueError('Unsupported distance metric')
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    
    # Create graph and add edges between vectors below threshold distance
    G = nx.Graph()
    X = nx.Graph()
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if threshold is None or dist_matrix[i,j] <= threshold:
                X.add_edge(i, j, weight=dist_matrix[i,j])
                # Add edge
                if vectors[i][-1] == vectors[j][-1]:
                    G.add_edge(i, j, weight=dist_matrix[i,j])
    
    # Extract clusters as connected components of the graph
    clusters = list(nx.connected_components(G))
    
    if draw == True:
        
        # Draw Graph
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax = axes.flatten()

        # Main
        edge_labels = dict( ((u, v), round(d["weight"])) for u, v, d in X.edges(data=True))
        pos = nx.spring_layout(X)
        nx.draw(X, pos, node_size=600, node_color='pink', alpha=0.9,labels={round(node, 2): node for node in X.nodes()}, ax=ax[0])
        nx.draw_networkx_edge_labels(X, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[0])
        ax[0].set_title("Graph")

        # Clusters
        plt.title('Clusters')
        labels = {i: f"{i}" for i in range(len(vectors))}
        labels = {k: v for k, v in labels.items() if k in pos}
        edge_labels = nx.get_edge_attributes(G, 'weight')
        for i, cluster in enumerate(clusters):
            nx.draw_networkx_nodes(G, pos, nodelist=list(cluster), node_color=f"C{i}", label=f"Cluster {i}")
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_size=8)
        plt.legend()
        plt.show()
    
    return clusters


def mst_clusterize(vectors, num_clusters=2, metric='euclidean', draw=False, figsize=(10, 10)):
    """
    Clusters input vectors using minimum spanning tree algorithm in NetworkX.
    Parameters:
        - vectors: list of input vectors to be clustered
        - num_clusters: number of clusters to form
        - metric: distance metric to be used (default is Euclidean distance)
        - draw: plot results if True
        - figsize: plot size
    Returns:
        - clusters: a list of clusters, where each cluster is a list of vector indices
    """
    # Compute pairwise distances between vectors
    dist_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            if metric == 'euclidean':
                dist = np.linalg.norm(vectors[i] - vectors[j])
            elif metric == 'cosine':
                dist = 1 - np.dot(vectors[i], vectors[j])/(np.linalg.norm(vectors[i])*np.linalg.norm(vectors[j]))
            else:
                raise ValueError('Unsupported distance metric')
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    
    # Create graph and add edges between vectors based on distance
    G = nx.Graph()
    X = nx.Graph()
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            X.add_edge(i, j, weight=dist_matrix[i,j])
            if vectors[i][-1] == vectors[j][-1]:
                G.add_edge(i, j, weight=dist_matrix[i,j])
    
    # Compute minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    MST = nx.minimum_spanning_tree(X)

    # Extract clusters as num_clusters largest connected components of the graph
    clusters = sorted(list(nx.connected_components(T)), key=len, reverse=True)[:num_clusters]
    
    if draw == True:
        # Draw minimum spanning tree
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        ax = axes.flatten()

        edge_labels = dict( ((u, v), round(d["weight"], 3)) for u, v, d in MST.edges(data=True))
        pos = nx.spring_layout(MST)
        nx.draw(MST, pos, node_size=600, node_color='pink', alpha=0.9, labels={round(node, 2): node for node in MST.nodes()}, ax=ax[0])
        nx.draw_networkx_edge_labels(MST, pos, edge_labels=edge_labels, font_color='red', font_size=10, ax=ax[0])
        ax[0].set_title("Minimum Spanning Tree")

        # Draw clusters
        plt.sca(ax[1])
        plt.title('Clusters')
        pos = nx.spring_layout(T)
        labels = {i: f"{i}" for i in range(len(vectors))}
        labels = {k: v for k, v in labels.items() if k in pos}
        for i, cluster in enumerate(clusters):
            nx.draw_networkx_nodes(G, pos, nodelist=list(cluster), node_color=f"C{i}", label=f"Cluster {i}")
            nx.draw_networkx_labels(G, pos, labels, font_size=10)
        nx.draw_networkx_edges(G, pos)
        plt.legend()
    
    return clusters

# class Forel:
#     """
#     A class representing FOREL clustering algorithm. 
#     The algorithm itself determines the number of clusters. 

#     Example:
#         from neulab.Clustering import Forel
#         import pandas as pd

#         df = pd.read_csv('tests/csv/iris.csv').drop('Name', axis = 1)
#         forel = Forel(data=df, verbose=True, scale = True, radius = 60)
#         cluster = forel.get_clusters()

#     """

#     def __init__(self, data, radius=None, metric=euclidean_distance, scale=False, verbose=False):

#         """
#         Init function
        
#         Parameter data: pandas.DataFrame with objects to cluster
        
#         Parameter radius: search radius for local clusters
#         Precondition: radius equals mean distance between objects divided by 2

#         Parameter metric: metric function used to calculate distance between objects
#         Precondition: neulab.Algorithms.euclidean_distance

#         Parameter scale: If true all object parametres scaled in [0, 100]
#         Precondition: False

#         Parameter verbose: Show additional info
#         Precondition: False

#         """

#         self.metric = metric
#         self.scale = scale
#         if self.scale:
#             min_max_scaler = MinMaxScaler((0,100))
#             self.data = pd.DataFrame(min_max_scaler.fit_transform(data), columns = data.columns)
#         else:
#             self.data = data.copy()
#         self.verbose = verbose
#         self.points = {}
    
#         for index, row in self.data.iterrows():
#             self.points.update({index: row.to_numpy()})
#         # if self.verbose:
#         #     print(f'Points: {self.points}')
        
#         if radius is None:
#             combs = combinations(list(self.points.values()), 2)
#             dist = list(map(lambda x: self.metric(*x), list(combs)))
#             self.radius = mean(dist)/2
#             if verbose:
#                 print('R parameter was automaticly calculated.')
#                 print(f'R = {self.radius}')
#         else:
#             self.radius = radius

#     def __dist(self, point_1, point_2):
#         """
#         Function for distance calculation between objects
#         """
    
#         return self.metric(point_1, point_2)

#     def __in_cluster(self, center, point):
#         """
#         Returns true if object in cluster
#         """
    
#         return self.metric(center, point) <= self.radius 

#     def __get_neighbors(self, p, points):
#         """
#         Function to find objects in a radius
#         """
#         neighbors = [point for point in points if self.__in_cluster(p, point)]
#         return np.array(neighbors)

#     def __get_centroid(self, points):
#         """
#         Function for center of mass calculation
#         """
#         return np.mean(points, axis=0)

#     def __get_random_point(self, points):
#         """
#         Function for getting random object
#         """
#         random_index = np.random.choice(len(points), 1)[0]
#         return points[random_index]

#     def __remove_points(self, subset, points):
#         """
#         Function for objects list filtering
#         """
#         subset = [list(i) for i in subset]
#         points = [p for p in points if list(p) not in subset]
#         return points

#     def __get_centroids(self, tol=1e-5):
#         """
#         Function with FOREL algorithm
#         """
#         self.centroids = []
#         points = list(self.points.values())
#         while len(points) != 0:
#             current_point = self.__get_random_point(points)
#             neighbors = self.__get_neighbors(current_point, points)
#             centroid = self.__get_centroid(neighbors)
#             while self.__dist(current_point, centroid) > tol:
#                 current_point = centroid
#                 neighbors = self.__get_neighbors(current_point, points)
#                 centroid = self.__get_centroid(neighbors)
#             points = self.__remove_points(neighbors, points)
#             self.centroids.append(current_point)

#     def __cluster_mapping(self, point):
#         """
#         Function mapping point and cluster
#         """
#         for i in range(len(self.centroids)):
#             if self.__in_cluster(self.centroids[i], point):
#                 return f"cluster {i+1}", self.centroids[i]

#     def __detect_clusters(self):
#         """
#         Returns df with resulting clusters
#         """
        
#         df = self.data.copy()
#         df['point'] = list(self.points.values()) 
#         df['cluster'], df['cluster_center'] = zip(*df.point.apply(lambda x: self.__cluster_mapping(x)))
#         return df

#     def __visualise(self):
#         """
#         Function for clusters visualisation
#         """
#         pd.plotting.parallel_coordinates(self.__detect_clusters().drop(['point', 'cluster_center'], axis = 1), 'cluster')

#     def get_clusters(self):
#         self.__get_centroids()
#         df = self.__detect_clusters()
#         if self.verbose:
#             self.__visualise()
#         return df

# def Kmeans(data, num_clusters):
#     from sklearn.cluster import Kmeans

#     cluster = Kmeans(n_clusters=num_clusters)
#     model = cluster.fit(data)
#     data['cluster'] = model.labels_
#     return data