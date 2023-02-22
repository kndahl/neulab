import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random

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


def forel_clusterize(vectors, radius, draw=False, figsize=(10, 10)):
    """
    Clusters input vectors using Forel algorithm.
    Parameters:
        - vectors: list of input vectors to be clustered
        - radius: distance of an existing centroid, the vector is assigned to the closest cluster.
        - draw: plot results if True
        - figsize: plot size
    Returns:
        - clusters: a list of clusters, where each cluster is a list of vector indices
    """
    def distance(a, b):
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))
    
    def plot_clusters(vectors, centroids, clusters, radius):
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots(figsize=(8, 8))
        for i in range(len(clusters)):
            cluster_color = colors[i % len(colors)]
            cluster_indices = clusters[i]
            cluster_vectors = [vectors[j] for j in cluster_indices]
            centroid = centroids[i]
            circle = plt.Circle(centroid, radius, color=cluster_color, fill=False)
            ax.add_artist(circle)
            ax.annotate('Cluster {}'.format(i), xy=centroid, xytext=(centroid[0] + 0.5, centroid[1] + 0.5))
            plt.scatter([v[0] for v in cluster_vectors], [v[1] for v in cluster_vectors], c=cluster_color)
            plt.scatter(centroid[0], centroid[1], c='k', marker='x', s=100)
        ax.set_aspect('equal', adjustable='box')
        max_val = max(max(v[0], v[1]) for v in vectors)
        ax.set_xlim([0, max_val])
        ax.set_ylim([0, max_val])
        plt.title('Forel Clustering (Radius = {})'.format(radius))
        plt.show()

    centroids = []
    clusters = []

    for i in range(len(vectors)):
        if all(distance(vectors[i], c) > radius for c in centroids):
            centroids.append(vectors[i])
            clusters.append([i])
        else:
            closest_centroid = min(range(len(centroids)), key=lambda j: distance(vectors[i], centroids[j]))
            clusters[closest_centroid].append(i)

    if draw == True:
        plot_clusters(vectors, centroids, clusters, radius=radius)

    return clusters


def kmeans_clusterize(vectors, num_clusters, max_iterations=100, draw=False, figsize=(10, 10)):
    """
    Clusters input vectors using KMeans algorithm.
    Parameters:
        - vectors: list of input vectors to be clustered
        - num_clusters: number of clusters to create
        - max_iterations: maximum number of iterations for convergence
        - draw: plot results if True
        - figsize: plot size
    Returns:
        - clusters: a list of clusters, where each cluster is a list of vector indices
    """
    def distance(a, b):
        return math.sqrt(sum((a[i] - b[i])**2 for i in range(len(a))))

    def plot_clusters(vectors, centroids, clusters):
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        for i, c in enumerate(clusters):
            color = colors[i % len(colors)]
            plt.scatter([vectors[j][0] for j in c], [vectors[j][1] for j in c], c=color)
        plt.title('K-Means Clustering (k={})'.format(len(clusters)))
        plt.show()

    # Initialize centroids randomly
    centroids = random.sample(list(vectors), num_clusters)
    # Initialize clusters to empty lists
    clusters = [[] for _ in range(num_clusters)]

    for i in range(max_iterations):
        # Assign each vector to the nearest cluster
        for j in range(len(vectors)):
            distances = [distance(vectors[j], centroid) for centroid in centroids]
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(j)

        # Update centroids based on the new clusters
        for k in range(num_clusters):
            if len(clusters[k]) > 0:
                new_centroid = np.mean([vectors[j] for j in clusters[k]], axis=0)
                centroids[k] = new_centroid
            clusters[k] = []

    # Assign each vector to its final cluster
    for j in range(len(vectors)):
        distances = [distance(vectors[j], centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(j)

    if draw == True:
        plot_clusters(vectors, centroids, clusters)

    return clusters