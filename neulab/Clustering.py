from numpy import spacing
from neulab.Algorithms import ManhattanMetric, EuclidMertic, MaxMetric, Median
from neulab.Normalization import InterNormalization
import networkx as nx

def CGraph(df, metric='euclid', r='std', rnd=3, draw=False, info=True):
    ''' Graph based clustering algorithm. The algorithm itself determines the number of clusters. Returns clusters linked in an array.'''
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
                DIST = round(ManhattanMetric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'euclid':
                DIST = round(EuclidMertic(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'max':
                DIST = round(MaxMetric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
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
    from neulab.Algorithms import Mean, Median, StdDeviation
    dist_list = list(distances.values())
    if r == 'std':
        R = round(StdDeviation(vector=dist_list), ROUND)
    if r == 'mean':
        R = round(Mean(vector=dist_list), ROUND)
    if r == 'median':
        R = round(Median(vector=dist_list), ROUND)
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
    '''Graph based clustering algorithm.The user himself sets the number of clusters. Returns clusters linked in an array.'''
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
                DIST = round(ManhattanMetric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'euclid':
                DIST = round(EuclidMertic(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
            if metric == 'max':
                DIST = round(MaxMetric(vector1=df.loc[indx[x]], vector2=df.loc[indx[i+1]]), ROUND)
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