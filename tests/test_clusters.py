import sys
sys.path.append('.')
from neulab.clusters import graph_clusterize
from neulab.clusters import mst_clusterize
import numpy as np

def test_graph_clusterize():
    vectors = [
    np.array([25, 0]),  # 25-year-old female    #0
    np.array([35, 1]),  # 35-year-old male      #1  
    np.array([45, 0]),  # 45-year-old female    #2
    np.array([55, 1]),  # 55-year-old male      #3
    np.array([65, 0]),  # 65-year-old female    #4
    np.array([75, 1]),  # 75-year-old male      #5
    np.array([41, 1]),  # 41-year-old male      #6
    np.array([24, 0]),  # 24-year-old female    #7
    np.array([18, 0]),  # 18-year-old female    #8
    np.array([17, 1]),  # 17-year-old male      #9
    ]
    clusters = graph_clusterize(vectors, metric='euclidean', threshold=20)
    assert clusters == [{0, 2, 4, 7, 8}, {1, 3, 5, 6, 9}]

def test_mst_clusterize():
    vectors = [
        np.array([18, 0]),  # 18-year-old female    #0
        np.array([33, 1]),  # 33-year-old male      #1  
        np.array([42, 1]),  # 42-year-old male      #2
        np.array([24, 0]),  # 24-year-old female    #3
        np.array([19, 2]),  # 19-year-old unknown   #4
        np.array([25, 2]),  # 25-year-old unknown   #5
    ]

    clusters = mst_clusterize(vectors, num_clusters=3, metric='cosine')
    assert clusters == [{0, 3}, {1, 2}, {4, 5}]