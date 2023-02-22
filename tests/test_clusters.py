import sys
sys.path.append('.')
from neulab.clusters import graph_clusterize
from neulab.clusters import mst_clusterize
from neulab.clusters import forel_clusterize
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

def test_forel_clusterize():
    vectors = [(7.86771508123449, 5.235102791011277), 
            (6.5670750253990695, 9.087067684099475), 
            (0.4152340708959723, 9.174182538910815), 
            (4.8665882226902095, 1.9831868314593526), 
            (7.5425091509416395, 7.213993476058874), 
            (5.266191108905593, 0.11107598299808563), 
            (4.717771550004307, 8.399979340041007), 
            (9.500878448739446, 7.0511552519333565), 
            (1.4902042826268402, 8.658528318228976), 
            (3.0572531449179796, 6.91710659443622), 
            (9.756331748405758, 8.254757888603669), 
            (7.325594489334932, 9.95086512741726), 
            (9.989036481365442, 7.777703903396976), 
            (7.603875457309648, 2.06314661111331), 
            (8.952699506188871, 7.081667413332172), 
            (7.924379321034759, 5.68104125738329), 
            (4.992796496638821, 3.0098354801745506), 
            (3.6327244153463667, 0.4019247842999796), 
            (2.5460515564994237, 4.619029907162165), 
            (1.3078608092363186, 3.501730713389135), 
            (7.8837822256068515, 6.426819420542), 
            (2.3738148642835344, 7.965539850977919),
            ]

    clusters = forel_clusterize(vectors=vectors, radius=5, draw=False)
    assert clusters == [[0, 1, 3, 4, 7, 10, 11, 12, 14, 15, 20], [2, 6, 8, 9, 21], [5, 13, 16, 17], [18, 19]]