import sys
sys.path.append('.')
from neulab.Clustering import *
import pandas as pd

def test_CG():
    d = {'Age': [18, 33, 42, 24, 19, 25], 'Sex': [0, 1, 1, 0, 2, 2]}
    df = pd.DataFrame(data=d, index=['A', 'B', 'C', 'D', 'E', 'F'])

    assert CGraph(df, metric='manhattan', r='std', rnd=3, draw=False, info=False) in [[['A', 'D'], ['C', 'B'], ['E', 'F']], [['D', 'A'], ['C', 'B'], ['E', 'F']],
    [['A', 'D'], ['B', 'C'], ['E', 'F']], [['A', 'D'], ['C', 'B'], ['F', 'E']]]

def test_MST():
    d = {'Age': [18, 33, 42, 24, 19, 25], 'Sex': [0, 1, 1, 0, 2, 2]}
    df = pd.DataFrame(data=d, index=['A', 'B', 'C', 'D', 'E', 'F'])

    assert CGraphMST(df, clst_num=3, metric='manhattan', rnd=3, draw=False, info=False) in [[['A', 'D'], ['C', 'B'], ['E', 'F']], [['D', 'A'], ['C', 'B'], ['E', 'F']],
    [['A', 'D'], ['B', 'C'], ['E', 'F']], [['A', 'D'], ['C', 'B'], ['F', 'E']]]