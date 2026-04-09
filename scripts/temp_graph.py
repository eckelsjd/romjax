import os
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

import romjax as rox


def fun1(x):
    return x + 1


def fun1_inv(x):
    return x - 1


if Path(os.getcwd()).name != 'scripts':
    os.chdir('scripts')

fg = rox.YamlLoader.load("graph.yml")
graph = fg.graph()

edge = graph.edges[("v1", "v2")]["object"]
for x in range(5):
    print(f"x: {x}, f(x): {edge(x)}, f-1(f(x)): {edge(edge(x), direction='backward')}")

nx.draw(graph, with_labels=True)
plt.show()