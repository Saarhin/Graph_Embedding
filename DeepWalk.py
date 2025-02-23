import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from gensim.models import Word2Vec


def generate_random_walk(start_node, walk_length, graph):
    walk_sequence = [str(start_node)]

    for _ in range(walk_length):
        neighbours = [neighbour_node for neighbour_node in graph.neighbors(start_node)]
        next_node = np.random.choice(neighbours, 1)[0]
        walk_sequence.append(str(next_node))
        start_node = next_node

    return walk_sequence

G = nx.erdos_renyi_graph(25, 0.25, directed=False)


nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0))



sequence = []
for x in  range(100):
    sequence.append(generate_random_walk(np.random.choice(25),15,G))

#print(sequence[:3])

skip_gram = Word2Vec(sequence, vector_size=100, window=2, sg=1, min_count=1)
skip_gram.train(sequence, total_examples=skip_gram.corpus_count, epochs=30, report_delay=1)


print('Nodes that are similar to noe 7:')

for similarity in skip_gram.wv.most_similar(positive=['7']):
    print(similarity)


plt.show()