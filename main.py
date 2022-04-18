from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import numpy as np
import os

text_files = [file for file in os.listdir("facebook") if file.endswith('.edges')]
for i in range(len(text_files)):
    text_files[i] =  "facebook/" + text_files[i]
adjacency_matrix = np.zeros((4039,4039))
i = 0
for file_names in text_files:
    with open(file_names) as file:
        for line in file:
            line = line.split()
            adjacency_matrix[int(line[0])][int(line[1])] = 1
            adjacency_matrix[int(line[1])][int(line[0])] = 1
degree_matrix = np.diag(adjacency_matrix.sum(axis=1))

laplacian = degree_matrix - adjacency_matrix

def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx_nodes(gr, pos=nx.spring_layout(gr),node_size=2, node_color=['blue'])
    nx.draw_networkx_edges(gr, pos=nx.spring_layout(gr),width=0.1, alpha=0.5)
    plt.show()
show_graph_with_labels(adjacency_matrix)
# eigenvalues and eigenvectors
vals, vecs = np.linalg.eig(laplacian)

# sort these based on the eigenvalues
vecs = vecs[:,np.argsort(vals)]
vals = vals[np.argsort(vals)]

# kmeans on first three vectors with nonzero eigenvalues
kmeans = KMeans(n_clusters=4)
kmeans.fit(vecs[:,1:4])
colors = kmeans.labels_

print("Clusters:", colors)
