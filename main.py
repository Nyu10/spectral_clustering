from sklearn.datasets import make_circles
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import numpy as np
import os


#get all .edges files
text_files = [file for file in os.listdir("facebook") if file.endswith('.edges')]
for i in range(len(text_files)):
    text_files[i] =  "facebook/" + text_files[i]
# print(text_files)

adjacency_matrix = np.zeros((4039,4039))
i = 0
for file_names in text_files:
    with open(file_names) as file:
        for line in file:
            line = line.split()
            adjacency_matrix[int(line[0])][int(line[1])] = 1
            adjacency_matrix[int(line[1])][int(line[0])] = 1

# for i in range(len(adjacency_matrix)):
#     for j in range(len(adjacency_matrix)):
#         if (adjacency_matrix[i][j] == 1):
#             print(i,j)

# debug_output = open("debug_output.txt", "w")
# for i in range(len(adjacency_matrix)):
#     for j in range(len(adjacency_matrix)):
#         debug_output.write(np.array2string(adjacency_matrix[i][j]) + " ")
#     debug_output.write("\n")

# ON POSTER
# DEGREE MATRIX IS THE AMOUNT OF FRIENDS
degree_matrix = np.diag(adjacency_matrix.sum(axis=1))
# for i in range(len(degree_matrix)):
#     print(degree_matrix[i][i])

laplacian = degree_matrix - adjacency_matrix
print(laplacian)
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
# debug_output = open("debug_output.txt", "w")
# for i in range(len(laplacian)):
#     for j in range(len(laplacian)):
#         debug_output.write(np.array2string(laplacian[i][j]) + " ")
#     debug_output.write("\n")