import numpy as np
from scipy import sparse
from sknetwork.ranking import PageRank
from sknetwork.visualization import svg_graph
import sknetwork

#create the graph
def read_graph_from_file(file_path):
    adjacency_list = []
    with open(file_path, 'r') as file:
        for line in file:
            node_id, friend_id = map(int, line.split())
            adjacency_list.append((node_id, friend_id))
    return adjacency_list

# Obtain the graph of a social network.
# Custom graph file path
custom_graph_file = "data/facebook_combined.txt"

# Create the adjacency matrix of this graph.
#we will use a adjacency list to represent the graph
edges = read_graph_from_file(custom_graph_file)

#we're going to convert the adjacency list to an adjacency matrix

# Get the number of nodes
num_nodes = max(max(edge) for edge in edges) + 1


# convert adjacency list to adjacency matrix
adjacency_matrix = np.zeros((num_nodes, num_nodes))
for edge in edges:
    adjacency_matrix[edge[0], edge[1]] = 1
    adjacency_matrix[edge[1], edge[0]] = 1

# Define the corresponding transition matrix (Markov Chain) P using the adjacency matrix & sknetwork
# Convert the adjacency matrix to a sparse matrix
adjacency_matrix_sparse = sparse.csr_matrix(adjacency_matrix)
# Create the transition matrix
transition_matrix = adjacency_matrix_sparse / adjacency_matrix_sparse.sum(axis=1)
# Define the damping factor
damping_factor = 0.85
# Create the PageRank object
pagerank = PageRank(damping_factor)
# Fit the model
pagerank.fit(transition_matrix)
# Get the scores
scores = pagerank.scores_
#we take the position matrix from the graph
position = sknetwork.embedding.Spring()
position.fit(adjacency_matrix_sparse)


# Visualize the graph
image = svg_graph(adjacency_matrix, position, scores=scores)

# Save the image
with open('graph.svg', 'w') as file:
    file.write(image)
    


# Define the matrix A=alpha.P+ (1-alpha). G with alpha in [0,1].
# Compute / implement the stationary distribution of the virus and normalize it.
# Implement the spread epidemic without vaccination, with random vaccination and vaccination based on infection vector issued from step 5.
# Experiment and present the performances of step 6 including stochastic simulation using the infection vector.

print("Number of nodes:", num_nodes)

# Convert edge list to adjacency matrix


