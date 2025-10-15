import numpy as np
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def _build_adjacency_matrix(mesh):
    k, h = mesh.shape[1:]
    adj_matrix = np.zeros((h*k , h*k))

    for theta_idx, radius_idx in np.ndindex(mesh.shape[1:]):
        # Current index, flattened for adjacency matrix entry
        adj_idx = h*theta_idx + radius_idx

        ## Create Entries ##
        if radius_idx != h-1:  # Not on outer edge (covers all radial)

            # Radial outward motion
            outward = mesh[:,theta_idx,radius_idx+1] - mesh[:,theta_idx,radius_idx]
            adj_matrix[adj_idx,adj_idx+1] = np.linalg.norm(outward)
            
        if theta_idx != k-1:  # Not at end of track segment (prereq for forward/diagonal)
            
            # Inner diagonal motion
            if radius_idx != 0:
                inner_diag = mesh[:,theta_idx+1,radius_idx-1] - mesh[:,theta_idx,radius_idx]
                adj_matrix[adj_idx , adj_idx + h-1] = np.linalg.norm(inner_diag)
                
            # Direct forward motion
            forward = mesh[:,theta_idx+1,radius_idx] - mesh[:,theta_idx,radius_idx]
            adj_matrix[adj_idx , adj_idx + h] = np.linalg.norm(forward)
            
            # Outer diagonal motion
            if radius_idx != h-1:
                outer_diag = mesh[:,theta_idx+1,radius_idx+1] - mesh[:,theta_idx,radius_idx]
                adj_matrix[adj_idx , adj_idx + h+1] = np.linalg.norm(outer_diag)

    # Mirror upper triangular to lower triangular for accurate bidirectional adjacency matrix
    adj_matrix = csr_matrix(adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix)))

    return adj_matrix


def dijkstra_path(mesh, source_idx: int, dest_idx: int):
    # Construct adjacency matrix
    adj_matrix = _build_adjacency_matrix(mesh)

    # Do Dijkstra's, get shortest path length to dest
    dist_matrix, predecessors = shortest_path(adj_matrix, directed=False, indices=source_idx, return_predecessors=True)
    cost = dist_matrix[dest_idx]

    # Reconstruct the shortest path
    curr_node = dest_idx
    shortest_path_idxs = []
    while curr_node != -9999:
        shortest_path_idxs = [curr_node] + shortest_path_idxs
        curr_node = int(predecessors[curr_node])
    
    # Reconstruct (theta, radius) mesh indices from adjacency matrix indices
    h = mesh.shape[2]
    shortest_path_theta_idxs = [i//h for i in shortest_path_idxs]
    shortest_path_radius_idxs = [i%h for i in shortest_path_idxs]

    # Get the list of coordinates corresponding to the nodes along the shortest path
    shortest_path_spatial = mesh[ : , shortest_path_theta_idxs , shortest_path_radius_idxs ]

    return shortest_path_spatial, cost


def a_star_path(mesh, source_idx: int, dest_idx: int):
    pass


def get_shortest_path_on_track_plot(shortest_path_spatial, cost, ax):
    ax.scatter(shortest_path_spatial[0], shortest_path_spatial[1], color='r', marker='.')
    ax.plot(shortest_path_spatial[0], shortest_path_spatial[1], color='r', label=f"Optimal Path ({cost} m)")

    return ax


# Running this is identical to running main
if __name__ == "__main__":
    from track_representation import *

    inner_radius, outer_radius = 7, 10
    h, k = 24, 128
    mesh = build_track_mesh(inner_radius, outer_radius, h, k)

    shortest_path_spatial, cost = dijkstra_path(mesh, source_idx=h-1, dest_idx=k*h-1)
    
    _, ax = get_track_plot(mesh)
    ax = get_shortest_path_on_track_plot(shortest_path_spatial, cost, ax)

    plt.title("Shortest Path between Outside Corners on a Semicircular Track [meters]")
    plt.legend()
    plt.show()
