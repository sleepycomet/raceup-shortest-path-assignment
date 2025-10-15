import numpy as np
import heapq

from scipy.sparse.csgraph import shortest_path


def dijkstra_path(adj_matrix, source_idx: int, dest_idx: int):
    # Do Dijkstra's, get shortest path length to dest
    dist_matrix, predecessors = shortest_path(adj_matrix, directed=False, indices=source_idx, return_predecessors=True)
    cost = dist_matrix[dest_idx]

    # Reconstruct the shortest path
    current_idx = dest_idx
    shortest_path_idxs = []
    while current_idx != -9999:
        shortest_path_idxs = [current_idx] + shortest_path_idxs
        current_idx = int(predecessors[current_idx])

    return shortest_path_idxs, cost


def _reconstruct_a_star_path(came_from: dict, current_idx: int):
    shortest_path_idxs = [current_idx]

    while current_idx in came_from.keys():
        current_idx = came_from[current_idx]
        shortest_path_idxs = [current_idx] + shortest_path_idxs

    return shortest_path_idxs


def a_star_path(adj_matrix, source_idx: int, dest_idx: int):
    came_from = {}

    def cost(current_idx: int, other_idx: int):
        return adj_matrix[current_idx, other_idx]

    def heuristic(current_idx: int):
        return cost(current_idx, dest_idx)

    # Shortest known path cost from source to node
    g_score = {source_idx: 0}

    # f_score[node] = g_score[node] + heuristic(node) ; current estimate of cost from node -> dest
    f_score = {source_idx: heuristic(source_idx)}

    # Defining the open set entries as ( priority:=f_score(node) , node ) for heapq ordering on f_score
    open_set = [(f_score[source_idx], source_idx)]

    while open_set:
        current_idx = heapq.heappop(open_set)[1]  # Get and remove node with lowest f_score
        if current_idx == dest_idx:
            shortest_path_idxs = _reconstruct_a_star_path(came_from, current_idx)
            return shortest_path_idxs, g_score[current_idx]

        neighbors = adj_matrix[current_idx].coords[0]
        for neighbor in neighbors:
            tentative_g_score = g_score[current_idx] + cost(current_idx, neighbor)

            if tentative_g_score < g_score.get(neighbor, np.inf):
                came_from[neighbor] = current_idx
                g_score[neighbor] = tentative_g_score
                old_f_score = f_score.get(neighbor)
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)
                if (old_f_score, neighbor) in open_set:
                    print(old_f_score)
                    idx = open_set.index((old_f_score, neighbor))
                    open_set[idx] = (f_score, neighbor)
                    heapq.heapify(open_set)
                else:
                    heapq.heappush(open_set, (f_score, neighbor))

    return None, None


def get_shortest_path_on_track_plot(shortest_path_spatial, cost, ax):
    ax.scatter(shortest_path_spatial[0], shortest_path_spatial[1], color='r', marker='.')
    ax.plot(shortest_path_spatial[0], shortest_path_spatial[1], color='r', label=f"Optimal Path ({cost} m)")

    return ax


if __name__ == "__main__":
    from track_representation import *

    inner_radius, outer_radius = 7, 10
    h, k = 24, 128
    mesh = build_track_mesh(inner_radius, outer_radius, h, k)

    adj_matrix = build_adjacency_matrix(mesh)
    
    # As it would happen, my implementation of A* from the wikipedia pseudocode
    # is SUBSTANTIALLY (x1200!!) slower than just using scipy's shortest path.

    # shortest_path_idxs, cost = a_star_path(adj_matrix, source_idx=h-1, dest_idx=k*h-1)

    shortest_path_idxs, cost = dijkstra_path(adj_matrix, source_idx=h-1, dest_idx=k*h-1)

    # Reconstruct (theta, radius) mesh indices from adjacency matrix indices in shortest path
    h = mesh.shape[2]
    shortest_path_theta_idxs = [i//h for i in shortest_path_idxs]
    shortest_path_radius_idxs = [i%h for i in shortest_path_idxs]

    # Get the list of coordinates corresponding to the nodes along the shortest path
    shortest_path_spatial = mesh[ : , shortest_path_theta_idxs , shortest_path_radius_idxs ]
    
    _, ax = get_track_plot(mesh)
    ax = get_shortest_path_on_track_plot(shortest_path_spatial, cost, ax)

    plt.title("Shortest Path between Outside Corners on a Semicircular Track [meters]")
    plt.legend()
    plt.show()
