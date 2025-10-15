import numpy as np
from time import time
from find_shortest_path import dijkstra_path, a_star_path
from track_representation import build_track_mesh, build_adjacency_matrix


if __name__ == '__main__':
    K, H = 128, 24
    inner, outer = 7, 10
    source_idx = H-1
    dest_idx = H*K - 1

    mesh = build_track_mesh(inner,outer,H,K)
    adj_matrix = build_adjacency_matrix(mesh)

    dijkstra_time = []
    # a_star_time = []
    iters = 10000
    for _ in range(iters):
        start = time()
        dijkstra_path(adj_matrix, source_idx, dest_idx)
        end = time()
        dijkstra_time.append(end-start)

    # for _ in range(1):
    #     start = time()
    #     a_star_path(adj_matrix, source_idx, dest_idx)
    #     end = time()
    #     a_star_time.append(end-start)

    print(f"Dijkstra Execution Time: {1000*np.mean(dijkstra_time)} ms")
    # print(f"A* Execution Time: {1000*np.mean(a_star_time)} ms")