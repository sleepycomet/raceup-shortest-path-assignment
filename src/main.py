from track_representation import build_track_mesh, get_track_plot
from find_shortest_path import dijkstra_path, get_shortest_path_on_track_plot

import matplotlib.pyplot as plt


def main():
    inner_radius, outer_radius = 7, 10
    h, k = 24, 128
    mesh = build_track_mesh(inner_radius, outer_radius, h, k)

    shortest_path_spatial, cost = dijkstra_path(mesh, source_idx=h-1, dest_idx=k*h-1)
    
    _, ax = get_track_plot(mesh)
    ax = get_shortest_path_on_track_plot(shortest_path_spatial, cost, ax)

    plt.title("Shortest Path between Outside Corners on a Semicircular Track [meters]")
    plt.show()


if __name__ == "__main__":
    main()
