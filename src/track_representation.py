import numpy as np
from scipy.sparse import csr_array

import matplotlib.pyplot as plt


def build_track_mesh(inner_radius, outer_radius, h, k):
    thetas = np.linspace(0, np.pi, k)
    radii = np.linspace(inner_radius, outer_radius, h)

    def arc(thetas, radius):
        return radius * np.array([np.cos(thetas),np.sin(thetas)])
    
    mesh = np.zeros((2,k,h))
    for i, radius in enumerate(radii):
        mesh[:,:,i] = arc(thetas, radius)
    
    return mesh


def build_adjacency_matrix(mesh):
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
    adj_matrix = csr_array(adj_matrix + adj_matrix.T - np.diag(np.diag(adj_matrix)))

    return adj_matrix


def get_track_plot(mesh):
    fig, ax = plt.subplots(figsize=(20,20))

    x = np.ravel(mesh[0,:,:])
    y = np.ravel(mesh[1,:,:])

    ax.scatter(x, y, marker='.')

    # Draw radial lines
    for radius_idx in range(mesh.shape[2]):
        x, y = mesh[0, :, radius_idx], mesh[1, :, radius_idx]
        ax.plot(x, y, color='gray', alpha=0.4)

    # Draw arc lines
    for theta_idx in range(mesh.shape[1]):
        x, y = mesh[0, theta_idx, :], mesh[1, theta_idx, :]
        ax.plot(x, y, color='gray', alpha=0.4)
    
    ax.set_aspect('equal')

    return fig, ax

if __name__ == "__main__":
    mesh = build_track_mesh(7, 10, 24, 128)
    
    _, ax = get_track_plot(mesh)
    
    plt.title("Track [meters]")
    plt.show()
