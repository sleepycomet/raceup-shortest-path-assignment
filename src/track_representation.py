import numpy as np

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
