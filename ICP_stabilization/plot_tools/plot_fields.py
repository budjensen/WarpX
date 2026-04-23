import matplotlib.pyplot as plt
import numpy as np

def plot_center_density_v_time(density_data, time_steps):
    """
    Plot the density at the center of the simulation as a function of time.

    Parameters
    ----------
    density_data: array-like
        Density values at each time step at each node.
    time_steps: array-like
        Time values corresponding to each density measurement.
    """
    # Find the center node index (assuming the center is at the middle of the spatial domain)
    center_idx = density_data.shape[1] // 2

    # Extract the center density at each time step
    center_density = density_data[:, center_idx-5:center_idx+5].mean(axis=1)  # Average over a small region around the center for stability

    # Plot the center density over time
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_steps, center_density, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Center Density (m$^{-3}$)')
    ax.set_title('Center Density Evolution Over Time')
    ax.grid()

    return fig, ax

def plot_current_v_time(Jy_data, time_steps, J_icp: callable):
    """
    Plot the current density as a function of time and compare it with the prescribed ICP current density.

    Parameters
    ----------
    Jy_data: array-like
        Y-component of current total density values at each time step at each node.
    time_steps: array-like
        Time values corresponding to each current density measurement.
    J_icp: callable
        Function representing the prescribed ICP current density.
    """
    # Find the top of the ICP region, lowest index from the right where all values above are zero
    top_icp_idx = len(Jy_data[0]) - 1 - np.argmax(np.abs(Jy_data[0][::-1]) > 0.)
    # Find the bottom of the ICP region, lowest index from the left where all values below are zero
    bottom_icp_idx = np.argmax(np.abs(Jy_data[0]) > 0.)
    if top_icp_idx <= bottom_icp_idx:
        raise ValueError("Invalid current density data: no non-zero values found or ICP region not properly defined.")

    # Average the current density within the ICP region at each time step
    Jy = np.average(Jy_data[:, bottom_icp_idx:top_icp_idx + 1], axis=1)

    # Plot the current density and the prescribed ICP current density
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(time_steps, Jy, color='red', label='Actual Current Density')
    ax[0].plot(time_steps, J_icp(time_steps), color='blue', linestyle='--', label='Prescribed Current Density')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Current Density (A/m$^2$)')
    ax[0].set_title('Current Density Evolution Over Time')
    ax[0].grid()
    ax[0].legend()

    # Plot the errors
    error = Jy - J_icp(time_steps)
    ax[1].plot(time_steps, error, color='purple')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Error (A/m$^2$)')
    ax[1].set_title('Error Between Actual and Prescribed Current Density')
    ax[1].grid()

    return fig, ax
