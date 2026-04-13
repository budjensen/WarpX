import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_field(field_data, time_steps, z_grid):
    """Animate the evolution of a field over time."""
    fig, ax = plt.subplots()
    line, = ax.plot(z_grid, field_data[0], color='blue')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Field Value')
    ax.set_title('Field Evolution Over Time')
    ax.set_xlim(z_grid[0], z_grid[-1])
    ax.set_ylim(np.min(field_data), np.max(field_data))
    title_text = ax.set_title(f"Field Evolution at t={time_steps[0]:.3e} s")

    def update(frame):
        line.set_ydata(field_data[frame])
        title_text.set_text(f'Field Evolution at t={time_steps[frame]:.3e} s')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=len(time_steps), blit=True)

    return ani
