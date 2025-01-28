import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Slider

# Constants
R2 = 40      # Radius of Body 2 (pulley)
r2 = 30      # Small radius of Body 2
R3 = 15      # Radius of Body 3
s_M_default = 4  # Default arc length for M

def calculate_kinematics(s_M):
    t_max = np.sqrt(s_M / (4 * R3))  # Corrected formula for t_max
    t = np.linspace(0, t_max, 20)
    x_body1 = 3 + 80 * t**2        # Position of Body 1
    theta2 = 2 * t**2              # Angle of Body 2 (radians)
    theta3 = 4 * t**2              # Angle of Body 3 (radians)
    omega2 = 4 * t                 # Angular velocity of Body 2
    omega3 = 8 * t                 # Angular velocity of Body 3
    v_M = 120 * t                  # Velocity of point M
    a_t = 120                      # Tangential acceleration of M
    a_n = 960 * t**2               # Normal acceleration of M
    return t, x_body1, theta2, theta3, omega2, omega3, v_M, a_t, a_n


# Initialize kinematics with default s_M
t, x_body1, theta2, theta3, omega2, omega3, v_M, a_t, a_n = calculate_kinematics(s_M_default)

# Set up the figure
fig, (ax, ax_plot) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 2]})
plt.subplots_adjust(bottom=0.25, wspace=0.4)

# Animation axis
ax.set_aspect('equal')  # Ensure equal aspect ratio
ax.grid()

# Initialize components
# Body 2 (pulley)
pulley = Circle((0, 0), R2, edgecolor='black', facecolor='none')
ax.add_patch(pulley)

# Body 3 (inside Body 2)
body3 = Circle((0, 0), R3, edgecolor='red', facecolor='none')
ax.add_patch(body3)

# Body 1 (rectangular weight)
body1 = Rectangle((35, -x_body1[0]), 10, 10, facecolor='blue')
ax.add_patch(body1)

# Point M on Body 3 (starts at angle 0)
point_M = Circle((R3, 0), 2, color='green')
ax.add_patch(point_M)

# Velocity and acceleration vectors (scaled for visibility)
quiver_v = ax.quiver([], [], [], [], color='blue', scale=500, label='Velocity')  # Scaled down
quiver_a_t = ax.quiver([], [], [], [], color='green', scale=500, label='Tangential Acceleration')
quiver_a_n = ax.quiver([], [], [], [], color='red', scale=500, label='Normal Acceleration')
ax.legend()

# Plot axis for angular velocities
ax_plot.set_title("Angular Velocities")
ax_plot.set_xlabel("Time (s)")
ax_plot.set_ylabel("Angular Velocity (rad/s)")
ax_plot.set_xlim(0, t[-1])
ax_plot.set_ylim(0, max(max(omega2), max(omega3)) * 1.2)  # Extend y-axis for better scaling
line_omega2, = ax_plot.plot([], [], label=r'$\omega_2$', color='blue')
line_omega3, = ax_plot.plot([], [], label=r'$\omega_3$', color='red')
ax_plot.legend()

# Animation update function
def update(frame):
    # Update Body 1 position
    body1.set_y(-x_body1[frame] - 20)  # Move
    # Update M's position on Body 3
    angle = theta3[frame]
    x_M = R3 * np.cos(angle)
    y_M = R3 * np.sin(angle)
    point_M.center = (x_M, y_M)

    # Update velocity and acceleration vectors
    v_x = -v_M[frame] * np.sin(angle)
    v_y = v_M[frame] * np.cos(angle)
    quiver_v.set_offsets([[x_M, y_M]])
    quiver_v.set_UVC([v_x], [v_y])

    a_t_x = -a_t * np.sin(angle)
    a_t_y = a_t * np.cos(angle)
    quiver_a_t.set_offsets([[x_M, y_M]])
    quiver_a_t.set_UVC([a_t_x], [a_t_y])

    a_n_x = -a_n[frame] * np.cos(angle)
    a_n_y = -a_n[frame] * np.sin(angle)
    quiver_a_n.set_offsets([[x_M, y_M]])
    quiver_a_n.set_UVC([a_n_x], [a_n_y])

    # Update angular velocity plots
    line_omega2.set_data(t[:frame+1], omega2[:frame+1])
    line_omega3.set_data(t[:frame+1], omega3[:frame+1])

    return body1, point_M, quiver_v, quiver_a_t, quiver_a_n, line_omega2, line_omega3

# Set axis limits dynamically for better scaling
ax.set_xlim(-R2 * 1.5, R2 * 1.5)
ax.set_ylim(-200, R2 * 2)  # Extend y-axis

# Create slider for s_M adjustment
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 's_M', 1, 20, valinit=s_M_default, valstep=0.1)

# Slider update function
def slider_update(val):
    global t, x_body1, theta2, theta3, omega2, omega3, v_M, a_t, a_n
    s_M_new = slider.val
    t, x_body1, theta2, theta3, omega2, omega3, v_M, a_t, a_n = calculate_kinematics(s_M_new)
    ax_plot.set_xlim(0, t[-1])
    ax_plot.set_ylim(0, max(max(omega2), max(omega3)) * 1.2)
    ani.event_source.stop()
    ani.event_source.start()

slider.on_changed(slider_update)

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.show()
