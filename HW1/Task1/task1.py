import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the parametric equations
def x(t):
    return 3 * np.cos(2 * t) * np.cos(t) + 0.82

def y(t):
    return 3 * np.cos(2 * t) * np.sin(t) + 0.82

# Define the derivatives for velocity and acceleration
def dx_dt(t):
    return -3 * (2 * np.sin(2 * t) * np.cos(t) + np.cos(2 * t) * np.sin(t))

def dy_dt(t):
    return 3 * (np.cos(2 * t) * np.cos(t) - 2 * np.sin(2 * t) * np.sin(t))

def d2x_dt2(t):
    return -3 * (4 * np.cos(2 * t) * np.cos(t) - 4 * np.sin(2 * t) * np.sin(t) - np.cos(2 * t) * np.cos(t))

def d2y_dt2(t):
    return 3 * (-4 * np.sin(2 * t) * np.cos(t) - 4 * np.cos(2 * t) * np.sin(t) - np.cos(2 * t) * np.sin(t))

# Time interval
t_values = np.linspace(0, 10, 200)

# Compute x, y, v, a
x_values = x(t_values)
y_values = y(t_values)
vx_values = dx_dt(t_values)
vy_values = dy_dt(t_values)
v_values = np.sqrt(vx_values**2 + vy_values**2)
ax_values = d2x_dt2(t_values)
ay_values = d2y_dt2(t_values)
a_values = np.sqrt(ax_values**2 + ay_values**2)

# Compute tangential and normal acceleration
a_tau_values = (vx_values * ax_values + vy_values * ay_values) / v_values
a_n_values = np.sqrt(a_values**2 - a_tau_values**2)

# Compute curvature
kappa_values = np.abs(vx_values * ay_values - vy_values * ax_values) / (v_values**3)

# Create the figure and axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot trajectory
ax1.set_xlim(min(x_values) - 1, max(x_values) + 1)
ax1.set_ylim(min(y_values) - 1, max(y_values) + 1)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Trajectory and Vectors')

# Initialize quivers for velocity and acceleration
velocity_quiver = ax1.quiver([], [], [], [], color='r', scale=50, label='Velocity v')
acceleration_quiver = ax1.quiver([], [], [], [], color='b', scale=50, label='Acceleration a')
normal_acc_quiver = ax1.quiver([], [], [], [], color='g', scale=50, label='Normal Acceleration a_n')
tangential_acc_quiver = ax1.quiver([], [], [], [], color='m', scale=50, label='Tangential Acceleration a_r')

# Plot the trajectory
trajectory, = ax1.plot([], [], 'k-', label='Trajectory')

# Function to initialize the animation
def init():
    trajectory.set_data([], [])
    velocity_quiver.set_UVC([], [])
    acceleration_quiver.set_UVC([], [])
    normal_acc_quiver.set_UVC([], [])
    tangential_acc_quiver.set_UVC([], [])
    return trajectory, velocity_quiver, acceleration_quiver, normal_acc_quiver, tangential_acc_quiver

# Function to update the animation at each frame
def update(frame):
    # Update trajectory
    trajectory.set_data(x_values[:frame], y_values[:frame])
    
    # Update velocity vector
    velocity_quiver.set_UVC(vx_values[frame], vy_values[frame])
    velocity_quiver.set_offsets([x_values[frame], y_values[frame]])
    
    # Update acceleration vector
    acceleration_quiver.set_UVC(ax_values[frame], ay_values[frame])
    acceleration_quiver.set_offsets([x_values[frame], y_values[frame]])
    
    # Update normal acceleration vector
    normal_acc_quiver.set_UVC(-vy_values[frame] * kappa_values[frame], vx_values[frame] * kappa_values[frame])
    normal_acc_quiver.set_offsets([x_values[frame], y_values[frame]])
    
    # Update tangential acceleration vector
    tangential_acc_quiver.set_UVC(vx_values[frame] * a_tau_values[frame], vy_values[frame] * a_tau_values[frame])
    tangential_acc_quiver.set_offsets([x_values[frame], y_values[frame]])
    
    return trajectory, velocity_quiver, acceleration_quiver, normal_acc_quiver, tangential_acc_quiver

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_values), init_func=init, blit=True, interval=50)

# Add legend
ax1.legend()

# Plot v, a, a_n, a_tau, k
ax2.plot(t_values, v_values, label='Velocity v(t)')
ax2.plot(t_values, a_values, label='Acceleration a(t)')
ax2.plot(t_values, a_n_values, label='Normal Acceleration a_n(t)')
ax2.plot(t_values, a_tau_values, label='Tangential Acceleration a_r(t)')
ax2.plot(t_values, kappa_values, label='Curvature κ(t)')
ax2.set_xlabel('t')
ax2.set_ylabel('Values')
ax2.set_title('Plots of v, a, a_n, a_r, κ')
ax2.legend()

plt.tight_layout()
plt.show()