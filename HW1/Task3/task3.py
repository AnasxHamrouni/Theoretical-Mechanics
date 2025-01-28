import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Given parameters
y_A_func = lambda t: 22.5 + 10 * np.sin(np.pi / 5 * t)
AB = 45
BC = 30
t_values = np.linspace(0, 10, 200)

# Store positions
A_pos = np.zeros((len(t_values), 2))
B_pos = np.zeros_like(A_pos)
C_pos = np.zeros_like(A_pos)

for i, t in enumerate(t_values):
    y_A = y_A_func(t)
    A_pos[i] = [0, y_A]

    # B moves horizontally while maintaining AB
    x_B = np.sqrt(AB**2 - y_A**2)  # Solve x_B^2 + y_A^2 = AB^2
    B_pos[i] = [x_B, 0]

    # C is a fixed fraction along AB (e.g., 2/3 of the way)
    C_pos[i] = A_pos[i] + (B_pos[i] - A_pos[i]) * (2/3)

# Compute velocities and accelerations using central differences
dt = t_values[1] - t_values[0]
B_vel = np.gradient(B_pos, dt, axis=0)
C_vel = np.gradient(C_pos, dt, axis=0)
B_acc = np.gradient(B_vel, dt, axis=0)
C_acc = np.gradient(C_vel, dt, axis=0)

# Angular velocity of BA
theta = np.arctan2(B_pos[:, 0], A_pos[:, 1] - B_pos[:, 1])
omega = np.gradient(theta, dt)

# Animation setup
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-20, 60)
ax.set_ylim(-20, 50)
ax.grid(True)
ax.set_title('Mechanism Simulation')

# Plot elements
line_AB, = ax.plot([], [], 'bo-', lw=2, markersize=8, label='AB')
line_BC, = ax.plot([], [], 'ro-', lw=2, markersize=8, label='BC')
vec_B = ax.quiver([], [], [], [], color='g', scale=50, width=0.005, label='B Velocity')
vec_C = ax.quiver([], [], [], [], color='m', scale=50, width=0.005, label='C Velocity')
time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
ax.legend()

def init():
    line_AB.set_data([], [])
    line_BC.set_data([], [])
    vec_B.set_UVC(0, 0)
    vec_C.set_UVC(0, 0)
    time_text.set_text('')
    return line_AB, line_BC, vec_B, vec_C, time_text

def update(frame):
    A = A_pos[frame]
    B = B_pos[frame]
    C = C_pos[frame]
    
    # Update links
    line_AB.set_data([A[0], B[0]], [A[1], B[1]])
    line_BC.set_data([B[0], C[0]], [B[1], C[1]])
    
    # Update velocity vectors
    vec_B.set_offsets([B[0], B[1]])
    vec_B.set_UVC(B_vel[frame, 0], B_vel[frame, 1])
    vec_C.set_offsets([C[0], C[1]])
    vec_C.set_UVC(C_vel[frame, 0], C_vel[frame, 1])
    
    time_text.set_text(f'Time: {t_values[frame]:.2f}s')
    return line_AB, line_BC, vec_B, vec_C, time_text

ani = animation.FuncAnimation(fig, update, frames=len(t_values),
                              init_func=init, blit=False, interval=50)

# Plotting velocities, accelerations, and angular velocity
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t_values, np.linalg.norm(B_vel, axis=1), label='B')
plt.plot(t_values, np.linalg.norm(C_vel, axis=1), label='C')
plt.title('Velocity Magnitudes')
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_values, np.linalg.norm(B_acc, axis=1), label='B')
plt.plot(t_values, np.linalg.norm(C_acc, axis=1), label='C')
plt.title('Acceleration Magnitudes')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_values, omega, label='BA Angular Velocity')
plt.title('Angular Velocity of BA')
plt.xlabel('Time (s)')
plt.ylabel('rad/s')
plt.legend()

plt.tight_layout()
plt.show()
