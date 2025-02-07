from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd


# Constants
omega_O1A = 2  # rad/s
phi = np.radians(130)  # Convert degrees to radians
a, b, c = 31, 30, 50
O1A, O2B, O2C, O3D = 15, 30, 30, 50
AB, BC, CD, CE, EF = 40, 16, 60, 30, 30

dt = 0.001  # for numerical differentiation

# Define the links with their endpoints (including fixed points O1, O2, O3)
links = [
    {'name': 'O1A', 'points': ['O1', 'A']},
    {'name': 'AB', 'points': ['A', 'B']},
    {'name': 'O2C', 'points': ['O2', 'C']},
    {'name': 'O2B', 'points': ['O2', 'B']},
    {'name': 'BC', 'points': ['B', 'C']},
    {'name': 'CD', 'points': ['C', 'D']},
    {'name': 'O3D', 'points': ['O3', 'D']},
    {'name': 'CE', 'points': ['C', 'E']},
    {'name': 'EF', 'points': ['E', 'F']},
]

fixed_points = {
    'O1': (0, c),
    'O2': (0,0 ),
    'O3': (a + b, c),
}


def circleintersection(xa, ya, xb, yb, AC, BC):
    d = sqrt((xa - xb)**2 + (ya - yb)**2)

    # No intersection cases
    if d > AC + BC or d < abs(AC - BC):  # No solution
        return None
    if d == 0 and AC == BC:  # Infinite solutions
        return None
    
    # Compute intersection points
    a = (AC**2 - BC**2 + d**2) / (2 * d)
    h = sqrt(AC**2 - a**2)
    x3 = xa + a * (xb - xa) / d
    y3 = ya + a * (yb - ya) / d

    sol1 = [x3 + h * (yb - ya) / d, y3 - h * (xb - xa) / d]
    sol2 = [x3 - h * (yb - ya) / d, y3 + h * (xb - xa) / d]

    # Return max solution
    return max(sol1, sol2)

def xA(t):
    return O1A * cos(omega_O1A * t + phi)

def yA(t):
    return (O1A * sin(omega_O1A * t + phi)) + c

def xF(t):
    return a

def yF(t):
    xe = xE(t)
    ye = yE(t)
    return (2*ye+sqrt(4*ye**2 - 4*(ye**2 + a**2 -2*a*xe + xe**2 - EF**2)))/2

def xE(t):
    xc = xC(t)
    xd = xD(t)
    return (xd + xc)/2

def yE(t):
    yc = yC(t)
    yd = yD(t)
    return (yd + yc)/2

def xB(t):
    xa, ya = xA(t), yA(t)
    xo, yo = fixed_points['O2']
    intersection = circleintersection(xa, ya, xo, yo, AB, O2B)
    return intersection[0]

def yB(t):
    xa, ya = xA(t), yA(t)
    xo, yo = fixed_points['O2']
    intersection = circleintersection(xa, ya, xo, yo, AB, O2B)
    return intersection[1] 

def xC(t):
    xb, yb = xB(t), yB(t)
    if xb is None or yb is None:
        return None
    xo, yo = fixed_points['O2']
    intersection = circleintersection(xb, yb, xo, yo, BC, O2C)
    return intersection[0] 

def yC(t):
    xb, yb = xB(t), yB(t)
    if xb is None or yb is None:
        return None
    xo, yo = fixed_points['O2']
    intersection = circleintersection(xb, yb, xo, yo, BC, O2C)
    return intersection[1] 

def xD(t):
    xc, yc = xC(t), yC(t)
    if xc is None or yc is None:
        return None
    xo, yo = fixed_points['O3']
    intersection = circleintersection(xc, yc, xo, yo, CD, O3D)
    return intersection[0] 

def yD(t):
    xc, yc = xC(t), yC(t)
    if xc is None or yc is None:
        return None
    xo, yo = fixed_points['O3']
    intersection = circleintersection(xc, yc, xo, yo, CD, O3D)
    return intersection[1] 

def get_position(t, point):
    if point in fixed_points:
        return fixed_points[point]
    elif point == 'A':
        return (xA(t), yA(t))
    elif point == 'B':
        return (xB(t), yB(t))
    elif point == 'C':
        return (xC(t), yC(t))
    elif point == 'D':
        xd, yd = xD(t), yD(t)
        return (xd, yd) 
    elif point == 'E':
        xe, ye = xE(t), yE(t)
        return (xe, ye) 
    elif point == 'F':
        return (xF(t), yF(t))
    else:
        raise ValueError(f"Unknown point {point}")

def compute_velocities(t):
    velocities = {}
    for point in ['A', 'B', 'C', 'D', 'E', 'F']:
        if point in fixed_points:
            velocities[point] = (0.0, 0.0)
            continue
        x_plus, y_plus = get_position(t + dt, point)
        x_minus, y_minus = get_position(t - dt, point)
        if x_plus is None or x_minus is None:
            vx = 0.0
        else:
            vx = (x_plus - x_minus) / (2 * dt)
        if y_plus is None or y_minus is None:
            vy = 0.0
        else:
            vy = (y_plus - y_minus) / (2 * dt)
        velocities[point] = (vx, vy)
    return velocities

def compute_angular_velocity(link, velocities, t):
    pt1, pt2 = link['points']
    if pt1 in fixed_points:
        x1, y1 = fixed_points[pt1]
        vx1, vy1 = 0.0, 0.0
    else:
        x1, y1 = get_position(t, pt1)
        vx1, vy1 = velocities.get(pt1, (0.0, 0.0))
    if pt2 in fixed_points:
        x2, y2 = fixed_points[pt2]
        vx2, vy2 = 0.0, 0.0
    else:
        x2, y2 = get_position(t, pt2)
        vx2, vy2 = velocities.get(pt2, (0.0, 0.0))
    dx = x2 - x1
    dy = y2 - y1
    dvx = vx2 - vx1
    dvy = vy2 - vy1
    denominator = dx**2 + dy**2
    if denominator == 0:
        return 0.0
    omega = (dx * dvy - dy * dvx) / denominator
    return omega

def compute_angular_acceleration(link, t):
    omega_plus = compute_angular_velocity(link, compute_velocities(t + dt), t + dt)
    omega_minus = compute_angular_velocity(link, compute_velocities(t - dt), t - dt)
    alpha = (omega_plus - omega_minus) / (2 * dt)
    return alpha

# Animation setup
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-50, 150)
ax.set_ylim(-100, 100)
ax.set_aspect('equal')
ax.grid(True)

# Draw fixed points and label them
for pt, (x, y) in fixed_points.items():
    ax.plot(x, y, 'ko', markersize=8)
    ax.text(x + 2, y + 2, pt, fontsize=10, color='black')

# Initialize lines for each link
link_lines = {}
for link in links:
    link_lines[link['name']], = ax.plot([], [], 'b-', lw=2)

# Initialize velocity quivers and labels
velocity_quivers = {}
velocity_labels = {}
for point in ['A', 'B', 'C', 'D', 'E', 'F']:
    velocity_quivers[point] = ax.quiver([], [], [], [], color='r', scale=150, scale_units='width', width=0.005)
    velocity_labels[point] = ax.text(0, 0, f'v_{point}', fontsize=8, color='r', visible=False)

# Initialize angular acceleration quivers and labels
alpha_quivers = {}
alpha_labels = {}
for link in links:
    alpha_quivers[link['name']] = ax.quiver([], [], [], [], color='g', scale=500, scale_units='width', width=0.005)
    alpha_labels[link['name']] = ax.text(0, 0, f'α_{link["name"]}', fontsize=8, color='g', visible=False)

# Initialize point labels (non-fixed points)
point_labels = {}
for point in ['A', 'B', 'C', 'D', 'E', 'F']:
    point_labels[point] = ax.text(0, 0, point, fontsize=10, color='blue', visible=False)

# Data collection for plots
plot_data = {
    't': [],
    'B_vx': [], 'B_vy': [],
    'C_vx': [], 'C_vy': [],
    'E_vx': [], 'E_vy': [],
    'F_vx': [], 'F_vy': [],
    'CD_omega': [],
    'O2C_omega': [],
    'O3D_omega': [],
    # Position data for each point
    'A_x': [], 'A_y': [],
    'B_x': [], 'B_y': [],
    'C_x': [], 'C_y': [],
    'D_x': [], 'D_y': [],
    'E_x': [], 'E_y': [],
    'F_x': [], 'F_y': [],
}

def init():
    for link in links:
        link_lines[link['name']].set_data([], [])
    for point in velocity_quivers:
        velocity_quivers[point].set_offsets(np.array([np.nan, np.nan]))
        velocity_labels[point].set_visible(False)
    for link in alpha_quivers:
        alpha_quivers[link].set_offsets(np.array([np.nan, np.nan]))
        alpha_labels[link].set_visible(False)
    for point in point_labels:
        point_labels[point].set_visible(False)
    return list(link_lines.values()) + list(velocity_quivers.values()) + list(alpha_quivers.values()) + list(velocity_labels.values()) + list(alpha_labels.values()) + list(point_labels.values())

def update(frame):
    t = frame
    positions = {point: get_position(t, point) for point in ['A', 'B', 'C', 'D', 'E', 'F']}
    velocities = compute_velocities(t)
    
    # Update link positions
    for link in links:
        pt1, pt2 = link['points']
        x1, y1 = fixed_points[pt1] if pt1 in fixed_points else positions[pt1]
        x2, y2 = fixed_points[pt2] if pt2 in fixed_points else positions[pt2]
        link_lines[link['name']].set_data([x1, x2], [y1, y2])
    
    # Update velocity arrows and labels
    for point in ['A', 'B', 'C', 'D', 'E', 'F']:
        if point in fixed_points:
            continue
        x, y = positions[point]
        vx, vy = velocities.get(point, (0.0, 0.0))
        if np.isnan(x) or np.isnan(y):
            velocity_quivers[point].set_offsets(np.array([np.nan, np.nan]))
            velocity_labels[point].set_visible(False)
            point_labels[point].set_visible(False)
        else:
            velocity_quivers[point].set_offsets(np.array([[x, y]]))
            velocity_quivers[point].set_UVC(vx, vy)
            velocity_labels[point].set_position((x + vx * 0.5, y + vy * 0.5))
            velocity_labels[point].set_visible(True)
            point_labels[point].set_position((x + 2, y + 2))
            point_labels[point].set_visible(True)
    
    # Update angular acceleration arrows and labels
    for link in links:
        pt1, pt2 = link['points']
        x1, y1 = fixed_points[pt1] if pt1 in fixed_points else positions[pt1]
        x2, y2 = fixed_points[pt2] if pt2 in fixed_points else positions[pt2]
        if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
            alpha_quivers[link['name']].set_offsets(np.array([np.nan, np.nan]))
            alpha_labels[link['name']].set_visible(False)
            continue
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        length = sqrt(dx**2 + dy**2)
        if length == 0:
            tx, ty = 0, 0
        else:
            tx = -dy / length
            ty = dx / length
        alpha = compute_angular_acceleration(link, t)
        scale = 20
        u = tx * alpha * scale
        v = ty * alpha * scale
        alpha_quivers[link['name']].set_offsets(np.array([[mx, my]]))
        alpha_quivers[link['name']].set_UVC(u, v)
        alpha_labels[link['name']].set_position((mx + u * 0.2, my + v * 0.2))
        alpha_labels[link['name']].set_visible(True)
    
    # Collect data for plots
    plot_data['t'].append(t)
    for point in ['A', 'B', 'C', 'D', 'E', 'F']:
        x, y = positions[point]
        plot_data[f'{point}_x'].append(x)
        plot_data[f'{point}_y'].append(y)

    for point in ['B', 'C', 'E', 'F']:
        vx, vy = velocities.get(point, (0.0, 0.0))
        plot_data[f'{point}_vx'].append(vx)
        plot_data[f'{point}_vy'].append(vy)
    for link_name in ['CD', 'O2C', 'O3D']:
        link = next(link for link in links if link['name'] == link_name)
        omega = compute_angular_velocity(link, velocities, t)
        plot_data[f'{link_name}_omega'].append(omega)
    
    return list(link_lines.values()) + list(velocity_quivers.values()) + list(alpha_quivers.values()) + list(velocity_labels.values()) + list(alpha_labels.values()) + list(point_labels.values())

# Time array
t_max = 2 * np.pi / omega_O1A  # One full rotation period
times = np.linspace(0, t_max , 100)

ani = FuncAnimation(fig, update, frames=times, init_func=init, blit=True, interval=50, repeat=False)

plt.show()
time_array = np.array(plot_data['t'])
positions = {
    'A': np.column_stack((plot_data['A_x'], plot_data['A_y'])),
    'B': np.column_stack((plot_data['B_x'], plot_data['B_y'])),
    # Similarly for C, D, E, F
}
df_positions = pd.DataFrame({
    'Time (s)': plot_data['t'],
    'A_x': plot_data['A_x'], 'A_y': plot_data['A_y'],
    'B_x': plot_data['B_x'], 'B_y': plot_data['B_y'],
    'C_x': plot_data['C_x'], 'C_y': plot_data['C_y'],
    'D_x': plot_data['D_x'], 'D_y': plot_data['D_y'],
    'E_x': plot_data['E_x'], 'E_y': plot_data['E_y'],
    'F_x': plot_data['F_x'], 'F_y': plot_data['F_y'],
})

# Display the first few rows of the table
print(df_positions)
df_positions.to_csv('positions_table.csv', index=False)
# Post-animation plotting
t_array = np.array(plot_data['t'])

# Compute accelerations
for point in ['B', 'C', 'E', 'F']:
    vx = np.array(plot_data[f'{point}_vx'])
    vy = np.array(plot_data[f'{point}_vy'])
    ax = np.gradient(vx, t_array)
    ay = np.gradient(vy, t_array)
    plot_data[f'{point}_acc'] = np.sqrt(ax**2 + ay**2)

# Compute angular accelerations
for link_name in ['CD', 'O2C', 'O3D']:
    omega = np.array(plot_data[f'{link_name}_omega'])
    alpha = np.gradient(omega, t_array)
    plot_data[f'{link_name}_alpha'] = alpha

# Create plots
plt.figure(figsize=(15, 10))
ani.save('mechanism_animation.mp4', writer='ffmpeg', fps=30)
# Linear speeds
plt.subplot(2, 2, 1)
plt.plot(t_array, np.sqrt(np.array(plot_data['B_vx'])**2 + np.array(plot_data['B_vy'])**2), label='B')
plt.plot(t_array, np.sqrt(np.array(plot_data['C_vx'])**2 + np.array(plot_data['C_vy'])**2), label='C')
plt.plot(t_array, np.sqrt(np.array(plot_data['E_vx'])**2 + np.array(plot_data['E_vy'])**2), label='E')
plt.plot(t_array, np.sqrt(np.array(plot_data['F_vx'])**2 + np.array(plot_data['F_vy'])**2), label='F')
plt.xlabel('Time (s)')
plt.ylabel('Speed (units/s)')
plt.title('Linear Speeds')
plt.legend()

# Linear accelerations
plt.subplot(2, 2, 2)
plt.plot(t_array, plot_data['B_acc'], label='B')
plt.plot(t_array, plot_data['C_acc'], label='C')
plt.plot(t_array, plot_data['E_acc'], label='E')
plt.plot(t_array, plot_data['F_acc'], label='F')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (units/s²)')
plt.title('Linear Accelerations')
plt.legend()

# Angular speeds
plt.subplot(2, 2, 3)
plt.plot(t_array, plot_data['CD_omega'], label='CD')
plt.plot(t_array, plot_data['O2C_omega'], label='O2C')
plt.plot(t_array, plot_data['O3D_omega'], label='O3D')
plt.xlabel('Time (s)')
plt.ylabel('Angular Speed (rad/s)')
plt.title('Angular Speeds')
plt.legend()

# Angular accelerations
plt.subplot(2, 2, 4)
plt.plot(t_array, plot_data['CD_alpha'], label='CD')
plt.plot(t_array, plot_data['O2C_alpha'], label='O2C')
plt.plot(t_array, plot_data['O3D_alpha'], label='O3D')
plt.xlabel('Time (s)')
plt.ylabel('Angular Acceleration (rad/s²)')
plt.title('Angular Accelerations')
plt.legend()

plt.savefig("kinematics_plots.png", dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()