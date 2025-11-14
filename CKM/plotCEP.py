import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

qs = np.load('qs.npy')
centers = qs[0:15,0:3]
print(centers)
# Example 15 3D centers


cep_radius = 5
num_points_per_center = 1000  # Number of points for each center

# Prepare figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for center in centers:
    # Generate random angles and radii for a 3D spherical distribution
    phi = np.random.uniform(0, 2 * np.pi, num_points_per_center)
    theta = np.random.uniform(0, np.pi, num_points_per_center)
    r = np.random.normal(0, cep_radius / 1.177, num_points_per_center)  # Adjusted for 50% within 5 meters

    # Convert spherical coordinates to Cartesian coordinates and offset by center
    x = center[0] + r * np.sin(theta) * np.cos(phi)
    y = center[1] + r * np.sin(theta) * np.sin(phi)
    z = center[2] + r * np.cos(theta)

    # Plot the points for each center with different colors based on distance from center
    error_magnitude = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    ax.scatter(x, y, z, c=error_magnitude, cmap='viridis', marker='o', alpha=0.5,s=50)

# Hide axes for a clean plot
ax.set_axis_off()

# Show plot
plt.savefig('CEP',dpi=400)
plt.show()
