import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon

def simulate_lidar(sensor, angles, obstacles, max_range=10):
    distances = []
    ray_points = []  # end points of rays for visualization
    for angle in angles:
        # Calculate the end point of the ray at maximum range
        ray_end = Point(sensor.x + max_range * np.cos(angle),
                        sensor.y + max_range * np.sin(angle))
        ray = LineString([sensor, ray_end])
        min_distance = max_range
        for obs in obstacles:
            inter = ray.intersection(obs)
            if not inter.is_empty:
                # If the intersection is a point, compute the distance
                if inter.geom_type == 'Point':
                    d = sensor.distance(inter)
                    if d < min_distance:
                        min_distance = d
                # If the intersection is a line (or collection), choose the nearest point
                else:
                    # Use the first coordinate as a representative (could be improved)
                    point_candidate = Point(list(inter.coords)[0])
                    d = sensor.distance(point_candidate)
                    if d < min_distance:
                        min_distance = d
        distances.append(min_distance)
        # Calculate the actual end point of the ray based on the detected distance
        ray_points.append((sensor.x + min_distance * np.cos(angle),
                           sensor.y + min_distance * np.sin(angle)))
    return distances, ray_points

# Sensor position (as a Shapely Point)
sensor = Point(0, 0)

# Define obstacles (for example, a vertical line and a rectangular polygon)
obstacle1 = LineString([(2, -1), (2, 1)])  # vertical wall at x = 2
obstacle2 = Polygon([(4, -2), (5, -2), (5, -1), (4, -1)])  # a rectangle
obstacles = [obstacle1, obstacle2]

# Define LiDAR parameters: scanning angles (e.g., from -45° to 45°)
angles = np.linspace(-np.pi/4, np.pi/4, 50)

# Run the simulation
distances, ray_points = simulate_lidar(sensor, angles, obstacles, max_range=10)

# Visualization
plt.figure(figsize=(8, 8))
plt.plot(sensor.x, sensor.y, 'ro', label='Sensor')
# Plot obstacles
for obs in obstacles:
    if hasattr(obs, 'exterior'):
        plt.plot(*obs.exterior.xy, 'k-')
    else:
        xs, ys = zip(*list(obs.coords))
        plt.plot(xs, ys, 'k-')
# Plot rays
for pt in ray_points:
    plt.plot([sensor.x, pt[0]], [sensor.y, pt[1]], 'b-', alpha=0.3)
plt.xlim(-1, 11)
plt.ylim(-5, 5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("2D LiDAR Simulation")
plt.grid(True)
plt.show()
