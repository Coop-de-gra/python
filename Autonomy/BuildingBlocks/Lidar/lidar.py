import numpy as np
import matplotlib.pyplot as plt


def simulate_lidar_scan(num_rays=360, max_distance=5000, obstacles=None):
    """
    Simulate a LiDAR scan by casting rays from the origin.

    Parameters:
        num_rays (int): Number of rays to simulate (angles evenly spaced between 0 and 360 degrees).
        max_distance (float): Maximum LiDAR range.
        obstacles (list): List of obstacles; each obstacle is a dict with:
                          - "type": currently only 'circle'
                          - "center": (x, y) tuple for circle center
                          - "radius": radius of the circle

    Returns:
        list of tuples: Each tuple contains (angle in degrees, measured distance).
    """
    if obstacles is None:
        obstacles = []

    scan_data = []
    angles = np.linspace(0, 360, num_rays, endpoint=False)

    for angle in angles:
        angle_rad = np.deg2rad(angle)
        distance = max_distance  # default: no obstacle encountered

        # Check for intersection with each obstacle
        for obs in obstacles:
            if obs["type"] == "circle":
                cx, cy = obs["center"]
                r = obs["radius"]
                # The ray from the origin: x = t*cos(angle_rad), y = t*sin(angle_rad)
                # Solve: (t*cos(angle_rad) - cx)^2 + (t*sin(angle_rad) - cy)^2 = r^2
                # This expands to a quadratic in t: t^2 - 2*(cx*cos(angle_rad)+cy*sin(angle_rad))*t + (cx^2+cy^2 - r^2)=0
                a = 1
                b = -2 * (cx * np.cos(angle_rad) + cy * np.sin(angle_rad))
                c = cx ** 2 + cy ** 2 - r ** 2
                discriminant = b ** 2 - 4 * a * c

                if discriminant >= 0:
                    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
                    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
                    # Choose the smallest positive t (the first intersection along the ray)
                    t_candidates = [t for t in [t1, t2] if t > 0]
                    if t_candidates:
                        t_hit = min(t_candidates)
                        if t_hit < distance:
                            distance = t_hit

        scan_data.append((angle, distance))

    return scan_data


def plot_scan(scan_data, obstacles=None):
    """
    Plot the simulated LiDAR scan on a polar plot.
    also plot the obstacles in a separate Cartesian plot.
    """
    # Polar scatter plot for the LiDAR scan data
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    angles = [np.deg2rad(point[0]) for point in scan_data]
    distances = [point[1] for point in scan_data]
    ax.scatter(angles, distances, s=10)
    ax.set_title("Simulated LiDAR Scan")
    plt.show()

    # Cartesian plot to visualize obstacles
    if obstacles:
        fig2, ax2 = plt.subplots()
        # Plot obstacles (assumed to be circles)
        for obs in obstacles:
            if obs["type"] == "circle":
                circle = plt.Circle(obs["center"], obs["radius"], fill=False, edgecolor='r', linewidth=2)
                ax2.add_patch(circle)
        ax2.set_xlim(-max([obs["center"][0] for obs in obstacles] + [5000]) - 500,
                     max([obs["center"][0] for obs in obstacles] + [5000]) + 500)
        ax2.set_ylim(-max([obs["center"][1] for obs in obstacles] + [5000]) - 500,
                     max([obs["center"][1] for obs in obstacles] + [5000]) + 500)
        ax2.set_aspect('equal')
        ax2.set_title("Obstacle Map")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()


if __name__ == "__main__":
    # Define obstacles: each obstacle is a circle with a center (x, y) and a radius.
    obstacles = [
        {"type": "circle", "center": (2000, 1000), "radius": 500},
        {"type": "circle", "center": (-1500, -1500), "radius": 700},
        {"type": "circle", "center": (-2000, 2000), "radius": 300},
    ]

    # Run the simulation
    scan_data = simulate_lidar_scan(num_rays=360, max_distance=5000, obstacles=obstacles)
    plot_scan(scan_data, obstacles=obstacles)
