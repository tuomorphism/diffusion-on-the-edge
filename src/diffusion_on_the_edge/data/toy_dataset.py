import numpy as np

def generate_two_moons(n=5000, noise=0.01, circle1_rad = 1.4, circle2_rad = 0.5):
    points = []
    for theta in np.linspace(0, np.pi, int(n // 2)):
        points.append([circle1_rad * np.cos(theta) + 0.6, 0.2 + circle1_rad * np.sin(theta)])
        points.append([circle2_rad * np.cos(theta + np.pi) - 0.5, circle2_rad * np.sin(theta + np.pi) - 1.0])

    points = np.array(points)
    points += np.random.normal(scale=noise, size=points.shape)

    return points