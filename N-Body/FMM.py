import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Particle:
    def __init__(self, position, mass):
        self.position = np.array(position)
        self.mass = mass
        self.potential = 0
        self.force = np.zeros(3)


class OctreeNode:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.children = []
        self.particles = []
        self.multipole = np.zeros(10)  # Example: monopole, dipole, quadrupole terms
        self.local_expansion = np.zeros(10)  # Similar structure as multipole for local expansion


def build_octree(particles, center, size, max_particles_per_node):
    node = OctreeNode(center, size)
    if len(particles) <= max_particles_per_node:
        node.particles = particles
    else:
        half_size = size / 2
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                for dz in [-1, 1]:
                    new_center = center + np.array([dx, dy, dz]) * half_size / 2
                    child_particles = [p for p in particles if np.all(np.abs(p.position - new_center) <= half_size / 2)]
                    if child_particles:
                        child_node = build_octree(child_particles, new_center, half_size, max_particles_per_node)
                        node.children.append(child_node)
    return node


def compute_multipole_expansion(node):
    if node.children:
        for child in node.children:
            compute_multipole_expansion(child)
            node.multipole += child.multipole  # Combine child multipole expansions
    else:
        for p in node.particles:
            r_vec = p.position - node.center
            r = np.linalg.norm(r_vec)
            node.multipole[0] += p.mass  # Monopole moment (total mass)
            node.multipole[1:4] += p.mass * r_vec  # Dipole moment
            node.multipole[4:10] += p.mass * np.outer(r_vec, r_vec).flatten()[:6]  # Quadrupole moment


def compute_local_expansion(node, source_node):
    if is_far_enough(node, source_node):
        r_vec = node.center - source_node.center
        r = np.linalg.norm(r_vec)
        node.local_expansion += translate_multipole_to_local(source_node.multipole, r_vec, r)
    else:
        for child in source_node.children:
            compute_local_expansion(node, child)

def translate_multipole_to_local(multipole, r_vec, r):
    # Translate multipole terms to local expansion
    # Simplified example; should include proper translation logic
    return multipole / r**3


def is_far_enough(node, source_node):
    distance = np.linalg.norm(node.center - source_node.center)
    return distance > 2 * max(node.size, source_node.size)


def calculate_potential_and_force(node):
    if node.children:
        for child in node.children:
            calculate_potential_and_force(child)
    else:
        for p in node.particles:
            for other in node.particles:
                if p != other:
                    r_vec = p.position - other.position
                    r = np.linalg.norm(r_vec)
                    p.potential += other.mass / r
                    p.force -= other.mass * r_vec / r**3
            # Include contributions from multipole and local expansions as necessary
            for i in range(10):
                p.potential += node.local_expansion[i]  # Simplified for demonstration
                p.force -= node.local_expansion[i]  # Simplified for demonstration


def visualize_particles(particles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for p in particles:
        ax.scatter(p.position[0], p.position[1], p.position[2], marker='o')
    plt.show()


# Example particle setup
particles = [
    Particle([0.1, 0.2, 0.3], 1),
    Particle([0.4, 0.5, 0.6], 1),
    # Add more particles as needed
]

# Build the octree
root = build_octree(particles, center=[0, 0, 0], size=1, max_particles_per_node=4)

# Compute multipole expansions
compute_multipole_expansion(root)

# Compute local expansions
for node in root.children:
    compute_local_expansion(node, root)

# Calculate potential and forces
calculate_potential_and_force(root)

# Print results
for p in particles:
    print(f"Particle at {p.position} has potential {p.potential} and force {p.force}")

# Visualize particles
visualize_particles(particles)