import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import os

matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting

# Gravitational constant
G = 1.0

# Particle mass
particle_mass = 1.0 / 2000.0

# Particle class definition
class Particle:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = particle_mass

# OctreeNode class definition
class OctreeNode:
    def __init__(self, center, half_width, depth=0, max_depth=10):
        self.center = np.array(center)
        self.half_width = half_width
        self.particle = None
        self.children = [None] * 8
        self.mass = 0.0
        self.center_of_mass = np.zeros(3)
        self.depth = depth
        self.max_depth = max_depth

    def is_leaf(self):
        return all(child is None for child in self.children)

    def insert(self, particle):
        if self.is_leaf():
            if self.particle is None:
                self.particle = particle
                self.mass = particle.mass
                self.center_of_mass = particle.position
            else:
                if self.depth >= self.max_depth:
                    self.combine_particle(particle)
                else:
                    old_particle = self.particle
                    self.particle = None
                    self.subdivide()
                    self.insert_particle(old_particle)
                    self.insert_particle(particle)
        else:
            self.insert_particle(particle)
        self.update_mass_and_com(particle)

    def subdivide(self):
        half = self.half_width / 2
        for i in range(8):
            offset = np.array([
                (i & 1) * half,
                (i >> 1 & 1) * half,
                (i >> 2 & 1) * half
            ])
            child_center = self.center + offset - half / 2
            self.children[i] = OctreeNode(child_center, half, self.depth + 1, self.max_depth)

    def insert_particle(self, particle):
        index = 0
        if particle.position[0] > self.center[0]: index |= 1
        if particle.position[1] > self.center[1]: index |= 2
        if particle.position[2] > self.center[2]: index |= 4
        self.children[index].insert(particle)

    def update_mass_and_com(self, particle):
        total_mass = self.mass + particle.mass
        self.center_of_mass = (self.center_of_mass * self.mass + particle.position * particle.mass) / total_mass
        self.mass = total_mass

    def combine_particle(self, particle):
        self.center_of_mass = (self.center_of_mass * self.mass + particle.position * particle.mass) / (self.mass + particle.mass)
        self.mass += particle.mass

# Function to calculate force from a node
def calculate_force_from_node(node, particle, theta, G):
    force = np.zeros(3)
    if node.is_leaf():
        if node.particle is not None and node.particle is not particle:
            r_vec = node.particle.position - particle.position
            distance = np.linalg.norm(r_vec)
            if distance > 0:
                force_magnitude = G * particle.mass * node.particle.mass / distance ** 2
                force = force_magnitude * r_vec / distance
    else:
        r_vec = node.center_of_mass - particle.position
        distance = np.linalg.norm(r_vec)
        if distance > 0:
            if node.half_width / distance < theta:
                force_magnitude = G * particle.mass * node.mass / distance ** 2
                force = force_magnitude * r_vec / distance
            else:
                for child in node.children:
                    if child is not None:
                        force += calculate_force_from_node(child, particle, theta, G)
    return force


def calculate_potential_from_node(node, position, theta, G):
    potential = 0.0
    if node.is_leaf():
        if node.particle is not None:
            r_vec = node.particle.position - position
            distance = np.linalg.norm(r_vec)
            if distance > 0:
                potential = -G * node.particle.mass / distance
    else:
        r_vec = node.center_of_mass - position
        distance = np.linalg.norm(r_vec)
        if distance > 0:
            if node.half_width / distance < theta:
                potential = -G * node.mass / distance
            else:
                for child in node.children:
                    if child is not None:
                        potential += calculate_potential_from_node(child, position, theta, G)
    return potential


# Function to calculate forces for all particles using octree
def calculate_forces_octree(root, particles, theta=0.5, G=1):
    forces = [np.zeros(3) for _ in particles]
    for i, particle in enumerate(particles):
        forces[i] = calculate_force_from_node(root, particle, theta, G)
    return forces

# Function to build the octree
def build_octree(particles, center, half_width, max_depth=10):
    root = OctreeNode(center, half_width, max_depth=max_depth)
    for particle in particles:
        root.insert(particle)
    return root

# Function to update particles
def update_particles(particles, forces, dt):
    for i, particle in enumerate(particles):
        particle.velocity += forces[i] * dt / particle.mass
        particle.position += particle.velocity * dt

# Function to calculate potential energy of the system
def calculate_potential_energy(particles, root, theta):
    potential_energy = 0.0
    for particle in particles:
        potential = calculate_potential_from_node(root, particle.position, theta)
        potential_energy += 0.5 * particle.mass * potential  # Factor of 0.5 to avoid double-counting
    return potential_energy

# Function to calculate kinetic energy of the system
def calculate_kinetic_energy(particles):
    kinetic_energy = 0.0
    for particle in particles:
        speed_squared = np.dot(particle.velocity, particle.velocity)
        kinetic_energy += 0.5 * particle.mass * speed_squared
    return kinetic_energy

# Function to read positions and velocities from a file
def read_particles_from_file(filename):
    particles = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            x, y, z, vx, vy, vz = map(float, line.split())
            position = [x, y, z]
            velocity = [vx, vy, vz]
            particles.append(Particle(position=position, velocity=velocity))
    return particles

# Function to run the simulation and save figures and energy data at each timestep
def simulate(particles, num_steps, dt, half_width, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open CSV files to write particle positions/velocities and energies
    particle_csv_filename = os.path.join(output_dir, 'particles_output.csv')
    energy_csv_filename = os.path.join(output_dir, 'energy_output.csv')
    
    with open(particle_csv_filename, mode='w', newline='') as particle_file, open(energy_csv_filename, mode='w', newline='') as energy_file:
        particle_writer = csv.writer(particle_file)
        energy_writer = csv.writer(energy_file)
        
        # Write headers to the CSV files
        particle_writer.writerow(["Step", "Particle", "X", "Y", "Z", "VX", "VY", "VZ"])
        energy_writer.writerow(["Step", "Potential Energy", "Kinetic Energy", "Total Energy"])
        
        for step in range(num_steps):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            center = np.mean([p.position for p in particles], axis=0)
            root = build_octree(particles, center, half_width, max_depth=10)
            forces = calculate_forces_octree(root, particles)
            update_particles(particles, forces, dt)
            
            positions = np.array([p.position for p in particles])

            # XY plane
            axs[0].scatter(positions[:, 0], positions[:, 1], s=0.5, c='black')
            axs[0].set_xlim(-2, 2)
            axs[0].set_ylim(-2, 2)
            axs[0].set_xlabel('X')
            axs[0].set_ylabel('Y')
            axs[0].set_title('XY Plane')

            # XZ plane
            axs[1].scatter(positions[:, 0], positions[:, 2], s=0.5, c='black')
            axs[1].set_xlim(-2, 2)
            axs[1].set_ylim(-2, 2)
            axs[1].set_xlabel('X')
            axs[1].set_ylabel('Z')
            axs[1].set_title('XZ Plane')

            # YZ plane
            axs[2].scatter(positions[:, 1], positions[:, 2], s=0.5, c='black')
            axs[2].set_xlim(-2, 2)
            axs[2].set_ylim(-2, 2)
            axs[2].set_xlabel('Y')
            axs[2].set_ylabel('Z')
            axs[2].set_title('YZ Plane')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'frame_{step:04d}.png'))
        plt.close(fig)

        # Write particle positions and velocities to the CSV file
        for i, particle in enumerate(particles):
            particle_writer.writerow([step, i, particle.position[0], particle.position[1], particle.position[2],
                                      particle.velocity[0], particle.velocity[1], particle.velocity[2]])
        
        # Calculate and write energies to the CSV file
        potential_energy = calculate_potential_energy(particles, root, theta=0.5)
        kinetic_energy = calculate_kinetic_energy(particles)
        total_energy = potential_energy + kinetic_energy
        energy_writer.writerow([step, potential_energy, kinetic_energy, total_energy])


input_filename = 'tbini.txt'  # Replace with your input file path
output_dir = 'output'  # Replace with your desired output directory
particles = read_particles_from_file(input_filename)

simulate(particles, num_steps=1000, dt=0.005, half_width=50.0, output_dir=output_dir)

