import numpy as np
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, position, velocity, mass):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.force = np.zeros(3, dtype=float)

def initialize_particles(num_particles, cloud_radius, mass, initial_velocity, angular_momentum_percentage):
    particles = []
    
    for _ in range(num_particles):
        # Randomly distribute particles within a sphere
        r = cloud_radius * (random.random() ** (1/3))
        theta = random.uniform(0, 2 * np.pi)
        phi = random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        position = [x, y, z]
        
        # Small initial velocities with a percentage of angular momentum
        velocity_magnitude = initial_velocity
        velocity = np.random.normal(0, velocity_magnitude, 3)
        
        # Adding angular momentum component
        if angular_momentum_percentage > 0:
            angular_velocity = np.cross(position, velocity) * (angular_momentum_percentage / 100)
            velocity += angular_velocity
        
        particles.append(Particle(position, velocity, mass))
    
    return particles

class Octree:
    def __init__(self, center, size):
        self.center = np.array(center)
        self.size = size
        self.children = []
        self.particles = []
        self.mass = 0.0
        self.mass_center = np.zeros(3)
        
    def insert(self, particle):
        if not self.contains(particle.position):
            return False
        
        if len(self.children) == 0:
            self.particles.append(particle)
            if len(self.particles) > 1:
                self.subdivide()
            return True
        
        for child in self.children:
            if child.insert(particle):
                return True
        
        return False
    
    def subdivide(self):
        half_size = self.size / 2.0
        quarter_size = self.size / 4.0
        
        offsets = [
            np.array([ quarter_size,  quarter_size,  quarter_size]),
            np.array([ quarter_size,  quarter_size, -quarter_size]),
            np.array([ quarter_size, -quarter_size,  quarter_size]),
            np.array([ quarter_size, -quarter_size, -quarter_size]),
            np.array([-quarter_size,  quarter_size,  quarter_size]),
            np.array([-quarter_size,  quarter_size, -quarter_size]),
            np.array([-quarter_size, -quarter_size,  quarter_size]),
            np.array([-quarter_size, -quarter_size, -quarter_size]),
        ]
        
        self.children = [Octree(self.center + offset, half_size) for offset in offsets]
        
        for particle in self.particles:
            for child in self.children:
                if child.insert(particle):
                    break
        
        self.particles = []
    
    def contains(self, point):
        return all(abs(point - self.center) <= self.size / 2)
    
    def compute_mass_distribution(self):
        if len(self.children) == 0:
            if len(self.particles) == 1:
                self.mass = self.particles[0].mass
                self.mass_center = self.particles[0].position
        else:
            self.mass = 0.0
            self.mass_center = np.zeros(3)
            for child in self.children:
                child.compute_mass_distribution()
                self.mass += child.mass
                self.mass_center += child.mass * child.mass_center
            if self.mass > 0:
                self.mass_center /= self.mass

def compute_force(tree, particle, theta=0.5, G=1.0):
    if len(tree.children) == 0:
        if len(tree.particles) == 1 and tree.particles[0] is particle:
            return np.zeros(3)
        else:
            direction = tree.mass_center - particle.position
            distance = np.linalg.norm(direction)
            if distance == 0:
                return np.zeros(3)
            force_magnitude = G * particle.mass * tree.mass / (distance ** 2)
            return force_magnitude * direction / distance
    else:
        direction = tree.mass_center - particle.position
        distance = np.linalg.norm(direction)
        if distance == 0:
            return np.zeros(3)
        
        if tree.size / distance < theta:
            force_magnitude = G * particle.mass * tree.mass / (distance ** 2)
            return force_magnitude * direction / distance
        else:
            force = np.zeros(3)
            for child in tree.children:
                force += compute_force(child, particle, theta, G)
            return force

def update_forces(particles, tree, theta=0.5, G=1.0):
    for particle in particles:
        particle.force = compute_force(tree, particle, theta, G)

def leapfrog_integration(particles, dt, theta=0.5, G=1.0):
    for particle in particles:
        particle.velocity += 0.5 * particle.force * dt / particle.mass
        particle.position += particle.velocity * dt

    tree = Octree(center=[0, 0, 0], size=2 * max(np.linalg.norm(p.position) for p in particles))
    for particle in particles:
        tree.insert(particle)
    
    tree.compute_mass_distribution()
    update_forces(particles, tree, theta, G)
    
    for particle in particles:
        particle.velocity += 0.5 * particle.force * dt / particle.mass

def visualize_particles(particles, step):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    xs = [p.position[0] for p in particles]
    ys = [p.position[1] for p in particles]
    zs = [p.position[2] for p in particles]
    
    # XY plane
    axs[0].scatter(xs, ys, s=1)
    axs[0].set_xlim([-20, 20])
    axs[0].set_ylim([-20, 20])
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('XY Plane')

    # XZ plane
    axs[1].scatter(xs, zs, s=1)
    axs[1].set_xlim([-20, 20])
    axs[1].set_ylim([-20, 20])
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Z')
    axs[1].set_title('XZ Plane')

    # YZ plane
    axs[2].scatter(ys, zs, s=1)
    axs[2].set_xlim([-20, 20])
    axs[2].set_ylim([-20, 20])
    axs[2].set_xlabel('Y')
    axs[2].set_ylabel('Z')
    axs[2].set_title('YZ Plane')

    plt.suptitle(f"Step {step}")
    plt.show()

def simulate(num_particles, cloud_radius, mass, initial_velocity, angular_momentum_percentage, num_steps, dt, theta=0.5, G=1.0):
    particles = initialize_particles(num_particles, cloud_radius, mass, initial_velocity, angular_momentum_percentage)
    
    tree = Octree(center=[0, 0, 0], size=2 * cloud_radius)
    for particle in particles:
        tree.insert(particle)
    
    tree.compute_mass_distribution()
    update_forces(particles, tree, theta, G)
    
    for step in range(num_steps):
        leapfrog_integration(particles, dt, theta, G)
        
        # Visualize particles
        if step % 10 == 0:  # Adjust this value to visualize at desired intervals
            visualize_particles(particles, step)
    
    return particles

# Example usage
num_particles = 1000
cloud_radius = 10.0
particle_mass = 1.0
initial_velocity = 0.1
angular_momentum_percentage = 5.0
num_steps = 100
dt = 0.01

particles = simulate(num_particles, cloud_radius, particle_mass, initial_velocity, angular_momentum_percentage, num_steps, dt)