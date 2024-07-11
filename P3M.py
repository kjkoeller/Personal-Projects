# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:01:38 2024

@author: Kyle Koeller
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


G = 1.0 # gravitational constant
L = 100.0 # size of simulation box
N_particles = 500 # number of particles
N_grid = 64 # size of the grid for mesh calculations
circle_radius = 1
sigma = 2.0 # gaussian distribution standard deviation
angular_speed = 0.1 # 10% of desired angular momentum


def vector_pos(radius):
    while True:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        
        r = np.sqrt(x**2 + y**2)
        
        if r <= radius:
            break
    
    return x, y


def vector_vel(n, radius):
    t = [(np.random.uniform(-radius, radius)/10000, np.random.uniform(-radius, radius)/10000) for i in range(n)]
    
    x = [i[0] for i in t]
    y = [i[1] for i in t]
    
    return x, y


# particle properties
masses = np.full(N_particles, 1/N_particles)

positions = np.array([vector_pos(circle_radius) for _ in range(N_particles)])
velocities = np.zeros((N_particles, 2))  # zero initial velocities

# Center of mass correction
x_cm = np.mean(positions[:, 0])
y_cm = np.mean(positions[:, 1])

positions[:, 0] -= x_cm
positions[:, 1] -= y_cm

# Add small random velocities
vel_x, vel_y = vector_vel(N_particles, circle_radius)
velocities[:, 0] = vel_x
velocities[:, 1] = vel_y

# Add controlled angular momentum
for i in range(N_particles):
    r = np.sqrt(positions[i, 0]**2 + positions[i, 1]**2)
    if r > 0:
        vel_perpendicular = angular_speed * np.array([-positions[i, 1], positions[i, 0]]) / r
        velocities[i] += vel_perpendicular

# Center of mass velocity correction
vx_cmx = np.mean(velocities[:, 0])
vy_cmy = np.mean(velocities[:, 1])

velocities[:, 0] -= vx_cmx
velocities[:, 1] -= vy_cmy

r_cutoff = 0.1 * L # short-range force cutoff

dt = 0.01
T = 2


def gravitational_forces(positions, masses):
    N = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(i+1, N):
            r = positions[j] - positions[i]
            r_mag = np.linalg.norm(r)
            
            if r_mag > 0:
                f = G * masses[i] * masses[j] * r / r_mag**3
                forces[i] += f
                forces[j] -= f
    return forces


def compute_short_range_forces(positions, masses, r_cutoff):
    N = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(N):
        for j in range(i+1, N):
            r = positions[j] - positions[i]
            r_mag = np.linalg.norm(r)
            
            if r_mag < r_cutoff:
                f = G * masses[i] * masses[j] * r / r_mag**3
                forces[i] += f
                forces[j] += f
    return forces


def compute_long_range_forces(positions, masses, L, N_grid):
    # Create a grid for potential calculation
    rho = np.zeros((N_grid, N_grid))
    grid_spacing = L / N_grid
    
    # assign masses to the grid using CIC (Cloud-in-cell) method
    for pos, mass in zip(positions, masses):
        i = int((pos[0] + L/2) // grid_spacing)
        j = int((pos[1] + L/2) // grid_spacing)
        
        dx = ((pos[0] + L/2) % grid_spacing) / grid_spacing
        dy = ((pos[1] + L/2) % grid_spacing) / grid_spacing
        
        i1, i2 = i % N_grid, (i+1) % N_grid
        j1, j2 = j % N_grid, (j+1) % N_grid
        
        rho[i1, j1] += mass * (1-dx) * (1-dy)
        rho[i2, j1] += mass * dx * (1-dy)
        rho[i1, j2] += mass * (1-dx) * dy
        rho[i2, j2] += mass * dx * dy
    
    # compute gravitation potential with Poisson's equation
    phi = gaussian_filter(rho, sigma=1)
    
    # compute forces from the potential
    forces = np.zeros_like(positions)
    for k, pos in enumerate(positions):
        i = int((pos[0] + L/2) // grid_spacing)
        j = int((pos[1] + L/2) // grid_spacing)
        
        i1, i2 = i % N_grid, (i+1) % N_grid
        j1, j2 = j % N_grid, (j+1) % N_grid
        
        fx = -(phi[i2, j1] - phi[i1, j1]) / (2 * grid_spacing)
        fy = -(phi[i1, j2] - phi[i1, j1]) / (2 * grid_spacing)
        
        forces[k] = masses[k] * np.array([fx, fy])
    
    return forces


# Leap Frog method
def integrate(positions, velocities, masses, dt, L, N_grid, r_cutoff):
    forces = gravitational_forces(positions, masses) + \
             compute_short_range_forces(positions, masses, r_cutoff) + \
             compute_long_range_forces(positions, masses, L, N_grid)
    
    # update velocities by half step
    velocities_half_step = velocities + 0.5 * (forces/masses[:, np.newaxis]) * dt
    # update positions by full step
    positions += velocities_half_step * dt

    forces = gravitational_forces(positions, masses) + \
             compute_short_range_forces(positions, masses, r_cutoff) + \
             compute_long_range_forces(positions, masses, L, N_grid)
    
    # recompute velocities by another half step and keep positions within bounds
    velocities = velocities_half_step + 0.5 * (forces / masses[:, np.newaxis]) * dt
    
    return positions, velocities


plt.figure(figsize=(8,6))
plt.scatter(positions[:, 0], positions[:, 1], s=0.75)
plt.xlim(-circle_radius, circle_radius)
plt.ylim(-circle_radius, circle_radius)
plt.gca().set_aspect("equal")
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.show()

positions_history = []
num_steps = int(T/dt)

for t in tqdm(range(num_steps + 1), desc="Integrating"):
    positions_history.append(positions.copy())
    positions, velocities = integrate(positions, velocities, masses, dt, L, N_grid, r_cutoff)
    
    plt.figure(figsize=(8,6))
    plt.scatter(positions[:, 0], positions[:, 1], s=0.75)
    plt.xlim(-circle_radius, circle_radius)
    plt.ylim(-circle_radius, circle_radius)
    plt.gca().set_aspect("equal")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()