import numpy as np

class FieldCoveragePSO:
    def __init__(self, num_sensors, field_dims, num_particles=30, max_iter=100):
       
        self.num_sensors = num_sensors
        self.field_dims = field_dims
        self.num_particles = num_particles
        self.max_iter = max_iter
        
        self.sensor_radius = 15.0 
        self.dimensions = num_sensors * 2 
        self.w = 0.7   
        self.c1 = 1.5  
        self.c2 = 1.5  
        self.positions = np.random.rand(self.num_particles, self.dimensions)
        self.positions[:, 0::2] *= field_dims[0]  
        self.positions[:, 1::2] *= field_dims[1]
        
        self.velocities = np.zeros((self.num_particles, self.dimensions))
        
        self.p_best_pos = np.copy(self.positions)
        self.p_best_fit = np.array([self.calculate_fitness(p) for p in self.positions])
        
        best_idx = np.argmin(self.p_best_fit)
        self.g_best_pos = np.copy(self.p_best_pos[best_idx])
        self.g_best_fit = self.p_best_fit[best_idx]

    def calculate_fitness(self, position):
        penalty = 0.0
        sensors = position.reshape((self.num_sensors, 2))
        for i in range(self.num_sensors):
            for j in range(i + 1, self.num_sensors):
                dist = np.linalg.norm(sensors[i] - sensors[j])
                if dist < (2 * self.sensor_radius):
                    overlap = (2 * self.sensor_radius) - dist
                    penalty += overlap ** 2  
                    
        return penalty

    def optimize(self):
        """Runs the Particle Swarm Optimization loop."""
        for iteration in range(self.max_iter):
            for i in range(self.num_particles):
                # Update velocity
                r1, r2 = np.random.rand(self.dimensions), np.random.rand(self.dimensions)
                cognitive = self.c1 * r1 * (self.p_best_pos[i] - self.positions[i])
                social = self.c2 * r2 * (self.g_best_pos - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                self.positions[i] += self.velocities[i]
                self.positions[i, 0::2] = np.clip(self.positions[i, 0::2], 0, self.field_dims[0])
                self.positions[i, 1::2] = np.clip(self.positions[i, 1::2], 0, self.field_dims[1])
                
                current_fitness = self.calculate_fitness(self.positions[i])
            
                if current_fitness < self.p_best_fit[i]:
                    self.p_best_fit[i] = current_fitness
                    self.p_best_pos[i] = np.copy(self.positions[i])
                    
                    if current_fitness < self.g_best_fit:
                        self.g_best_fit = current_fitness
                        self.g_best_pos = np.copy(self.positions[i])
        
            if self.g_best_fit == 0:
                break
                
        return self.g_best_pos.reshape((self.num_sensors, 2)), self.g_best_fit

if __name__ == "__main__":
    field_size = (100, 100)
    total_sensors = 10
    
    pso_deployment = FieldCoveragePSO(num_sensors=total_sensors, field_dims=field_size)
    best_coordinates, final_penalty = pso_deployment.optimize()
    
    print("--- Optimized Sensor Deployments Coordinates ---")
    for idx, coordinate in enumerate(best_coordinates):
        print(f"Sensor {idx+1}: X = {coordinate[0]:.2f}m, Y = {coordinate[1]:.2f}m")
    print(f"\nFinal Layout Penalty Score (Lower is better): {final_penalty:.4f}")