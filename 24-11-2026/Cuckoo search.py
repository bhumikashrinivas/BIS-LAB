import numpy as np
from scipy.special import gamma

a = np.array([0.008, 0.0096, 0.007])
b = np.array([7.0, 6.3, 6.8])
c = np.array([200.0, 180.0, 140.0])

P_min = np.array([10.0, 10.0, 10.0])
P_max = np.array([85.0, 80.0, 70.0])

P_demand = 150.0

B = np.array([
    [0.000218, 0.000093, 0.000028],
    [0.000093, 0.000228, 0.000017],
    [0.000028, 0.000017, 0.000179]
])

def evaluate_fitness(P):
    
    fuel_cost = np.sum(a * (P ** 2) + b * P + c)
    
    P_loss = np.dot(P, np.dot(B, P))
    
    mismatch = abs(np.sum(P) - (P_demand + P_loss))
    
    penalty_factor = 1000.0
    total_fitness = fuel_cost + penalty_factor * (mismatch ** 2)
    
    return total_fitness, fuel_cost, P_loss

def get_levy_step(beta, dim):
    """Generates step size using Mantegna's algorithm for Levy flights."""
    num = gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / (abs(v) ** (1 / beta))
    return step

def apply_bounds(P, P_min, P_max):
    """Clamps generator outputs within their min/max limits."""
    return np.clip(P, P_min, P_max)

def cuckoo_search_eld(n_nests=25, max_iter=500, pa=0.25, beta=1.5):
    dim = len(P_min)
    
    nests = np.random.uniform(P_min, P_max, size=(n_nests, dim))
    fitness = np.zeros(n_nests)
    
    for i in range(n_nests):
        fitness[i], _, _ = evaluate_fitness(nests[i])
        
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    for iteration in range(max_iter):
        for i in range(n_nests):
            step_size = get_levy_step(beta, dim)
            new_nest = nests[i] + 0.01 * step_size * (nests[i] - best_nest)
            new_nest = apply_bounds(new_nest, P_min, P_max)
            
            new_fit, _, _ = evaluate_fitness(new_nest)
            
            j = np.random.randint(0, n_nests)
            if new_fit < fitness[j]:
                nests[j] = new_nest
                fitness[j] = new_fit
                
        for i in range(n_nests):
            if np.random.rand() < pa:
                r1, r2 = np.random.randint(0, n_nests, size=2)
                step_rand = np.random.rand() * (nests[r1] - nests[r2])
                new_nest = apply_bounds(nests[i] + step_rand, P_min, P_max)
                
                new_fit, _, _ = evaluate_fitness(new_nest)
                if new_fit < fitness[i]:
                    nests[i] = new_nest
                    fitness[i] = new_fit
                    
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_nest = nests[current_best_idx].copy()
            
    return best_nest

if __name__ == "__main__":
    np.random.seed(42)  
    
    optimal_P = cuckoo_search_eld(n_nests=30, max_iter=1000, pa=0.25)
    _, total_cost, total_loss = evaluate_fitness(optimal_P)
    
    print("=== Economic Load Dispatch Results (Cuckoo Search) ===")
    for idx, p_val in enumerate(optimal_P, 1):
        print(f"Generator {idx} Output : {p_val:.4f} MW")
    print("-" * 52)
    print(f"Total Generation   : {np.sum(optimal_P):.4f} MW")
    print(f"Power Demand       : {P_demand:.4f} MW")
    print(f"Transmission Loss  : {total_loss:.4f} MW")
    print(f"Power Mismatch     : {abs(np.sum(optimal_P) - (P_demand + total_loss)):.6f} MW")
    print(f"Total Fuel Cost    : ${total_cost:.4f} / hr")
