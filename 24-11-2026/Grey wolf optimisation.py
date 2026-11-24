import random

approaches = ["North", "South", "East", "West"]

arrival_rate = [0.35, 0.30, 0.25, 0.20]
saturation_rate = [0.55, 0.55, 0.50, 0.50]

CYCLE_TIME = 120.0
YELLOW_TIME = 3.0
TOTAL_LOST_TIME = len(approaches) * YELLOW_TIME
EFFECTIVE_CYCLE = CYCLE_TIME - TOTAL_LOST_TIME

MIN_GREEN = 15.0
MAX_GREEN = 60.0

def repair_green_times(green_times):
    g = [max(MIN_GREEN, min(val, MAX_GREEN)) for val in green_times]
    
    for _ in range(20):
        diff = EFFECTIVE_CYCLE - sum(g)
        if abs(diff) < 1e-6:
            break
        if diff > 0:
            adjustable = [i for i in range(len(g)) if g[i] < MAX_GREEN]
            if not adjustable:
                break
            add_each = diff / len(adjustable)
            for i in adjustable:
                g[i] = min(MAX_GREEN, g[i] + add_each)
        else:
            adjustable = [i for i in range(len(g)) if g[i] > MIN_GREEN]
            if not adjustable:
                break
            sub_each = (-diff) / len(adjustable)
            for i in adjustable:
                g[i] = max(MIN_GREEN, g[i] - sub_each)
    return g

def evaluate_delay(green_times):
    g = repair_green_times(green_times)
    total_weighted_delay = 0.0
    
    for i in range(len(approaches)):
        q = arrival_rate[i]
        s = saturation_rate[i]
        lam = g[i] / CYCLE_TIME
        capacity = s * lam
        
        x = q / capacity
        
        if x >= 0.99:
            total_weighted_delay += 10000.0 * (x - 0.95) ** 2 + 500.0
        else:
            term1 = (CYCLE_TIME * ((1.0 - lam) ** 2)) / (2.0 * (1.0 - (q / s)))
            term2 = (x ** 2) / (2.0 * q * (1.0 - x))
            avg_delay = 0.9 * (term1 + term2)
            total_weighted_delay += q * avg_delay
            
    return total_weighted_delay

def grey_wolf_optimizer(n_wolves=30, max_iter=200):
    dim = len(approaches)
    
    positions = [
        repair_green_times([random.uniform(MIN_GREEN, MAX_GREEN) for _ in range(dim)])
        for _ in range(n_wolves)
    ]
    
    alpha_pos, alpha_score = [0.0] * dim, float("inf")
    beta_pos, beta_score = [0.0] * dim, float("inf")
    delta_pos, delta_score = [0.0] * dim, float("inf")
    
    for iteration in range(max_iter):
        for i in range(n_wolves):
            positions[i] = repair_green_times(positions[i])
            fitness = evaluate_delay(positions[i])
            
            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos[:]
                beta_score, beta_pos = alpha_score, alpha_pos[:]
                alpha_score, alpha_pos = fitness, positions[i][:]
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos[:]
                beta_score, beta_pos = fitness, positions[i][:]
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, positions[i][:]
                
        a = 2.0 - iteration * (2.0 / max_iter)
        
        for i in range(n_wolves):
            new_pos = [0.0] * dim
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = abs(C1 * alpha_pos[d] - positions[i][d])
                X1 = alpha_pos[d] - A1 * D_alpha
                
                r1, r2 = random.random(), random.random()
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = abs(C2 * beta_pos[d] - positions[i][d])
                X2 = beta_pos[d] - A2 * D_beta
                
                r1, r2 = random.random(), random.random()
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = abs(C3 * delta_pos[d] - positions[i][d])
                X3 = delta_pos[d] - A3 * D_delta
                
                new_pos[d] = (X1 + X2 + X3) / 3.0
                
            positions[i] = repair_green_times(new_pos)
            
    return alpha_pos, alpha_score

if __name__ == "__main__":
    random.seed(42)
    
    best_greens, min_delay = grey_wolf_optimizer(n_wolves=30, max_iter=200)
    total_arrivals = sum(arrival_rate)
    avg_delay_per_vehicle = min_delay / total_arrivals
    
    print("=== Traffic Signal Timing Optimization (Grey Wolf Optimization) ===")
    print(f"Total Cycle Time : {CYCLE_TIME:.1f} s (Yellow: {YELLOW_TIME:.1f} s/phase)")
    print("-" * 65)
    print(f"{'Approach':<10} | {'Arrival (veh/hr)':<18} | {'Optimal Green (s)':<18} | {'Saturation (x)'}")
    print("-" * 65)
    
    for i in range(len(approaches)):
        veh_per_hr = arrival_rate[i] * 3600
        deg_sat = arrival_rate[i] / (saturation_rate[i] * (best_greens[i] / CYCLE_TIME))
        print(f"{approaches[i]:<10} | {veh_per_hr:<18.0f} | {best_greens[i]:<18.2f} | {deg_sat:.3f}")
        
    print("-" * 65)
    print(f"Sum of Green Times       : {sum(best_greens):.2f} s (+ {TOTAL_LOST_TIME:.1f}s yellow = {sum(best_greens)+TOTAL_LOST_TIME:.1f}s)")
    print(f"Average Delay / Vehicle  : {avg_delay_per_vehicle:.2f} seconds")
