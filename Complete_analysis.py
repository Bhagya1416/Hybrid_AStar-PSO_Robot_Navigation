import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import distance_transform_edt

from AStar_MultiObj import baseline_astar, multi_objective_astar
from PSO_MultiObj import mo_pso_path_planner
from Hybrid_AStar_PSO import hybrid_astar_pso
from Map import map_a, map_b, map_c, map_d

# =================================================
# FIXED SETTINGS
# =================================================
start = (0, 0)
goal = (19, 19)

maps = [
    ("Map A: Sparse Complex", map_a),
    ("Map B: Dense Obstacle", map_b),
    ("Map C: Narrow Passage", map_c),
    ("Map D: Dead-End / Trap", map_d)
]

NUM_RUNS = 30

# =================================================
# METRIC FUNCTIONS
# =================================================
def compute_path_length(path):
    path_arr = np.array(path)
    return np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))

def compute_turns(path):
    turns = 0
    for i in range(2, len(path)):
        v1 = np.array(path[i-1]) - np.array(path[i-2])
        v2 = np.array(path[i]) - np.array(path[i-1])
        if not np.array_equal(v1, v2):
            turns += 1
    return turns

def compute_min_clearance(path, grid):
    clearance_map = distance_transform_edt(1 - grid)
    return min(clearance_map[int(p[0]), int(p[1])] for p in path)


# =================================================
# TABLE–1 : Environment / Map Description
# =================================================
print("\nTABLE–1: Environment / Map Description")
print("Map ID | Map Type | Grid Size | Obstacle Density | Description")
print("Map A: Sparse Complex | Sparse | 20x20 | Low | Sparse obstacle distribution")
print("Map B: Dense Obstacle | Dense | 20x20 | High | Dense, high obstacles")
print("Map C: Narrow Passage | Corridor-like | 20x20 | Medium | Corridor-like, medium obstacles")
print("Map D: Dead-End / Trap | Dead-end structures | 20x20 | Medium | Trap-like dead-end structures")


# TABLE–2 : Algorithm Parameters
print("\nTABLE–2: Algorithm Parameters")
print("Algorithm | Parameter | Value")
print("MO-A* | Heuristic | Euclidean")
print("MO-PSO | Particles | 30")
print("MO-PSO | Iterations | 100")
print("MO-PSO | Waypoints | 7")
print("MO-PSO | Weights w1,w2,w3 | 1.0,2.0,2.0")
print("Hybrid A*+PSO | Particles | 30")
print("Hybrid A*+PSO | Iterations | 100")
print("Hybrid A*+PSO | Waypoints | 9")
print("Hybrid A*+PSO | Weights w1,w2,w3 | 1.0,2.5,8.0")


# =================================================
# TABLE–3 : Single-Run Comparison
# =================================================
print("\nTABLE–3: Representative Single-Run Comparison (Fixed Start–Goal)")
print("Map | Method | Length | Turns | Min Clearance | Time (s)")

for map_name, grid in maps:
    # MO-A*
    t0 = time.time()
    path_a = multi_objective_astar(grid, start, goal)
    t_a = time.time() - t0
    if path_a is not None:
        length_a = compute_path_length(path_a)
        turns_a = compute_turns(path_a)
        clearance_a = compute_min_clearance(path_a, grid)
    else:
        length_a = turns_a = clearance_a = t_a = np.nan

    # MO-PSO
    t0 = time.time()
    path_pso = mo_pso_path_planner(grid, start, goal, num_particles=30, num_waypoints=7, max_iter=100)
    t_pso = time.time() - t0
    if path_pso is not None:
        length_pso = compute_path_length(path_pso)
        turns_pso = compute_turns(path_pso)
        clearance_pso = compute_min_clearance(path_pso, grid)
    else:
        length_pso = turns_pso = clearance_pso = t_pso = np.nan

    # Hybrid A* + PSO
    t0 = time.time()
    path_hybrid = hybrid_astar_pso(grid, start, goal, num_particles=30, num_waypoints=9, max_iter=100)
    t_hybrid = time.time() - t0
    if path_hybrid is not None:
        length_hybrid = compute_path_length(path_hybrid)
        turns_hybrid = compute_turns(path_hybrid)
        clearance_hybrid = compute_min_clearance(path_hybrid, grid)
    else:
        length_hybrid = turns_hybrid = clearance_hybrid = t_hybrid = np.nan

    print(f"{map_name} | MO-A* | {length_a:.2f} | {turns_a} | {clearance_a:.2f} | {t_a:.3f}")
    print(f"{map_name} | MO-PSO | {length_pso:.2f} | {turns_pso} | {clearance_pso:.2f} | {t_pso:.3f}")
    print(f"{map_name} | Hybrid | {length_hybrid:.2f} | {turns_hybrid} | {clearance_hybrid:.2f} | {t_hybrid:.3f}")

# =================================================
# TABLE–4 : Statistical Performance
# =================================================
results_hybrid = {}
results_pso = {}

for map_name, grid in maps:
    L_h, T_h, C_h, Time_h = [], [], [], []
    L_p, T_p, C_p, Time_p = [], [], [], []
    success_h, success_p = 0, 0

    for _ in range(NUM_RUNS):
        # MO-PSO
        t0 = time.time()
        path_pso = mo_pso_path_planner(grid, start, goal, num_particles=30, num_waypoints=7, max_iter=100)
        t = time.time() - t0
        if path_pso is not None:
            L_p.append(compute_path_length(path_pso))
            T_p.append(compute_turns(path_pso))
            C_p.append(compute_min_clearance(path_pso, grid))
            Time_p.append(t)
            success_p += 1

        # Hybrid
        t0 = time.time()
        path_hybrid = hybrid_astar_pso(grid, start, goal, num_particles=30, num_waypoints=9, max_iter=100)
        t = time.time() - t0
        if path_hybrid is not None:
            L_h.append(compute_path_length(path_hybrid))
            T_h.append(compute_turns(path_hybrid))
            C_h.append(compute_min_clearance(path_hybrid, grid))
            Time_h.append(t)
            success_h += 1

    results_hybrid[map_name] = {
        "Success": success_h/NUM_RUNS*100,
        "Length": (np.mean(L_h) if L_h else np.nan, np.std(L_h) if L_h else np.nan),
        "Turns": (np.mean(T_h) if T_h else np.nan, np.std(T_h) if T_h else np.nan),
        "Clearance": (np.mean(C_h) if C_h else np.nan, np.std(C_h) if C_h else np.nan),
        "Time": (np.mean(Time_h) if Time_h else np.nan, np.std(Time_h) if Time_h else np.nan)
    }
    results_pso[map_name] = {
        "Success": success_p/NUM_RUNS*100,
        "Length": (np.mean(L_p) if L_p else np.nan, np.std(L_p) if L_p else np.nan),
        "Turns": (np.mean(T_p) if T_p else np.nan, np.std(T_p) if T_p else np.nan),
        "Clearance": (np.mean(C_p) if C_p else np.nan, np.std(C_p) if C_p else np.nan),
        "Time": (np.mean(Time_p) if Time_p else np.nan, np.std(Time_p) if Time_p else np.nan)
    }

print("\nTABLE–4: Statistical Performance")
print("Method | Map | Success % | Length (Mean±Std) | Turns (Mean±Std) | Clearance (Mean±Std) | Time (s)")
for map_name in results_hybrid:
    h = results_hybrid[map_name]
    p = results_pso[map_name]
    print(f"Hybrid | {map_name} | {h['Success']:.1f}% | {h['Length'][0]:.2f}±{h['Length'][1]:.2f} | {h['Turns'][0]:.2f}±{h['Turns'][1]:.2f} | {h['Clearance'][0]:.2f}±{h['Clearance'][1]:.2f} | {h['Time'][0]:.3f}±{h['Time'][1]:.3f}")
    print(f"MO-PSO | {map_name} | {p['Success']:.1f}% | {p['Length'][0]:.2f}±{p['Length'][1]:.2f} | {p['Turns'][0]:.2f}±{p['Turns'][1]:.2f} | {p['Clearance'][0]:.2f}±{p['Clearance'][1]:.2f} | {p['Time'][0]:.3f}±{p['Time'][1]:.3f}")

# =================================================
# FIGURE–1 : Test Environments + Hybrid Paths
# =================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
for ax, (title, grid) in zip(axs, maps):
    ax.imshow(grid, cmap='gray_r', origin='lower')
    path_hybrid = hybrid_astar_pso(grid, start, goal, num_particles=30, num_waypoints=9, max_iter=100)
    if path_hybrid is not None:
        path_arr = np.array(path_hybrid)
        ax.plot(path_arr[:,1], path_arr[:,0], 'r', linewidth=2)
    ax.scatter(start[1], start[0], c='green', s=60, edgecolors='black')
    ax.scatter(goal[1], goal[0], c='red', s=60, edgecolors='black')
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# =================================================
# FIGURE–2 : Path Comparison Map B
# =================================================
grid = map_b
path_a = multi_objective_astar(grid, start, goal)
path_pso = mo_pso_path_planner(grid, start, goal, num_particles=30, num_waypoints=7, max_iter=100)
path_hybrid = hybrid_astar_pso(grid, start, goal, num_particles=30, num_waypoints=9, max_iter=100)

plt.figure(figsize=(6,6))
plt.imshow(grid, cmap='gray_r', origin='lower')
if path_a is not None: plt.plot(np.array(path_a)[:,1], np.array(path_a)[:,0], 'b--', label='MO-A*')
if path_pso is not None: plt.plot(np.array(path_pso)[:,1], np.array(path_pso)[:,0], 'g-.', label='MO-PSO')
if path_hybrid is not None: plt.plot(np.array(path_hybrid)[:,1], np.array(path_hybrid)[:,0], 'r', linewidth=2, label='Hybrid')
plt.scatter(start[1], start[0], c='green', s=60, label='Start')
plt.scatter(goal[1], goal[0], c='red', s=60, label='Goal')
plt.title("FIGURE–2: Path Comparison (Map B)")
plt.legend(); plt.show()


# =================================================
# TABLE–5 : Weight Sensitivity (Hybrid only) + FIGURE–3 & FIGURE–4
# =================================================
w3_values = [4, 6, 10]
weight_results, paths_w3 = [], []

for w3 in w3_values:
    path = hybrid_astar_pso(map_b, start, goal, num_particles=30, num_waypoints=9, max_iter=100, w3=w3)
    if path is not None:
        path_arr = np.array(path)
        weight_results.append((w3, compute_path_length(path_arr), compute_turns(path_arr), compute_min_clearance(path_arr, map_b)))
        paths_w3.append((w3, path_arr))

print("\nTABLE–5 : Weight Sensitivity")
print("w3 | Length | Turns | Clearance")
for r in weight_results:
    print(f"{r[0]} | {r[1]:.2f} | {r[2]} | {r[3]:.2f}")


# FIGURE–3 : Weight Sensitivity Curves
plt.figure()
plt.plot([r[0] for r in weight_results], [r[1] for r in weight_results], 'o-', label="Path Length")
plt.plot([r[0] for r in weight_results], [r[3] for r in weight_results], 's-', label="Clearance")
plt.xlabel("w3"); plt.ylabel("Metric Value")
plt.title("FIGURE–3: Weight Sensitivity Curves")
plt.legend(); plt.show()

# =================================================
# TABLE–6 : Waypoint Sensitivity (Hybrid only) + FIGURE–5
# =================================================
waypoints_list = [5, 7, 9]
wp_results, paths_wp = [], []

for wp in waypoints_list:
    path = hybrid_astar_pso(map_b, start, goal, num_particles=30, num_waypoints=wp, max_iter=100)
    if path is not None:
        path_arr = np.array(path)
        wp_results.append((wp, compute_path_length(path_arr), compute_turns(path_arr), compute_min_clearance(path_arr, map_b)))
        paths_wp.append((wp, path_arr))

print("\nTABLE–6 : Waypoint Sensitivity")
print("Waypoints | Length | Turns | Clearance")
for r in wp_results:
    print(f"{r[0]} | {r[1]:.2f} | {r[2]} | {r[3]:.2f}")

# FIGURE–4 : Waypoint Sensitivity Paths
plt.figure(figsize=(6,6))
plt.imshow(map_b, cmap='gray_r', origin='lower')
for wp, path in paths_wp:
    plt.plot(path[:,1], path[:,0], label=f"{wp} waypoints")
plt.title("FIGURE–4: Waypoint Sensitivity Paths")
plt.legend(); plt.show()


# =================================================
# FIGURE–5 : Representative Hybrid Paths
# =================================================
fig, axs = plt.subplots(2, 2, figsize=(10,10))
axs = axs.flatten()
for ax, (title, grid) in zip(axs, maps):
    path = hybrid_astar_pso(grid, start, goal, num_particles=30, num_waypoints=9, max_iter=100)
    ax.imshow(grid, cmap='gray_r', origin='lower')
    if path is not None:
        path_arr = np.array(path)
        ax.plot(path_arr[:,1], path_arr[:,0], 'r', linewidth=2)
    ax.scatter(start[1], start[0], c='green', s=60)
    ax.scatter(goal[1], goal[0], c='red', s=60)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
