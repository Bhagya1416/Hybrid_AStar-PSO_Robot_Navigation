import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from Map import map_a, map_b, map_c, map_d
from PSO_MultiObj import mo_pso_path_planner

# =================================================
# FIXED SETTINGS
# =================================================
start = (0, 0)
goal  = (19, 19)
NUM_RUNS = 30

maps = [
    ("Map A: Sparse Complex", map_a),
    ("Map B: Dense Obstacle", map_b),
    ("Map C: Narrow Passage", map_c),
    ("Map D: Dead-End / Trap", map_d)
]

# =================================================
# METRIC FUNCTIONS
# =================================================
def compute_path_length(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def compute_turns(path):
    turns = 0
    for i in range(2, len(path)):
        v1 = path[i-1] - path[i-2]
        v2 = path[i] - path[i-1]
        if not np.array_equal(v1, v2):
            turns += 1
    return turns

def compute_min_clearance(path, grid):
    clearance_map = distance_transform_edt(1 - grid)
    return min(clearance_map[int(p[0]), int(p[1])] for p in path)

# =================================================
# FIGURE–3: WEIGHT SENSITIVITY PATHS (PSO)
# =================================================
weight_sets = [
    (1.0, 1.0, 5.0),
    (1.0, 2.0, 10.0),
    (2.0, 2.0, 15.0)
]

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
grid = map_b  # representative complex map

for ax, (w1, w2, w3) in zip(axs, weight_sets):
    path = mo_pso_path_planner(
        grid, start, goal,
        num_particles=30,
        num_waypoints=9,
        max_iter=100,
        w1=w1, w2=w2, w3=w3
    )

    ax.imshow(grid, cmap='gray_r', origin='lower')
    if path is not None:
        ax.plot(path[:,1], path[:,0], linewidth=2)
    ax.scatter(start[1], start[0], c='green', s=40)
    ax.scatter(goal[1], goal[0], c='red', s=40)
    ax.set_title(f"w1={w1}, w2={w2}, w3={w3}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("FIGURE–3: PSO Weight Sensitivity Paths")
plt.tight_layout()
plt.show()

# =================================================
# FIGURE–4: WEIGHT SENSITIVITY CURVES (PSO)
# =================================================
w3_values = [2, 5, 10, 15, 20]
avg_length = []
avg_turns = []
avg_clearance = []

for w3 in w3_values:
    L, T, C = [], [], []

    for _ in range(NUM_RUNS):
        path = mo_pso_path_planner(
            map_b, start, goal,
            num_particles=30,
            num_waypoints=9,
            max_iter=100,
            w1=1.0, w2=2.0, w3=w3
        )
        if path is not None:
            L.append(compute_path_length(path))
            T.append(compute_turns(path))
            C.append(compute_min_clearance(path, map_b))

    avg_length.append(np.mean(L))
    avg_turns.append(np.mean(T))
    avg_clearance.append(np.mean(C))

plt.figure(figsize=(6,4))
plt.plot(w3_values, avg_length, marker='o', label="Path Length")
plt.plot(w3_values, avg_turns, marker='s', label="Turns")
plt.plot(w3_values, avg_clearance, marker='^', label="Clearance")
plt.xlabel("Clearance Weight (w3)")
plt.ylabel("Metric Value")
plt.title("FIGURE–4: PSO Weight Sensitivity Curves")
plt.legend()
plt.grid(True)
plt.show()

# =================================================
# FIGURE–5 (OPTIONAL): WAYPOINT SENSITIVITY PATHS
# =================================================
waypoints_list = [5, 7, 9, 11]

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for ax, wp in zip(axs, waypoints_list):
    path = mo_pso_path_planner(
        map_c, start, goal,
        num_particles=30,
        num_waypoints=wp,
        max_iter=100
    )

    ax.imshow(map_c, cmap='gray_r', origin='lower')
    if path is not None:
        ax.plot(path[:,1], path[:,0], linewidth=2)
    ax.scatter(start[1], start[0], c='green', s=40)
    ax.scatter(goal[1], goal[0], c='red', s=40)
    ax.set_title(f"Waypoints = {wp}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("FIGURE–5 (Optional): PSO Waypoint Sensitivity")
plt.tight_layout()
plt.show()

# =================================================
# TABLE–4: STATISTICAL PERFORMANCE (PSO)
# =================================================
results_pso = {}

for map_name, grid in maps:
    L, T, C, Time = [], [], [], []
    success = 0

    for _ in range(NUM_RUNS):
        t0 = time.time()
        path = mo_pso_path_planner(grid, start, goal)
        if path is not None:
            success += 1
            L.append(compute_path_length(path))
            T.append(compute_turns(path))
            C.append(compute_min_clearance(path, grid))
            Time.append(time.time() - t0)

    results_pso[map_name] = {
        "Success": success / NUM_RUNS * 100,
        "Length": (np.mean(L), np.std(L)),
        "Turns": (np.mean(T), np.std(T)),
        "Clearance": (np.mean(C), np.std(C)),
        "Time": (np.mean(Time), np.std(Time))
    }

print("\nTABLE–4: MO-PSO PERFORMANCE (30 RUNS)")
for k, v in results_pso.items():
    print(f"\n{k}")
    print(f"Success   : {v['Success']:.1f}%")
    print(f"Length    : {v['Length'][0]:.2f} ± {v['Length'][1]:.2f}")
    print(f"Turns     : {v['Turns'][0]:.2f} ± {v['Turns'][1]:.2f}")
    print(f"Clearance : {v['Clearance'][0]:.2f} ± {v['Clearance'][1]:.2f}")
    print(f"Time (s)  : {v['Time'][0]:.3f} ± {v['Time'][1]:.3f}")

# =================================================
# TABLE–7: MULTI-MAP ROBUSTNESS
# =================================================
print("\nTABLE–7: MULTI-MAP ROBUSTNESS (PATH LENGTH)")
for k, v in results_pso.items():
    print(f"{k}: {v['Length'][0]:.2f}")
