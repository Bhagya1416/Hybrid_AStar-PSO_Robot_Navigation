import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.ndimage import distance_transform_edt
from Map import map_a, map_b, map_c, map_d
from Hybrid_AStar_PSO import hybrid_astar_pso

# =================================================
# FIXED SETTINGS
# =================================================
start = (0, 0)
goal  = (19, 19)

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
# FIGURE–1 : TEST ENVIRONMENTS + HYBRID PATHS
# =================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for ax, (title, grid) in zip(axs, maps):

    path = hybrid_astar_pso(
        grid, start, goal,
        num_particles=30,
        num_waypoints=9,
        max_iter=100
    )

    ax.imshow(grid, cmap='gray_r', origin='lower')

    if path is not None:
        ax.plot(path[:, 1], path[:, 0], 'r', linewidth=2)

    ax.scatter(start[1], start[0], c='green', s=60)
    ax.scatter(goal[1], goal[0], c='red', s=60)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

# =================================================
# TABLE–4 : STATISTICAL PERFORMANCE (30 RUNS)
# =================================================
results = {}

for map_name, grid in maps:

    L, T, C, Time = [], [], [], []
    success = 0

    for _ in range(NUM_RUNS):
        t0 = time.time()

        path = hybrid_astar_pso(
            grid, start, goal,
            num_particles=30,
            num_waypoints=9,
            max_iter=100
        )

        if path is not None:
            success += 1
            L.append(compute_path_length(path))
            T.append(compute_turns(path))
            C.append(compute_min_clearance(path, grid))
            Time.append(time.time() - t0)

    results[map_name] = {
        "Success": success / NUM_RUNS * 100,
        "Length": (np.mean(L), np.std(L)),
        "Turns": (np.mean(T), np.std(T)),
        "Clearance": (np.mean(C), np.std(C)),
        "Time": (np.mean(Time), np.std(Time))
    }

print("\nTABLE–4 : STATISTICAL PERFORMANCE")
for k, v in results.items():
    print(f"\n{k}")
    print(f"Success Rate : {v['Success']:.1f}%")
    print(f"Length       : {v['Length'][0]:.2f} ± {v['Length'][1]:.2f}")
    print(f"Turns        : {v['Turns'][0]:.2f} ± {v['Turns'][1]:.2f}")
    print(f"Clearance    : {v['Clearance'][0]:.2f} ± {v['Clearance'][1]:.2f}")
    print(f"Time (s)     : {v['Time'][0]:.3f} ± {v['Time'][1]:.3f}")

# =================================================
# TABLE–5 + FIGURE–3 + FIGURE–4 : WEIGHT SENSITIVITY
# =================================================
w3_values = [4, 6, 10]
grid = map_b  # recommended (dense map)

weight_results = []
paths_w3 = []

for w3 in w3_values:
    path = hybrid_astar_pso(
        grid, start, goal,
        num_particles=30,
        num_waypoints=9,
        max_iter=100,
        w3=w3
    )

    if path is not None:
        weight_results.append((
            w3,
            compute_path_length(path),
            compute_turns(path),
            compute_min_clearance(path, grid)
        ))
        paths_w3.append((w3, path))

print("\nTABLE–5 : WEIGHT SENSITIVITY")
for r in weight_results:
    print(f"w3={r[0]} | Length={r[1]:.2f} | Turns={r[2]} | Clearance={r[3]:.2f}")

# FIGURE–3
plt.figure(figsize=(6,6))
plt.imshow(grid, cmap='gray_r', origin='lower')
for w3, path in paths_w3:
    plt.plot(path[:,1], path[:,0], label=f"w3={w3}")
plt.legend()
plt.title("FIGURE–3: Weight Sensitivity Paths")
plt.show()

# FIGURE–4
plt.figure()
plt.plot([r[0] for r in weight_results],
         [r[1] for r in weight_results], marker='o', label="Path Length")
plt.plot([r[0] for r in weight_results],
         [r[3] for r in weight_results], marker='s', label="Clearance")
plt.xlabel("w3")
plt.ylabel("Metric Value")
plt.title("FIGURE–4: Weight Sensitivity Curves")
plt.legend()
plt.show()

# =================================================
# TABLE–6 + FIGURE–5 : WAYPOINT SENSITIVITY
# =================================================
waypoints = [5, 7, 9]
wp_results = []
paths_wp = []

for wp in waypoints:
    path = hybrid_astar_pso(
        grid, start, goal,
        num_particles=30,
        num_waypoints=wp,
        max_iter=100
    )

    if path is not None:
        wp_results.append((
            wp,
            compute_path_length(path),
            compute_turns(path),
            compute_min_clearance(path, grid)
        ))
        paths_wp.append((wp, path))

print("\nTABLE–6 : WAYPOINT SENSITIVITY")
for r in wp_results:
    print(f"Waypoints={r[0]} | Length={r[1]:.2f} | Turns={r[2]} | Clearance={r[3]:.2f}")

# FIGURE–5 (Optional)
plt.figure(figsize=(6,6))
plt.imshow(grid, cmap='gray_r', origin='lower')
for wp, path in paths_wp:
    plt.plot(path[:,1], path[:,0], label=f"{wp} waypoints")
plt.legend()
plt.title("FIGURE–5: Waypoint Sensitivity Paths")
plt.show()

# =================================================
# TABLE–7 : MULTI-MAP ROBUSTNESS (HYBRID)
# =================================================
print("\nTABLE–7 : MULTI-MAP ROBUSTNESS (PATH LENGTH)")
for map_name, stats in results.items():
    print(f"{map_name} : {stats['Length'][0]:.2f}")
