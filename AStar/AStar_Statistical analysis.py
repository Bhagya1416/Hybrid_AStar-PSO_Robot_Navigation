import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from Map import map_a, map_b, map_c, map_d
from AStar_MultiObj import multi_objective_astar

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
# METRICS
# =================================================
def compute_path_length(path):
    path = np.array(path)
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

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
# FIGURE–2 / FIGURE–6 STYLE: PATH EXAMPLES
# =================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

for ax, (title, grid) in zip(axs, maps):
    path = multi_objective_astar(grid, start, goal)
    ax.imshow(grid, cmap='gray_r', origin='lower')
    if path:
        p = np.array(path)
        ax.plot(p[:,1], p[:,0], 'b', linewidth=2)
    ax.scatter(start[1], start[0], c='green', s=60)
    ax.scatter(goal[1], goal[0], c='red', s=60)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("MO-A*: Multi-Map Path Examples")
plt.tight_layout()
plt.show()

# =================================================
# TABLE–4 STYLE STATS (DETERMINISTIC)
# =================================================
results_astar = {}

for map_name, grid in maps:
    lengths, turns, clears, times = [], [], [], []
    success = 0

    for _ in range(NUM_RUNS):
        t0 = time.time()
        path = multi_objective_astar(grid, start, goal)
        if path:
            success += 1
            lengths.append(compute_path_length(path))
            turns.append(compute_turns(path))
            clears.append(compute_min_clearance(path, grid))
            times.append(time.time() - t0)

    results_astar[map_name] = {
        "Success": success / NUM_RUNS * 100,
        "Length": (np.mean(lengths), np.std(lengths)),
        "Turns": (np.mean(turns), np.std(turns)),
        "Clearance": (np.mean(clears), np.std(clears)),
        "Time": (np.mean(times), np.std(times))
    }

print("\nMO-A* PERFORMANCE (DETERMINISTIC)")
for k, v in results_astar.items():
    print(f"\n{k}")
    print(f"Success   : {v['Success']:.1f}%")
    print(f"Length    : {v['Length'][0]:.2f}")
    print(f"Turns     : {v['Turns'][0]:.2f}")
    print(f"Clearance : {v['Clearance'][0]:.2f}")
    print(f"Time (s)  : {v['Time'][0]:.4f}")

# =================================================
# TABLE–7 : MULTI-MAP ROBUSTNESS
# =================================================
print("\nTABLE–7: MULTI-MAP ROBUSTNESS (PATH LENGTH)")
for k, v in results_astar.items():
    print(f"{k}: {v['Length'][0]:.2f}")
