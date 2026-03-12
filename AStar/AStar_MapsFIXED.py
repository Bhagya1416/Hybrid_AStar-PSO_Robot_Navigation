import matplotlib.pyplot as plt
from Map import map_a, map_b, map_c, map_d
from AStar_MultiObj import baseline_astar, multi_objective_astar

# -------------------------------
# Fixed Start–Goal
# -------------------------------
start = (2, 2)
goal = (17, 17)

# -------------------------------
# Map Collection
# -------------------------------
maps = [
    ("Map A: Sparse Complex", map_a),
    ("Map B: Dense Obstacle", map_b),
    ("Map C: Narrow Passage", map_c),
    ("Map D: Dead-End / Trap", map_d)
]

# -------------------------------
# Create 2x2 Figure
# -------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

# -------------------------------
# Run Algorithms & Plot
# -------------------------------
for ax, (title, grid) in zip(axs, maps):

    baseline_path = baseline_astar(grid, start, goal)
    mo_path = multi_objective_astar(
        grid, start, goal, w1=1.0, w2=2.0, w3=3.0
    )

    ax.imshow(grid, cmap='gray_r', origin='lower')

    if baseline_path:
        bx, by = zip(*baseline_path)
        ax.plot(by, bx, 'b--', linewidth=1.5, label='Baseline A*')

    if mo_path:
        mx, my = zip(*mo_path)
        ax.plot(my, mx, 'r', linewidth=2, label='MO-A*')

    # Smaller start & goal points
    ax.scatter(start[1], start[0], c='green', s=60,
               edgecolors='black', linewidths=0.8, zorder=5)
    ax.scatter(goal[1], goal[0], c='red', s=60,
               edgecolors='black', linewidths=0.8, zorder=5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Solid black border (journal style)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

# -------------------------------
# Common Legend (One Time)
# -------------------------------
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
