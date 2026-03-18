import matplotlib.pyplot as plt
from Map import map_a, map_b, map_c, map_d
from Hybrid_AStar_PSO import hybrid_astar_pso

# -------------------------------------------------
# Fixed Start–Goal (MANDATORY for baseline figures)
# -------------------------------------------------
start = (0, 0)
goal  = (19, 19)

# -------------------------------------------------
# Map Collection
# -------------------------------------------------
maps = [
    ("Map A: Sparse Complex", map_a),
    ("Map B: Dense Obstacle", map_b),
    ("Map C: Narrow Passage", map_c),
    ("Map D: Dead-End / Trap", map_d)
]

# -------------------------------------------------
# Create 2x2 Figure
# -------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

# -------------------------------------------------
# Run Hybrid A* + PSO
# -------------------------------------------------
for ax, (title, grid) in zip(axs, maps):

    path = hybrid_astar_pso(
        grid,
        start,
        goal,
        num_particles=30,
        num_waypoints=9,
        max_iter=100
    )

    ax.imshow(grid, cmap='gray_r', origin='lower')

    if path is not None:
        ax.plot(path[:, 1], path[:, 0],
                'r', linewidth=2, label='Hybrid A* + PSO')

    ax.scatter(start[1], start[0],
               c='green', s=60, edgecolors='black', zorder=5)
    ax.scatter(goal[1], goal[0],
               c='red', s=60, edgecolors='black', zorder=5)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Journal-style border
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

# -------------------------------------------------
# Common Legend
# -------------------------------------------------
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
