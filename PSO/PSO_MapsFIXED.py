import matplotlib.pyplot as plt
from Map import map_a, map_b, map_c, map_d
from PSO_MultiObj import mo_pso_path_planner

# -------------------------------
# Fixed Start–Goal (MANDATORY SETUP)
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
# Run PSO on All Maps
# -------------------------------
for ax, (title, grid) in zip(axs, maps):

    # Run Multi-Objective PSO
    pso_path = mo_pso_path_planner(
        grid,
        start,
        goal,
        num_particles=30,
        num_waypoints=7,
        max_iter=100
    )

    # Plot map
    ax.imshow(grid, cmap='gray_r', origin='lower')

    # Plot PSO path
    if pso_path is not None:
        ax.plot(
            pso_path[:, 1],
            pso_path[:, 0],
            'r',
            linewidth=2,
            label='MO-PSO'
        )

    # Start & Goal (small, journal style)
    ax.scatter(
        start[1], start[0],
        c='green', s=50,
        edgecolors='black', linewidths=0.8, zorder=5
    )
    ax.scatter(
        goal[1], goal[0],
        c='red', s=50,
        edgecolors='black', linewidths=0.8, zorder=5
    )

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
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
