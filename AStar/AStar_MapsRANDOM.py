import matplotlib.pyplot as plt
from Map import map_a, map_b, map_c, map_d
from AStar_MultiObj import baseline_astar, multi_objective_astar

# -------------------------------------------------
# Function: Interactive Start–Goal Selection
# -------------------------------------------------
def select_start_goal_on_ax(ax, grid, title):
    ax.imshow(grid, cmap='gray_r', origin='lower')
    ax.set_title(
        title + "\nClick START (green) then GOAL (red)",
        fontsize=10,
        fontweight='bold'
    )
    ax.set_xticks([])
    ax.set_yticks([])

    points = []

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        r = int(round(event.ydata))
        c = int(round(event.xdata))

        # Prevent selecting obstacle cells
        if grid[r, c] == 1:
            print("Obstacle clicked — choose a free cell")
            return

        points.append((r, c))
        color = 'green' if len(points) == 1 else 'red'
        ax.scatter(c, r, c=color, s=60, edgecolors='black', linewidths=0.8, zorder=6)
        plt.draw()

        # Disconnect after selecting two points
        if len(points) == 2:
            fig.canvas.mpl_disconnect(cid)

    fig = ax.figure
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Wait until two points are selected
    while len(points) < 2:
        plt.pause(0.1)

    return points[0], points[1]

# -------------------------------------------------
# Maps
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

selected_pairs = []

# -------------------------------------------------
# Run Algorithms and Plot Paths
# -------------------------------------------------
for ax, (title, grid) in zip(axs, maps):

    # Mouse click selection
    start, goal = select_start_goal_on_ax(ax, grid, title)

    # Run Baseline A*
    baseline_path = baseline_astar(grid, start, goal)

    # Run Multi-Objective A*
    mo_path = multi_objective_astar(
        grid,
        start,
        goal,
        w1=1.0,
        w2=2.0,
        w3=3.0
    )

    # Plot Baseline path
    if baseline_path:
        bx, by = zip(*baseline_path)
        ax.plot(by, bx, 'b--', linewidth=1.5, label='Baseline A*')

    # Plot Multi-Objective path
    if mo_path:
        mx, my = zip(*mo_path)
        ax.plot(my, mx, 'r', linewidth=2, label='MO-A*')

    # Start & Goal markers (small, clean)
    ax.scatter(start[1], start[0], c='green', s=60, edgecolors='black', linewidths=0.8, zorder=6)
    ax.scatter(goal[1], goal[0], c='red', s=60, edgecolors='black', linewidths=0.8, zorder=6)

    # Black border (journal style)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # Save selected points
    selected_pairs.append((title, start, goal))

# -------------------------------------------------
# Common Legend
# -------------------------------------------------
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()

# -------------------------------------------------
# Print Selected Start–Goal Pairs (Reproducibility)
# -------------------------------------------------
print("\nUser-Defined Start–Goal Pairs:")
for item in selected_pairs:
    print(item)
