import matplotlib.pyplot as plt
import numpy as np

def create_base_map(size=(20, 20)):
    return np.zeros(size)

def plot_map(ax, grid, title, start=(2, 2), goal=(17, 17)):
    # origin='lower' ensures (0,0) is bottom-left
    # interpolation='nearest' ensures obstacles appear as solid blocks
    ax.imshow(grid, cmap='Greys', origin='lower', interpolation='nearest')
    
    # Plot Start and Goal
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', zorder=5) 
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal', zorder=5)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Solid black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('black')

# Initialize Figure
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# --- Map A: Sparse Complex ---
map_a = create_base_map()
map_a[4:7, 4:7] = 1
map_a[13:16, 13:16] = 1
map_a[10:12, 5:7] = 1
map_a[5:7, 12:14] = 1
plot_map(axs[0, 0], map_a, "Map A: Sparse Complex")

# --- Map B: Dense Obstacle (Cluttered Free Space) ---
map_b = create_base_map()
# Manually placed solid blocks to ensure a "cluttered" but navigable path
blocks = [
    (3, 3, 2, 2), (3, 8, 2, 3), (2, 14, 3, 2),
    (7, 2, 2, 2), (7, 7, 3, 3), (7, 15, 2, 2),
    (11, 4, 2, 2), (12, 10, 3, 2), (11, 16, 3, 2),
    (15, 2, 2, 3), (16, 7, 2, 2), (15, 13, 3, 2)
]
for (r, c, h, w) in blocks:
    map_b[r:r+h, c:c+w] = 1
plot_map(axs[0, 1], map_b, "Map B: Dense Obstacle")


# --- Map C: Narrow Passage (With 2 Additional Obstacles) ---
map_c = create_base_map()
# Original corridor walls
map_c[8:10, 0:14] = 1
map_c[13:15, 6:20] = 1
# New Obstacle 1: At the start approach
map_c[3:6, 10:12] = 1
# New Obstacle 2: At the final goal approach
map_c[16:19, 8:10] = 1
plot_map(axs[1, 0], map_c, "Map C: Narrow Passage (Enhanced)")

# --- Map D: Dead-End / Trap ---
map_d = create_base_map()
map_d[8:16, 6:8] = 1   
map_d[8:16, 14:16] = 1 
map_d[15:17, 6:16] = 1 
map_d[4:6, 6:14] = 1
plot_map(axs[1, 1], map_d, "Map D: Dead-End / Trap")

plt.tight_layout()
plt.show()
