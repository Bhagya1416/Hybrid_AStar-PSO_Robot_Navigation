import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


# -------------------------------
# 1. Helper Functions
# -------------------------------
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node, grid):
    GRID_SIZE = grid.shape[0]
    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]
    neighbors = []

    for dx, dy in moves:
        nx, ny = node[0] + dx, node[1] + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if grid[nx, ny] == 0:
                neighbors.append((nx, ny))
    return neighbors

def count_turns(path):
    turns = 0
    for i in range(2, len(path)):
        v1 = (path[i-1][0] - path[i-2][0],
              path[i-1][1] - path[i-2][1])
        v2 = (path[i][0] - path[i-1][0],
              path[i][1] - path[i-1][1])
        if v1 != v2:
            turns += 1
    return turns

# -------------------------------
# 2. Multi-Objective A*
# -------------------------------
def multi_objective_astar(grid, start, goal, w1=1.0, w2=2.0, w3=2.0):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    # Clearance map
    clearance_map = distance_transform_edt(1 - grid)

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            break

        for neighbor in get_neighbors(current, grid):
            step_cost = heuristic(current, neighbor)

            # Clearance penalty
            clearance_penalty = 1.0 / (clearance_map[neighbor] + 1e-5)

            tentative_g = g_cost[current] + w1 * step_cost + w3 * clearance_penalty

            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                priority = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_list, (priority, neighbor))
                came_from[neighbor] = current

    # Path reconstruction
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return None
    path.append(start)
    return path[::-1]

# -------------------------------
# 3. Baseline A*
# -------------------------------
def baseline_astar(grid, start, goal):
    return multi_objective_astar(grid, start, goal, w1=1.0, w2=0.0, w3=0.0)

# -------------------------------
# 4. Visualization (Optional)
# -------------------------------
def plot_paths(grid, start, goal, baseline_path, mo_path, title):
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='gray_r', origin='lower')

    if baseline_path:
        bx, by = zip(*baseline_path)
        plt.plot(by, bx, 'b--', label='Baseline A*')

    if mo_path:
        mx, my = zip(*mo_path)
        plt.plot(my, mx, 'r', linewidth=2, label='MO-A*')

    plt.scatter(start[1], start[0], c='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], c='red', s=100, label='Goal')

    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.show()
