"""
Hybrid A* + PSO Core Algorithm
--------------------------------
Workflow:
Grid → A* (feasible path) → Waypoint extraction → Multi-objective PSO refinement

Objectives:
1. Minimize path length
2. Minimize turning (smoothness)
3. Maximize obstacle clearance

Constraint:
- Hard collision avoidance
"""

import numpy as np
import heapq
from scipy.ndimage import distance_transform_edt


# =========================================================
# 1. A* ALGORITHM (CORE)
# =========================================================
def astar_core(grid, start, goal):
    rows, cols = grid.shape

    moves = [(-1,0),(1,0),(0,-1),(0,1),
             (-1,-1),(-1,1),(1,-1),(1,1)]

    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    open_list = []
    heapq.heappush(open_list, (0, tuple(start)))
    came_from = {}
    g_cost = {tuple(start): 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == tuple(goal):
            break

        for dx, dy in moves:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if grid[nx, ny] == 1:
                    continue

                neighbor = (nx, ny)
                new_cost = g_cost[current] + heuristic(current, neighbor)

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

    # Reconstruct path
    path = []
    node = tuple(goal)
    while node != tuple(start):
        path.append(node)
        node = came_from.get(node)
        if node is None:
            return None

    path.append(tuple(start))
    return np.array(path[::-1])


# =========================================================
# 2. WAYPOINT EXTRACTION
# =========================================================
def extract_waypoints(astar_path, num_waypoints):
    idx = np.linspace(0, len(astar_path) - 1,
                      num_waypoints + 2).astype(int)
    return astar_path[idx[1:-1]]


# =========================================================
# 3. COLLISION CHECKING
# =========================================================
def point_in_obstacle(grid, p):
    return grid[int(p[0]), int(p[1])] == 1


def segment_collision(grid, p1, p2):
    steps = int(np.linalg.norm(p2 - p1)) * 2
    for i in range(steps + 1):
        t = i / steps if steps != 0 else 0
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        if grid[x, y] == 1:
            return True
    return False


# =========================================================
# 4. MULTI-OBJECTIVE COST FUNCTION
# =========================================================
def path_cost(grid, clearance_map, path,
              w1=1.0, w2=2.0, w3=10.0):

    # Hard collision constraint
    for p in path:
        if point_in_obstacle(grid, p):
            return 1e6

    for i in range(len(path) - 1):
        if segment_collision(grid, path[i], path[i + 1]):
            return 1e6

    # Objective 1: Path length
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    # Objective 2: Turn penalty
    turns = 0
    for i in range(2, len(path)):
        v1 = path[i - 1] - path[i - 2]
        v2 = path[i] - path[i - 1]
        if not np.array_equal(v1, v2):
            turns += 1

    # Objective 3: Clearance penalty
    clearance_penalty = 0
    for p in path.astype(int):
        clearance_penalty += 1.0 / (clearance_map[p[0], p[1]] + 1e-5)

    return w1 * length + w2 * turns + w3 * clearance_penalty


# =========================================================
# 5. HYBRID A* + PSO CORE FUNCTION
# =========================================================
def hybrid_astar_pso(grid, start, goal,
                     num_particles=30,
                     num_waypoints=9,
                     max_iter=100,
                     w1=1.0, w2=2.5, w3=8.0):

    grid = grid.copy()
    start = np.array(start)
    goal = np.array(goal)

    clearance_map = distance_transform_edt(1 - grid)

    # ---------- A* Phase ----------
    astar_path = astar_core(grid, start, goal)
    if astar_path is None:
        return None

    base_wp = extract_waypoints(astar_path, num_waypoints)

    # ---------- PSO Initialization ----------
    particles = []
    velocities = []

    for _ in range(num_particles):
        wp = base_wp + np.random.uniform(-2, 2, base_wp.shape)
        wp = np.clip(wp, 0, grid.shape[0] - 1)
        particles.append(wp)
        velocities.append(np.zeros_like(wp))

    particles = np.array(particles)
    velocities = np.array(velocities)

    pbest = particles.copy()
    pbest_cost = np.full(num_particles, np.inf)

    gbest = None
    gbest_cost = np.inf

    w, c1, c2 = 0.6, 1.5, 1.5

    # ---------- PSO Loop ----------
    for _ in range(max_iter):
        for i in range(num_particles):
            path = np.vstack([start, particles[i], goal])
            cost = path_cost(grid, clearance_map, path, w1, w2, w3)

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = particles[i].copy()

            if cost < gbest_cost:
                gbest_cost = cost
                gbest = particles[i].copy()

        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, grid.shape[0] - 1)

    # ---------- Final Path ----------
    final_path = np.vstack([start, gbest, goal]).astype(int)
    return final_path
