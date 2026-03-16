"""
Multi-Objective PSO-Based Path Planning (CORE MODULE)

Objectives:
1. Minimize path length
2. Minimize turning (smoothness)
3. Maximize obstacle clearance

NOTE:
- No map definition
- No plotting
- No fixed start–goal
- Designed to be imported and reused
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


# =================================================
# MAIN ENTRY FUNCTION (WHAT YOU WILL CALL)
# =================================================
def mo_pso_path_planner(
    grid,
    start,
    goal,
    num_particles=30,
    num_waypoints=7,
    max_iter=100,
    w1=1.0,
    w2=2.0,
    w3=10.0
):
    """
    Multi-Objective PSO path planner

    Returns:
        final_path (Nx2 numpy array) or None if failed
    """

    GRID_SIZE = grid.shape[0]
    start = np.array(start)
    goal = np.array(goal)

    # Clearance map (for safety objective)
    clearance_map = distance_transform_edt(1 - grid)

    # -------------------------------------------------
    # Collision Checking
    # -------------------------------------------------
    def point_in_obstacle(p):
        return grid[int(p[0]), int(p[1])] == 1

    def segment_collision(p1, p2):
        dist = int(np.linalg.norm(p2 - p1)) * 2
        for i in range(dist + 1):
            t = i / dist if dist != 0 else 0
            x = int(p1[0] + t * (p2[0] - p1[0]))
            y = int(p1[1] + t * (p2[1] - p1[1]))
            if grid[x, y] == 1:
                return True
        return False

    # -------------------------------------------------
    # Multi-Objective Cost Function
    # -------------------------------------------------
    def path_cost(path):

        # Hard collision penalty
        for p in path:
            if point_in_obstacle(p):
                return 1e6

        for i in range(len(path) - 1):
            if segment_collision(path[i], path[i + 1]):
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

    # -------------------------------------------------
    # PSO Initialization (Waypoint-Based)
    # -------------------------------------------------
    particles = []
    velocities = []

    for _ in range(num_particles):
        wp = np.linspace(start, goal, num_waypoints + 2)[1:-1]
        wp += np.random.uniform(-3, 3, wp.shape)
        wp = np.clip(wp, 0, GRID_SIZE - 1)
        particles.append(wp)
        velocities.append(np.zeros_like(wp))

    particles = np.array(particles)
    velocities = np.array(velocities)

    pbest = particles.copy()
    pbest_cost = np.full(num_particles, np.inf)

    gbest = None
    gbest_cost = np.inf

    w, c1, c2 = 0.6, 1.5, 1.5

    # -------------------------------------------------
    # PSO Core Loop
    # -------------------------------------------------
    for _ in range(max_iter):
        for i in range(num_particles):
            path = np.vstack([start, particles[i], goal])
            cost = path_cost(path)

            if cost < pbest_cost[i]:
                pbest_cost[i] = cost
                pbest[i] = particles[i].copy()

            if cost < gbest_cost:
                gbest_cost = cost
                gbest = particles[i].copy()

        # Velocity & position update
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest[i] - particles[i])
                + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, GRID_SIZE - 1)

    # -------------------------------------------------
    # Final Path
    # -------------------------------------------------
    if gbest is None:
        return None

    final_path = np.vstack([start, gbest, goal]).astype(int)
    return final_path
