# Multi-Objective Mobile Robot Path Planning using Hybrid A*–PSO

This repository contains the implementation and experimental framework for the research project:

**“Multi-Objective Mobile Robot Path Planning Using a Hybrid A*–PSO Approach in Structured Grid Environments.”**

The project investigates hybrid deterministic–stochastic path planning techniques that combine classical graph search with swarm intelligence optimization.

---

## Project Overview

Mobile robot navigation in structured environments often requires balancing multiple objectives such as:

- Shortest path length
- Smooth trajectory
- Safe obstacle clearance

Traditional algorithms like **A*** guarantee feasibility but may produce sharp turns, while metaheuristic methods such as **PSO** provide smoother optimization but may lack stability.

This project proposes a **Hybrid A*–PSO framework** where:

1. **A*** computes a feasible global path
2. **PSO** refines the path to improve smoothness and safety

---

## Algorithms Implemented

- Multi-Objective A*
- Multi-Objective Particle Swarm Optimization
- Hybrid A*–PSO Path Planning

---

## Benchmark Environments

The algorithms are evaluated on four structured grid maps:

- Sparse obstacle map
- Dense obstacle map
- Narrow passage map
- Dead-end trap map

All maps are 20×20 grid environments.

---

## Evaluation Metrics

The algorithms are evaluated based on:

- Path Length
- Number of Turns
- Minimum Obstacle Clearance
- Computation Time
- Success Rate

Each stochastic algorithm is evaluated across **30 independent runs**.

---

## Research Paper

The complete research paper describing the methodology and experiments can be found in:

```
research_paper.pdf [LINK: https://drive.google.com/file/d/11mHyn_RpS36in5J4CN3XLKIkq5uDsJ1T/view?usp=drivesdk]
```

---

## Experimental Implementations

During the research phase, several experimental algorithms and early prototypes were developed. These are stored in:

```
research_experiments/
```

These codes were not included in the final evaluation results but are preserved for transparency and future research.

---

## Requirements

Example dependencies:

```
Python 3.9+
numpy
matplotlib
scipy
```

Install using:

```
pip install -r requirements.txt
```

---


## Future Work

Future extensions may include:

- Dynamic obstacle environments
- Real robot integration
- Continuous-space planning
- Reinforcement learning integration

---

## Author

Bhagya

---

## License

This project is released under the MIT License.
