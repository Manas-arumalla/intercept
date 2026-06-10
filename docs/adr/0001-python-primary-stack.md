# ADR-0001 — Python as the primary implementation stack

- **Status:** Accepted
- **Date:** 2026-06-07

## Context

INTERCEPT must support, in one codebase: classical guidance (PN/APN/OGL), state estimation
(EKF/UKF/IMM), optimization-based control (MPC/trajectory optimization), reinforcement learning
(PPO/SAC/TD3 + MARL), game-theoretic solvers, and publication-quality visualization — plus a
reproducible benchmark harness. The project is a software-only portfolio/research artifact, so
ecosystem breadth, readability, and recruiter/reviewer legibility matter as much as performance.

Candidate stacks considered: Python; C++ core + Python bindings; Julia; MATLAB/Simulink.

## Decision

Use **Python (≥3.10) as the primary stack.** Rationale:

- Best-in-class, mature libraries across *every* layer the project needs: `numpy`/`scipy`, `python-control`,
  `filterpy`, `CasADi`+`acados` (MPC), `Gymnasium`/`PettingZoo`/`Stable-Baselines3`/CleanRL (RL),
  `Matplotlib`/`PyVista`/`Plotly` (viz). The verified reference repo `propNav` is pure NumPy.
- Fastest iteration for a research/benchmark project; most legible to the target audience.
- The heavy numerical kernels the project depends on (NumPy, CasADi/acados, PyTorch) are C/C++/Fortran
  under the hood, so the performance cost of Python orchestration is acceptable at point-mass
  fidelity and for offline Monte-Carlo.

**MATLAB/Simulink** is retained as an *optional* high-fidelity 6-DOF / autopilot cross-check in a
later phase (P9), not a parallel implementation track.

## Consequences

- (+) One language across all paradigms; minimal context-switching; rich tooling.
- (+) Easy packaging (`pip install -e .`), CI, and docs (MkDocs).
- (−) Pure-Python loops are slow; mitigate by vectorizing Monte-Carlo and pushing inner loops to
  NumPy / compiled solvers. If a real-time bottleneck appears, isolate it behind the existing
  interfaces and consider a compiled kernel (Cython/Numba/C++), without changing the architecture.
- (−) RL/MPC dependencies are heavy; kept as optional extras (`.[rl]`, `.[mpc]`) so the core stays
  lightweight (numpy/scipy/matplotlib/pyyaml only).
