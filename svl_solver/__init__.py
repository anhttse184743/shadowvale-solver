"""ShadowVale Enemy Coordination Engine (solver side).

Layout
------
model/      the coordination problem itself (instance, objective, feasibility)
qubo/       QUBO / Ising formulation of that problem            (research objective O1)
instances/  seeded procedural instance generator + benchmark suites (O3)
solvers/    classical and quantum-inspired solvers behind one interface (O2)
bench/      reproducible benchmark runner and metrics            (O4)
api/        thin localhost HTTP layer the Unity client talks to
"""

__version__ = "0.1.0"
