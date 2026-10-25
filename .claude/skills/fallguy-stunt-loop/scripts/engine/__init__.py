"""FALLGUY stunt engine: plan -> MuJoCo scene, Unitree H1 performer, phase choreography, physics, Gate-1 metrics.

There are no stunt types: plan.stunt is a sequence of generic phases (hold, move, jump, fall, air, land) that
engine.choreo interprets (see reference/contracts.md).

Modules (import what you need; importing the package itself pulls in nothing heavy):
  params     tunable specs from plan.stunt.tunables, clamp(), apply()
  scene      plan.json -> MJCF, normalize_plan(), from_legacy_plan()
  performer  Unitree H1 composition, H1Rig pose mapping, leg IK
  choreo     phase interpreter (kinematic hold/move/take-off, flight pose schedule, landing)
  sim        run(plan, params, out_dir) -> scene.xml, trajectory.npz, events.json
  metrics    compute(plan, params, out_dir) -> metrics.json (Gate 1 via scripts/gates.py)
  render     videos and stills from a finished sim
"""
ENGINE_VERSION = "2.0"
