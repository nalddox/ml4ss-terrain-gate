# Idea Brief: Evidence-Gated Terrain Admission for a Wheeled Delivery Robot

**ECE 532 – Multimodal Agentic Learning for Sensing Systems | Fall 2026 Final Project**
**Author:** _[name]_ | **Date:** October 20, 2026

## Problem and Decision Boundary

A wheeled delivery robot must reach a goal GPS coordinate over a campus sidewalk graph. Route planning relies on map labels (e.g., OpenStreetMap `highway=steps`), which can be stale or wrong. A shortest-path planner that trusts a mislabeled edge may route the robot onto stairs it cannot traverse.

**Decision:** at each graph edge, may the robot commit to traversing it?

- **Act:** admit the edge and traverse.
- **Defer:** gather more evidence (approach, re-observe) before committing.
- **Repair:** reject the edge, remove it from the graph, and replan (e.g., reroute to a ramp).
- **Handoff:** no admissible route remains; escalate to a remote operator.

Costs are asymmetric: admitting stairs is catastrophic, while a false rejection costs a detour. The gate is fail-closed, so an edge is inadmissible unless evidence affirmatively supports it. The core question is whether a calibrated, multi-source policy can match a conservative "reject if any source objects" baseline on **false admission** while reducing **unnecessary rejection and deferral**.

## Modalities and Evidence Sources

| Source | Typed claim (example) | Risk relevance |
|---|---|---|
| RGB camera (semantic) | `terrain(e) = stairs, p = 0.91` | Primary hazard detection |
| Depth (stereo/LiDAR, geometric) | `max_step_height(e) = 0.14 m` | Independent physical check |
| Map prior (OSM / GND traversability map) | `tag(e) = walkway` | Planning-time prior; may be stale |
| GPS (+ IMU) | `on_edge(e), σ = 2.1 m` | Associates observations with the correct edge |

**Data:** GND (Global Navigation Dataset; CC0 1.0) supplies wheeled-robot campus traversals with GPS, a front RGB camera, LiDAR, IMU, and a human-labeled traversability map (including stairs) that serves as ground truth. SANPO (CC BY 4.0) supplies pixel-level stairs/curb segmentation for training the terrain model. OSM supplies an independent map prior.

## Plausible Checker

**Geometric step-height checker:** estimate the maximum ground-surface discontinuity on an edge from depth and reject it if the discontinuity exceeds the robot's traversable limit, regardless of the semantic label. This checker uses different evidence than the image classifier, so a high-risk edge is admitted only when both agree. A second checker for localization tests whether GPS uncertainty is small enough to attribute an observation to a single edge, since ramps often run parallel to stairs.

## CPU / Replay Fallback

- All sensor data is replayed from extracted, cached files. ROS bags are converted once with the pure-Python `rosbags` library, so no ROS install or network is needed after setup.
- Baseline is one small GND campus, subsampled to about 2 Hz.
- Terrain features use frozen, cached embeddings from a compact pretrained backbone, with a lightweight classifier head on CPU.
- **Minimum fallback:** a classical geometric checker, a map prior, and GPS, with no learned model. It still exercises the claims, gate, receipt, and failure injection end to end.

## Failure Injection (planned)

- Camera: blur, glare, low light.
- GPS: bias and drift.
- Map: stale or flipped edge labels.
- Distribution shift: segmenter trained on SANPO (human eye level), tested on GND (robot height).
- Adversarial: stair imagery that looks flat from overhead.

## Course Interfaces

- **A1:** cached terrain embeddings as representation evidence.
- **A2:** calibrated terrain confidence driving per-risk-tier admission thresholds.
- **A3:** typed causal claim attributing each rejection or detour to its source (e.g., "rejected due to geometric checker, not map").

## Questions for Feedback

1. Is a self-recorded held-out campus sequence (phone video + GPS, released with the repo) acceptable data under the "public, synthetic, or course-provided" rule?
2. This project shares a vision front-end with my Machine Vision course project. Is that acceptable if the ML4SS deliverables (claims, gate, policy, evaluation) are distinct?
