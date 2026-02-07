# Route Improvement Iteration Log

## Iteration 0: Baseline

**Route file:** `routes/route_001.json`
**Date:** 2026-02-07
**Distance:** 2286m (target 2000m)
**Steps:** 36 explore + 35 return nodes
**Novelty:** 57.7% (per display), but misleading — see analysis

### Analysis

#### 1. Convex polygon shape — FAIL

The route does not form a convex loop. It goes south along a footway, jogs east, continues south-southeast, turns west along a busy-road-adjacent footway, then continues south. The explore phase never arcs back toward the start point. The return path covers 989m (43% of total distance), meaning the planner's explore phase wandered away without looping back. Max distance from start reached 663m.

Shape is an inverted "J" or hook, not a loop.

#### 2. Zigzagging — PASS

No zigzag patterns detected. Bearings are consistent within each phase. Sharp turns represent genuine direction changes at intersections.

#### 3. Complex intersection traversal — FAIL

50% of steps (18/36) are under 15m. Multiple sequences of very short segments:
- Steps 0-4: five segments totaling 34m (start area)
- Steps 9-11: three segments totaling 14m (Dunkley Ave crossing)
- Steps 19-22: four segments totaling 16m (footway curve)

These are OSM nodes within single real-world paths, creating unnecessary route complexity.

#### 4. Busy road adjacency — CONCERN

Steps 27-32 walk 311m (24% of explore distance) along busy-road-adjacent footways paralleling Bay Road (secondary). The busy_road_proximity_penalty of 8 is too small to matter when no non-busy alternatives exist. Route does walk a 264m segment along a footway next to what appears to be Bay Road.

#### 5. Dead-end backtracking — PASS

Dead ends correctly avoided via the dead_end_penalty of 40.

#### 6. Return path — PRESENT but concerning

Return path triggered at budget_threshold. 35 return nodes, 989m distance. This is 43% of total route distance, indicating poor explore-phase loop shaping. Route overshot target by 286m (14.3%).

### Root Cause Analysis

**Primary issue: Road weights make novelty useless.** Footway=1 vs residential=100 means even a heavily-walked footway (score 1+15=16) always beats an unwalked residential street (score 100+0=100). The route is trapped on footways exclusively. Not a single named street was walked during the explore phase.

**Secondary issue: No convexity bias.** The planner has no mechanism to prefer directions that arc back toward start, causing the explore phase to wander linearly away.

**Tertiary issue: `break` at line 195.** When valid_options is empty, the planner breaks instead of routing home. This was left from debugging (the code after the break is unreachable).

### Priority for next iteration

1. **Restore return-home behavior** — Remove the debugging `break` at line 195 so the planner routes home when stuck instead of just stopping.
