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

---

## Iteration 1: Restore return-home behavior

**Route file:** `routes/route_002.json`
**Date:** 2026-02-07
**Change:** Removed debugging `break` in `planner.py` so `no_valid_options` triggers route-home via shortest path (with logging).

### Result

Route is identical to baseline (route_001). The `no_valid_options` code path was not triggered — the route hits `budget_threshold` return before ever running out of valid options. The fix is still correct (the dead code was a bug), but it has no effect on this particular route.

**Return trigger:** `budget_threshold` (same as before)
**Distance:** 2286m (unchanged)

### Priority for next iteration

The baseline analysis identified road weights as the primary issue. With footway=1 and residential=100, the planner is trapped on footways. This needs to be addressed before other improvements can take effect. However, per the plan, the next improvement is **virtual edges (15m jumps)**, which could also help by allowing the route to cross between disconnected footway networks.

---

## Iteration 2: Virtual edges (15m jumps)

**Route file:** `routes/route_003.json`
**Date:** 2026-02-07
**Change:** Added virtual edges between nodes within 15m that aren't connected, filtering out edges crossing busy roads. Graph grew from 10,604 to 37,882 segments (27,278 virtual edges added).

### Result

| Metric | Route 001 | Route 003 | Change |
|--------|-----------|-----------|--------|
| Distance | 2286m | 2059m | -227m (closer to 2000m target) |
| Explore steps | 36 | 34 | -2 |
| Return nodes | 35 | 33 | -2 |
| Return distance | 989m | 1026m | +37m |
| Novelty | 57.7% | 87.9% | +30.2pp |
| New segments | 41 | 58 | +17 |
| Virtual edge steps | 0 | 14/34 (41%) | new |
| Short segments (<15m) | 18/36 (50%) | 19/34 (56%) | slightly worse |

### Analysis

**Novelty dramatically improved.** Virtual edges allow the planner to jump between disconnected footway networks, finding fresh paths it couldn't reach before.

**Route shape** is still not a convex loop. The explore phase goes south, then south-east, hitting Bay Road, then further south along footways, then turns west and south to Jack Road, then heads back north via virtual edges and Graham Road. The return path (1026m, 33 nodes) is still very long.

**Virtual edges dominate short segments.** 14 of 34 explore steps are virtual edges, almost all under 15m. These are real crossings between disconnected paths (not OSM intersection complexity), but they still clutter the route. The route viewer will show these as unnamed segments.

**Busy road adjacency** is reduced — only 2 steps show busy_road_adjacent=true (vs 6 in baseline).

### Issues remaining
1. No convexity bias — return path still ~50% of route
2. Short segments still >50% — intersection simplification needed
3. Route still footway-dominated (road_type=footway for most non-virtual steps)

### Priority for next iteration

**Convexity bias** — add a scoring component that favors directions arcing back toward start after passing the midpoint. This will shorten the return path and create better loop shapes.

---

## Iteration 3: Convexity bias

**Route file:** `routes/route_004.json`
**Date:** 2026-02-07
**Change:** Added convexity bias to `score_edge`. After 35% of target distance, penalty scales linearly with progress for moving away from start. Weight: 0.5 per meter of delta. Also fixed `self.walked_distance` sync in `calculate_full_route()` so `score_edge` has correct progress info.

### Result

| Metric | Route 003 | Route 004 | Change |
|--------|-----------|-----------|--------|
| Distance | 2059m | 1899m | -160m |
| Explore steps | 34 | 36 | +2 |
| Return nodes | 33 | 29 | -4 |
| Return distance | 1026m | 892m | -134m |
| Return % | 50% | 47% | -3pp |
| Novelty | 87.9% | 87.5% | -0.4pp |

### Analysis

**Return path shortened by 134m**, route total closer to target (1899 vs 2000). The convexity bias creates penalties of up to 4.63 at step 28 (67.5m segment heading south at 67% progress).

**U-turn introduced.** At step 31, a virtual edge u-turn (turn angle 179.7 degrees) appears because the convexity bias makes heading north (toward start) attractive. This creates an awkward physical route: go south past a point, then u-turn north via a virtual edge, then zigzag southwest.

**Core limitation.** The convexity bias works on footway-vs-footway decisions but can't redirect the route to residential streets (weight gap too large). The route shape improved modestly but the fundamental issue remains: the route explores linearly south through connected footway networks, then the convexity bias starts fighting the southward trend too late to create a true loop.

### Issues remaining
1. U-turns and zigzags from convexity fighting limited options
2. Short segments (56% under 15m) — intersection simplification needed
3. Route still footway-dominated

### Priority for next iteration

**Intersection simplification** — merge chains of short segments (< 15m) into single logical steps. This reduces route complexity and eliminates the micro-navigation problem.
