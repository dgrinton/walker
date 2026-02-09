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

---

## Iteration 4: Intersection simplification + road weight rebalance

**Route file:** `routes/route_005.json`
**Date:** 2026-02-08
**Changes:**
1. Road weights rebalanced by user: footway=1, residential=2, service=3, tertiary=5, secondary=7, primary=9 (was footway=1, everything else=100)
2. Degree-2 node simplification: merge same-way, same-type segments under 20m at degree-2 nodes
3. Virtual edges disabled (caused u-turn spam with new weights)

### Result

| Metric | Route 004 | Route 005 | Change |
|--------|-----------|-----------|--------|
| Distance | 1899m | 1885m | -14m |
| Explore steps | 36 | 52 | +16 |
| Return nodes | 29 | 29 | same |
| Return distance | 892m | 844m | -48m |
| Return % | 47% | 45% | -2pp |
| Novelty | 87.5% | 82.5% | -5pp |
| Short segments (<15m) | - | 30/52 (58%) | improved from 80% pre-simplification |
| Named streets used | 0 | 5 (Thistle Grove, Graham Rd, Highett Rd, Train St, Livingston St, Worthing Rd) | major improvement |
| Nodes simplified | 0 | 2876 | new |

### Analysis

**Road weight rebalance is the biggest win.** The route now uses residential streets (4 segments), tertiary roads (1), and named streets. This was the #1 issue from the baseline analysis and its resolution completely changes route character.

**Simplification removed 2876 degree-2 nodes** (9051→6175). The same-way constraint prevents disconnecting paths that meet at degree-2 junctions between different OSM ways.

**Virtual edges disabled.** With balanced weights, the planner naturally uses residential streets to cross between footway networks. Virtual edges caused severe u-turn spam (40+ u-turns in 128 steps) because pairs of close nodes created bidirectional shortcuts the planner oscillated between.

**Short segments remain at 58%.** Same-way merging catches chains within a single OSM way, but many short segments occur at way boundaries (where two different OSM ways meet). These need cross-way simplification or a planner-level waypoint-merging pass.

**Return path still 45%.** The `no_valid_options` trigger fires at 1042m explore distance. Walk buffers block the route before the budget threshold kicks in.

### Issues remaining
1. Return path 45% — buffers too aggressive or convexity too strong/early
2. Short segments 58% — cross-way simplification needed
3. Route only 1885m vs 2000m target

### Priority for next iteration

**Relax walk buffers** — the 50m buffer width may be too aggressive now that the planner uses streets. Consider reducing buffer width or raising the min-length threshold to allow more routing freedom.

---

## Iteration 5: Better simplification + buffer tuning

**Route file:** `routes/route_006.json`
**Date:** 2026-02-08
**Changes:**
1. Removed merged-length cap in simplification: same-way chains are fully contracted regardless of resulting segment length (was capped at 20m)
2. Buffer width reduced from 50m to 30m

### Result

| Metric | Route 005 | Route 006 | Change |
|--------|-----------|-----------|--------|
| Distance | 1885m | 2285m | +400m |
| Explore steps | 52 | 41 | -11 |
| Return trigger | no_valid_options | budget_threshold | proper termination |
| Return distance | 844m | 872m | +28m |
| Return % | 45% | 38% | -7pp |
| Short segments | 30/52 (58%) | 18/41 (44%) | -14pp |
| Road types | 47 footway, 4 res | 15 footway, 25 res, 1 tert | streets dominant |
| Novelty | 82.5% | 66% | -16.5pp |
| Nodes simplified | 2876 | 3409 | +533 |

### Analysis

**Full chain simplification fixed the routing.** Removing the merged-length cap means entire same-way chains of short segments become single edges. Graph shrank from 6175 to 5642 nodes. Previously the planner got stuck at degree-2 nodes; now those nodes are eliminated.

**Return trigger is now budget_threshold** — the planner explores 1413m before routing home via 872m return. No longer hits `no_valid_options`. This is the correct behavior.

**Route heavily uses residential streets** (25/41 steps). This is the combined effect of the weight rebalance (residential=2 vs old 100) and better graph simplification. Novelty drops to 66% because residential streets around the start were previously walked.

**Short segments at 44%** — still high but improved. Remaining short segments are at way boundaries where different OSM ways meet. These can't be simplified with same-way constraint.

### Issues remaining
1. Return path 38% of route — convexity bias could be tuned
2. Short segments at 44% — cross-way simplification could help
3. Novelty 66% — lower than footway-only routes, but more useful exploration

### Priority for next iteration

The system is now functional and producing reasonable routes. Pausing for evaluation.

---

## Iteration 6: Loop steering + scenic return + OSM caching

**Route files:** `routes/route_023.json` (5km), `routes/route_024.json` (2km)
**Date:** 2026-02-08
**Changes:**
1. **Loop steering** replaces convexity bias: after onset (15% of distance), penalizes edges proportional to their absolute distance from start (not just delta). Creates strong homeward pull in second half.
2. **Scenic return path**: `_return_path_weight` now penalizes historically-walked segments (5x for route-used, 1+0.5*walks for walked), making the return explore novel streets.
3. **Buffer relaxation after onset**: Walk buffers are disabled after 15% of the walk. Anti-zigzag protection remains for early steps; later the planner relies on `visited_node_penalty` + `novelty_factor`.
4. **Buffer width scaling**: For walks > 2km, buffer width scales down (`max(15, 30 * 2000/target)`). A 5km walk uses 15m buffers.
5. **Visited-node penalty**: `visited_node_penalty=15` per prior visit to target node (trap avoidance).
6. **OSM data caching**: Responses cached to `osm_cache/` for 7 days. A covering cache (larger radius, nearby center) is reused. Eliminates repeated API calls during testing.
7. **Scaled fetch radius**: `max(1500, target_distance * 0.6)` — 5km walk fetches 3000m radius, 10km fetches 6000m.
8. **Scaled Overpass timeout**: `max(30, radius / 50)` seconds for larger queries.

### Result (5km walk)

| Metric | Route 006 (2km) | Route 023 (5km) | Route 024 (2km) |
|--------|-----------------|-----------------|-----------------|
| Distance | 2285m | 4984m | 1802m |
| Target | 2000m | 5000m | 2000m |
| Utilization | 114% | 99.7% | 90% |
| Explore distance | 1413m | 4648m | 1734m |
| Return distance | 872m (38%) | 337m (7%) | 68m (4%) |
| Return trigger | budget_threshold | budget_threshold | no_valid_options |
| Max dist from start | 411m | 808m | 435m |
| Dist at explore end | 411m | 226m | 60m |
| Return shared segs | ? | 0/12 (0%) | ? |

### Analysis

**Loop steering is the breakthrough.** The distance-from-start penalty creates a smooth trajectory: explore outward for the first third, then the increasing penalty curves the route homeward. At 5km, the explore phase covers 4648m and ends just 226m from start — the route nearly closes on its own with only 337m shortest-path return needed.

**Return path at 7% (5km) and 4% (2km)** — dramatically better than the previous 33-38%. The route IS the loop now, not an out-and-back with a long return.

**Buffer relaxation was critical.** Without it, the route ran out of options at ~1300m regardless of target distance. Disabling buffers after onset (15%) and relying on visited_node_penalty + novelty scoring prevents both zigzagging and boxing-in.

**OSM caching eliminates repeated API calls.** Cache covers any request within the cached circle, with 7-day expiry. The covering-circle check (`cached_radius >= dist + requested_radius`) means a 3000m cache also covers any 1500m request from the same area.

### Lessons learned

- Distance-based penalty (abs dist to start) >> delta-based (moving away/toward) for loop steering
- Tangent-based steering doesn't work in grid road networks
- Walk buffers must be relaxed for longer walks or the route boxes itself in
- OSM fetch radius must scale with target distance
- The 2km target was too short to evaluate loop quality — 5km+ reveals the real behavior
