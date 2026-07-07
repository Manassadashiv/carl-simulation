# Task Checklist - Apex Agent Architecture

## Phase 1: World Mapping
- `[x]` Create and initialize the `carl_mapping.py` module
- `[x]` Integrate `carl_mapping.py` into the main `carl_harvest.py` simulation loop
- `[x]` Test map caching and save/load persistence locally
- `[x]` Verify map confidence increases as Bob explores

## Phase 2: Path Planning
- `[x]` Create and initialize the `carl_planner.py` module
- `[x]` Integrate `carl_planner.py` into `carl_harvest.py` for target tracking and path generation

## Phase 3: The Apex State Manager (The Mode Switcher)
- `[x]` Create Mode Switcher logic (toggling between neural EXPLORER and A* SPEED_DEMON)
- `[x]` Integrate Braitenberg safety reflex safety triggers for state fallback
- `[x]` Implement metabolism & energy consumption constraints
