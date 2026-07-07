# Codebase Audit Report

This report lists potential loose ends, dummy code, blank files, misleading markers, and file redundancies across the codebase.

## 1. Empty Files (0 Bytes)
- `GENESIS\carl_harvest_recovered.py`
- `GENESIS\harvest_out.txt`
- `GENESIS_BIPED\carl_harvest_recovered.py`
- `GENESIS_BIPED\harvest_out.txt`
- `GENESIS_HAND\carl_harvest_recovered.py`
- `GENESIS_HAND\harvest_out.txt`

## 2. File Redundancy (Identical Duplicates)
Files in `GENESIS_BIPED` and `GENESIS_HAND` that are identical to their counterparts in `GENESIS` or other directories:

- Primary: `GENESIS\MUJOCO_LOG.TXT`
  - Duplicate: `GENESIS_BIPED\MUJOCO_LOG.TXT`
  - Duplicate: `GENESIS_HAND\MUJOCO_LOG.TXT`
- Primary: `GENESIS\blob_brain.py`
  - Duplicate: `GENESIS_BIPED\blob_brain.py`
  - Duplicate: `GENESIS_HAND\blob_brain.py`
- Primary: `GENESIS\blob_infancy.py`
  - Duplicate: `GENESIS_BIPED\blob_infancy.py`
  - Duplicate: `GENESIS_HAND\blob_infancy.py`
- Primary: `GENESIS\blob_telemetry.py`
  - Duplicate: `GENESIS_BIPED\blob_telemetry.py`
  - Duplicate: `GENESIS_HAND\blob_telemetry.py`
- Primary: `GENESIS\blob_toddler.py`
  - Duplicate: `GENESIS_BIPED\blob_toddler.py`
  - Duplicate: `GENESIS_HAND\blob_toddler.py`
- Primary: `GENESIS\blob_trainer.py`
  - Duplicate: `GENESIS_BIPED\blob_trainer.py`
  - Duplicate: `GENESIS_HAND\blob_trainer.py`
- Primary: `GENESIS\blob_world.xml`
  - Duplicate: `GENESIS_BIPED\blob_world.xml`
  - Duplicate: `GENESIS_HAND\blob_world.xml`
- Primary: `GENESIS\carl_active_inference.py`
  - Duplicate: `GENESIS_BIPED\carl_active_inference.py`
  - Duplicate: `GENESIS_HAND\carl_active_inference.py`
- Primary: `GENESIS\carl_agent.py`
  - Duplicate: `GENESIS_BIPED\carl_agent.py`
  - Duplicate: `GENESIS_HAND\carl_agent.py`
- Primary: `GENESIS\carl_arm_train.py`
  - Duplicate: `GENESIS_BIPED\carl_arm_train.py`
  - Duplicate: `GENESIS_HAND\carl_arm_train.py`
- Primary: `GENESIS\carl_bios.py`
  - Duplicate: `GENESIS_BIPED\carl_bios.py`
  - Duplicate: `GENESIS_HAND\carl_bios.py`
- Primary: `GENESIS\carl_body.xml`
  - Duplicate: `GENESIS_BIPED\carl_body.xml`
  - Duplicate: `GENESIS_HAND\carl_body.xml`
- Primary: `GENESIS\carl_brainstem.py`
  - Duplicate: `GENESIS_BIPED\carl_brainstem.py`
  - Duplicate: `GENESIS_HAND\carl_brainstem.py`
- Primary: `GENESIS\carl_cpg.py`
  - Duplicate: `GENESIS_BIPED\carl_cpg.py`
  - Duplicate: `GENESIS_HAND\carl_cpg.py`
- Primary: `GENESIS\carl_curiosity.py`
  - Duplicate: `GENESIS_BIPED\carl_curiosity.py`
  - Duplicate: `GENESIS_HAND\carl_curiosity.py`
- Primary: `GENESIS\carl_double_buffer.py`
  - Duplicate: `GENESIS_BIPED\carl_double_buffer.py`
  - Duplicate: `GENESIS_HAND\carl_double_buffer.py`
- Primary: `GENESIS\carl_eval.py`
  - Duplicate: `GENESIS_BIPED\carl_eval.py`
  - Duplicate: `GENESIS_HAND\carl_eval.py`
- Primary: `GENESIS\carl_expression.py`
  - Duplicate: `GENESIS_BIPED\carl_expression.py`
  - Duplicate: `GENESIS_HAND\carl_expression.py`
- Primary: `GENESIS\carl_genesis.html`
  - Duplicate: `GENESIS_BIPED\carl_genesis.html`
  - Duplicate: `GENESIS_HAND\carl_genesis.html`
- Primary: `GENESIS\carl_harvest.py`
  - Duplicate: `GENESIS_BIPED\carl_harvest.py`
  - Duplicate: `GENESIS_HAND\carl_harvest.py`
- Primary: `GENESIS\carl_harvest_found.py`
  - Duplicate: `GENESIS_BIPED\carl_harvest_found.py`
  - Duplicate: `GENESIS_HAND\carl_harvest_found.py`
- Primary: `GENESIS\carl_imagination.py`
  - Duplicate: `GENESIS_BIPED\carl_imagination.py`
  - Duplicate: `GENESIS_HAND\carl_imagination.py`
- Primary: `GENESIS\carl_mapping.py`
  - Duplicate: `GENESIS_BIPED\carl_mapping.py`
  - Duplicate: `GENESIS_HAND\carl_mapping.py`
- Primary: `GENESIS\carl_matrix_download.py`
  - Duplicate: `GENESIS_BIPED\carl_matrix_download.py`
  - Duplicate: `GENESIS_HAND\carl_matrix_download.py`
- Primary: `GENESIS\carl_maze_gen.py`
  - Duplicate: `GENESIS_BIPED\carl_maze_gen.py`
  - Duplicate: `GENESIS_HAND\carl_maze_gen.py`
- Primary: `GENESIS\carl_metabolism.py`
  - Duplicate: `GENESIS_BIPED\carl_metabolism.py`
  - Duplicate: `GENESIS_HAND\carl_metabolism.py`
- Primary: `GENESIS\carl_obstacle_controller.py`
  - Duplicate: `GENESIS_BIPED\carl_obstacle_controller.py`
  - Duplicate: `GENESIS_HAND\carl_obstacle_controller.py`
- Primary: `GENESIS\carl_planner.py`
  - Duplicate: `GENESIS_BIPED\carl_planner.py`
  - Duplicate: `GENESIS_HAND\carl_planner.py`
- Primary: `GENESIS\carl_social.py`
  - Duplicate: `GENESIS_BIPED\carl_social.py`
  - Duplicate: `GENESIS_HAND\carl_social.py`
- Primary: `GENESIS\carl_train.py`
  - Duplicate: `GENESIS_BIPED\carl_train.py`
  - Duplicate: `GENESIS_HAND\carl_train.py`
- Primary: `GENESIS\carl_train_multirate.py`
  - Duplicate: `GENESIS_BIPED\carl_train_multirate.py`
  - Duplicate: `GENESIS_HAND\carl_train_multirate.py`
- Primary: `GENESIS\diag_lidar.py`
  - Duplicate: `GENESIS_BIPED\diag_lidar.py`
  - Duplicate: `GENESIS_HAND\diag_lidar.py`
- Primary: `GENESIS\eval_loop.py`
  - Duplicate: `GENESIS_BIPED\eval_loop.py`
  - Duplicate: `GENESIS_HAND\eval_loop.py`
- Primary: `GENESIS\eval_results.txt`
  - Duplicate: `GENESIS_BIPED\eval_results.txt`
  - Duplicate: `GENESIS_HAND\eval_results.txt`
- Primary: `GENESIS\eval_results2.txt`
  - Duplicate: `GENESIS_BIPED\eval_results2.txt`
  - Duplicate: `GENESIS_HAND\eval_results2.txt`
- Primary: `GENESIS\fast_forward.py`
  - Duplicate: `GENESIS_BIPED\fast_forward.py`
  - Duplicate: `GENESIS_HAND\fast_forward.py`
- Primary: `GENESIS\found_line.txt`
  - Duplicate: `GENESIS_BIPED\found_line.txt`
  - Duplicate: `GENESIS_HAND\found_line.txt`
- Primary: `GENESIS\genesis_body.xml`
  - Duplicate: `GENESIS_BIPED\genesis_body.xml`
  - Duplicate: `GENESIS_HAND\genesis_body.xml`
- Primary: `GENESIS\genesis_physics.py`
  - Duplicate: `GENESIS_BIPED\genesis_physics.py`
  - Duplicate: `GENESIS_HAND\genesis_physics.py`
- Primary: `GENESIS\genesis_reservoir.py`
  - Duplicate: `GENESIS_BIPED\genesis_reservoir.py`
  - Duplicate: `GENESIS_HAND\genesis_reservoir.py`
- Primary: `GENESIS\genesis_run.py`
  - Duplicate: `GENESIS_BIPED\genesis_run.py`
  - Duplicate: `GENESIS_HAND\genesis_run.py`
- Primary: `GENESIS\genesis_telemetry.py`
  - Duplicate: `GENESIS_BIPED\genesis_telemetry.py`
  - Duplicate: `GENESIS_HAND\genesis_telemetry.py`
- Primary: `GENESIS\genesis_world.py`
  - Duplicate: `GENESIS_BIPED\genesis_world.py`
  - Duplicate: `GENESIS_HAND\genesis_world.py`
- Primary: `GENESIS\legacy_safe\carl_harvest.py`
  - Duplicate: `GENESIS_BIPED\legacy_safe\carl_harvest.py`
  - Duplicate: `GENESIS_HAND\legacy_safe\carl_harvest.py`
- Primary: `GENESIS\legacy_safe\vessel_kinetic.xml`
  - Duplicate: `GENESIS_BIPED\legacy_safe\vessel_kinetic.xml`
  - Duplicate: `GENESIS_HAND\legacy_safe\vessel_kinetic.xml`
- Primary: `GENESIS\make_eval.py`
  - Duplicate: `GENESIS_BIPED\make_eval.py`
  - Duplicate: `GENESIS_HAND\make_eval.py`
- Primary: `GENESIS\preseed_traits.py`
  - Duplicate: `GENESIS_BIPED\preseed_traits.py`
  - Duplicate: `GENESIS_HAND\preseed_traits.py`
- Primary: `GENESIS\scratch\check_maze.py`
  - Duplicate: `GENESIS_BIPED\scratch\check_maze.py`
  - Duplicate: `GENESIS_HAND\scratch\check_maze.py`
- Primary: `GENESIS\scratch\generate_adj_list.py`
  - Duplicate: `GENESIS_BIPED\scratch\generate_adj_list.py`
  - Duplicate: `GENESIS_HAND\scratch\generate_adj_list.py`
- Primary: `GENESIS\scratch\generate_maze.py`
  - Duplicate: `GENESIS_BIPED\scratch\generate_maze.py`
  - Duplicate: `GENESIS_HAND\scratch\generate_maze.py`
- Primary: `GENESIS\scratch\test_brain.py`
  - Duplicate: `GENESIS_BIPED\scratch\test_brain.py`
  - Duplicate: `GENESIS_HAND\scratch\test_brain.py`
- Primary: `GENESIS\scratch\verify_carl.py`
  - Duplicate: `GENESIS_BIPED\scratch\verify_carl.py`
  - Duplicate: `GENESIS_HAND\scratch\verify_carl.py`
- Primary: `GENESIS\test_final.py`
  - Duplicate: `GENESIS_BIPED\test_final.py`
  - Duplicate: `GENESIS_HAND\test_final.py`
- Primary: `GENESIS\test_speed.py`
  - Duplicate: `GENESIS_BIPED\test_speed.py`
  - Duplicate: `GENESIS_HAND\test_speed.py`
- Primary: `GENESIS\test_stability.py`
  - Duplicate: `GENESIS_BIPED\test_stability.py`
  - Duplicate: `GENESIS_HAND\test_stability.py`
- Primary: `GENESIS\vessel_kinetic.xml`
  - Duplicate: `GENESIS_BIPED\vessel_kinetic.xml`
  - Duplicate: `GENESIS_HAND\vessel_kinetic.xml`

## 3. Empty Python Classes & Functions (AST Analysis)
Functions or classes that have empty or stubbed bodies (e.g., only contain `pass`, docstring, or raise `NotImplementedError`):

No empty classes/functions found.

## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)
Instances of development keywords found in code comments or docstrings:

### `GENESIS\carl_bios.py`
- **Line 221** (`DUMMY`): `# Dummy workloads previously caused deadline misses by consuming`

### `GENESIS\legacy_safe\carl_harvest.py`
- **Line 249** (`DUMMY`): `cpg_torques = np.array([0.0, 0.0])  # dummy base`

### `GENESIS_BIPED\carl_bios.py`
- **Line 221** (`DUMMY`): `# Dummy workloads previously caused deadline misses by consuming`

### `GENESIS_BIPED\legacy_safe\carl_harvest.py`
- **Line 249** (`DUMMY`): `cpg_torques = np.array([0.0, 0.0])  # dummy base`

### `GENESIS_HAND\carl_bios.py`
- **Line 221** (`DUMMY`): `# Dummy workloads previously caused deadline misses by consuming`

### `GENESIS_HAND\legacy_safe\carl_harvest.py`
- **Line 249** (`DUMMY`): `cpg_torques = np.array([0.0, 0.0])  # dummy base`

### `scratch\audit_codebase.py`
- **Line 7** (`TEMP`): `REPORT_PATH = r"C:\Users\MANAS\AppData\Local\Temp\codebase_audit_report.md" # We will write it to the brain directory instead.`
- **Line 25** (`TODO`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`FIXME`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`STUB`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`DUMMY`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`PLACEHOLDER`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`NOTIMPLEMENTED`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`XXX`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`TEMP`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`BLANK`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 25** (`MISLEADING`): `KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']`
- **Line 152** (`DUMMY`): `f.write("This report lists potential loose ends, dummy code, blank files, misleading markers, and file redundancies across the codebase.\n\n")`
- **Line 152** (`BLANK`): `f.write("This report lists potential loose ends, dummy code, blank files, misleading markers, and file redundancies across the codebase.\n\n")`
- **Line 152** (`MISLEADING`): `f.write("This report lists potential loose ends, dummy code, blank files, misleading markers, and file redundancies across the codebase.\n\n")`
- **Line 186** (`TODO`): `f.write("## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)\n")`
- **Line 186** (`FIXME`): `f.write("## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)\n")`
- **Line 186** (`STUB`): `f.write("## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)\n")`
- **Line 186** (`DUMMY`): `f.write("## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)\n")`

