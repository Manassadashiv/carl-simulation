"""
cleanup_workspace.py — Organizes the workspace by archiving legacy CARL v4 (Bob) files,
moving standalone arm files to standalone_arm/ (updating their paths dynamically),
and cleaning the root directory.
"""

import os
import shutil

ROOT_DIR = "D:/carl_simulation/GENESIS"
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive_v4")
STANDALONE_DIR = os.path.join(ROOT_DIR, "standalone_arm")

# Directories to create
DIRS_TO_CREATE = [
    os.path.join(ARCHIVE_DIR, "legacy_scripts"),
    os.path.join(ARCHIVE_DIR, "old_arm_demos"),
    os.path.join(ARCHIVE_DIR, "logs"),
    os.path.join(ARCHIVE_DIR, "csv_metrics"),
    STANDALONE_DIR,
]

# Legacy scripts to move
LEGACY_SCRIPTS = [
    "carl_harvest.py",
    "carl_harvest_found.py",
    "carl_genesis.html",
    "carl_maze_gen.py",
    "carl_train.py",
    "carl_train_multirate.py",
    "carl_expression.py",
    "carl_habits.py",
    "carl_social.py",
    "carl_dmn.py",
    "carl_imagination.py",
    "carl_crucible.py",
    "carl_behavior.py",
    "carl_workspace.py",
    "carl_metrics.py",
    "blob_telemetry.py",
    "carl_active_inference.py",
    "run_experiments.py",
    "run_remaining_experiments.py",
    "analyze_emergence.py",
]

# Old arm demonstration/training files
OLD_ARM_DEMOS = [
    "carl_arm_bc.py",
    "carl_arm_dual.py",
    "carl_arm_train.py",
    "carl_dual_arm_demo.py",
    "carl_jenga_demo.py",
    "carl_industrial_motion.py",
]

# Log and text files
LOG_FILES = [
    "emergence_metrics.log",
    "metabolic_ecology.log",
    "harvest_run.log",
    "telemetry_bios.log",
    "train.log",
    "debug.log",
    "jenga_out.txt",
    "MUJOCO_LOG.TXT",
]

# CSV files containing legacy metrics
CSV_FILES = [
    "emergence_metrics_control.csv",
    "emergence_metrics_no_culture.csv",
    "emergence_metrics_no_goals.csv",
    "emergence_metrics_no_inheritance.csv",
    "emergence_metrics_no_persistence.csv",
    "metabolic_ecology_control.csv",
    "metabolic_ecology_no_goals.csv",
    "metabolic_ecology_no_inheritance.csv",
    "metabolic_ecology_no_persistence.csv",
    "evolutionary_fossil_control.csv",
    "evolutionary_fossil_no_culture.csv",
    "evolutionary_fossil_no_goals.csv",
    "evolutionary_fossil_no_inheritance.csv",
    "evolutionary_fossil_no_persistence.csv",
    "evolutionary_fossil_record.csv",
]

# Directories to move to archive_v4/legacy_folders
LEGACY_DIRS = [
    "harvest_dataset",
    "harvest_debug",
    "harvest_test",
    "legacy",
    "legacy_safe",
]

def main():
    print("=" * 60)
    print("  CARL Workspace Organizer — Reorganization Suite")
    print("=" * 60)

    # 1. Create target subdirectories
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)
    os.makedirs(os.path.join(ARCHIVE_DIR, "legacy_folders"), exist_ok=True)

    # 2. Re-route and move Standalone Arm files
    # Move and update carl_arm_standalone_train.py
    src_train = os.path.join(ROOT_DIR, "carl_arm_standalone_train.py")
    dst_train = os.path.join(STANDALONE_DIR, "carl_arm_standalone_train.py")
    moved_standalone = 0
    if os.path.exists(src_train):
        with open(src_train, 'r') as f:
            content = f.read()
        # Update save path to memory folder in parent directory
        content = content.replace('"memory/carl_arm_standalone_weights_v2.npz"', '"../memory/carl_arm_standalone_weights_v2.npz"')
        with open(dst_train, 'w') as f:
            f.write(content)
        os.remove(src_train)
        moved_standalone += 1
        print("  -> Moved and patched carl_arm_standalone_train.py -> standalone_arm/")

    # Move and update carl_arm_standalone_eval.py
    src_eval = os.path.join(ROOT_DIR, "carl_arm_standalone_eval.py")
    dst_eval = os.path.join(STANDALONE_DIR, "carl_arm_standalone_eval.py")
    if os.path.exists(src_eval):
        with open(src_eval, 'r') as f:
            content = f.read()
        # Update weights path to memory folder in parent directory
        content = content.replace('"memory/carl_arm_standalone_weights_v2.npz"', '"../memory/carl_arm_standalone_weights_v2.npz"')
        with open(dst_eval, 'w') as f:
            f.write(content)
        os.remove(src_eval)
        moved_standalone += 1
        print("  -> Moved and patched carl_arm_standalone_eval.py -> standalone_arm/")

    # Move carl_arm_standalone.xml
    src_xml = os.path.join(ROOT_DIR, "carl_arm_standalone.xml")
    dst_xml = os.path.join(STANDALONE_DIR, "carl_arm_standalone.xml")
    if os.path.exists(src_xml):
        shutil.move(src_xml, dst_xml)
        moved_standalone += 1
        print("  -> Moved carl_arm_standalone.xml -> standalone_arm/")

    # Helper function to move files to a category directory
    def move_files(files, category_dir):
        count = 0
        for f in files:
            src = os.path.join(ROOT_DIR, f)
            dst = os.path.join(category_dir, f)
            if os.path.exists(src):
                try:
                    shutil.move(src, dst)
                    count += 1
                except Exception as e:
                    print(f"Error moving {f}: {e}")
        return count

    # 3. Move categories
    moved_scripts = move_files(LEGACY_SCRIPTS, os.path.join(ARCHIVE_DIR, "legacy_scripts"))
    moved_arms = move_files(OLD_ARM_DEMOS, os.path.join(ARCHIVE_DIR, "old_arm_demos"))
    moved_logs = move_files(LOG_FILES, os.path.join(ARCHIVE_DIR, "logs"))
    moved_csvs = move_files(CSV_FILES, os.path.join(ARCHIVE_DIR, "csv_metrics"))

    # 4. Move folders
    moved_folders = 0
    for folder in LEGACY_DIRS:
        src = os.path.join(ROOT_DIR, folder)
        dst = os.path.join(ARCHIVE_DIR, "legacy_folders", folder)
        if os.path.exists(src):
            try:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.move(src, dst)
                moved_folders += 1
            except Exception as e:
                print(f"Error moving folder {folder}: {e}")

    # Print summary report
    print("\nWorkspace Reorganization Complete:")
    print(f"  -> Re-routed and moved {moved_standalone} standalone arm assets to standalone_arm/")
    print(f"  -> Moved {moved_scripts} legacy scripts to archive_v4/legacy_scripts/")
    print(f"  -> Moved {moved_arms} old arm demo files to archive_v4/old_arm_demos/")
    print(f"  -> Moved {moved_logs} log files to archive_v4/logs/")
    print(f"  -> Moved {moved_csvs} CSV metric sheets to archive_v4/csv_metrics/")
    print(f"  -> Moved {moved_folders} legacy folders to archive_v4/legacy_folders/")
    print("\nAll active CARL Primate Scout control scripts are now cleanly presented in the root!")
    print("=" * 60)

if __name__ == "__main__":
    main()
