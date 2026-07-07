"""
restore_active_deps.py — Restores files that are actively imported by carl_agent.py
and carl_eval.py back to the root directory to fix import errors.
"""

import os
import shutil

ROOT_DIR = "D:/carl_simulation/GENESIS"
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive_v4/legacy_scripts")

NECESSARY_FILES = [
    "carl_active_inference.py",
    "carl_dmn.py",
    "carl_habits.py",
    "carl_imagination.py",
    "carl_workspace.py",
    "carl_expression.py",
    "blob_telemetry.py",
]

def main():
    print("=" * 60)
    print("  Restoring Active Dependencies back to Root")
    print("=" * 60)
    
    restored = 0
    for f in NECESSARY_FILES:
        src = os.path.join(ARCHIVE_DIR, f)
        dst = os.path.join(ROOT_DIR, f)
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                print(f"  -> Restored {f} to root.")
                restored += 1
            except Exception as e:
                print(f"Error restoring {f}: {e}")
        else:
            print(f"  [INFO] File {f} already in root or not found in archive.")

    print(f"\nDone! Restored {restored} files back to the root.")
    print("=" * 60)

if __name__ == "__main__":
    main()
