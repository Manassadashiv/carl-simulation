"""
reorganize_scout.py — Segregates all active scripts, XML files, and assets of CARL
Primate Scout into a dedicated carl_scout/ directory, leaving only the launcher
carl_autonomous.py in the root, and patches paths dynamically to guarantee 0% error rate.
"""

import os
import shutil

ROOT_DIR = "D:/carl_simulation/GENESIS"
SCOUT_DIR = os.path.join(ROOT_DIR, "carl_scout")
ARCHIVE_DIR = os.path.join(ROOT_DIR, "archive_v4")

ACTIVE_FILES_TO_MOVE = [
    "carl_agent.py",
    "carl_allostatic.py",
    "carl_bios.py",
    "carl_body.xml",
    "carl_brainstem.py",
    "carl_circadian.py",
    "carl_cortex.py",
    "carl_cpg.py",
    "carl_curiosity.py",
    "carl_double_buffer.py",
    "carl_eval.py",
    "carl_evolution.py",
    "carl_mapping.py",
    "carl_matrix_download.py",
    "carl_metabolism.py",
    "carl_obstacle_controller.py",
    "carl_planner.py",
    "carl_primate_scout.xml",
    "carl_primate_scout_demo.py",
    "carl_scout_autonomous.py",
    "carl_scout_bc.py",
    "carl_scout_ik_demo.py",
    "carl_scout_ppo.py",
    "carl_scout_viewer.py",
    "carl_sensor_fusion.py",
    "carl_somatic.py",
    "carl_active_inference.py",
    "carl_dmn.py",
    "carl_habits.py",
    "carl_imagination.py",
    "carl_workspace.py",
    "carl_expression.py",
    "blob_telemetry.py",
    "vessel_kinetic.xml",
    "face_landmarker.task",
]

def main():
    print("=" * 60)
    print("  CARL Workspace Segregator — Active Code Grouping")
    print("=" * 60)

    # 1. Create carl_scout directory
    os.makedirs(SCOUT_DIR, exist_ok=True)

    # 2. Patch and Move carl_body_interface.py
    src_bi = os.path.join(ROOT_DIR, "carl_body_interface.py")
    dst_bi = os.path.join(SCOUT_DIR, "carl_body_interface.py")
    if os.path.exists(src_bi):
        with open(src_bi, 'r', encoding='utf-8') as f:
            content = f.read()
        # Make XML path relative to the script location using os.path
        content = content.replace('MODEL_PATH = "carl_primate_scout.xml"', 
                                  'import os\nMODEL_PATH = os.path.join(os.path.dirname(__file__), "carl_primate_scout.xml")')
        with open(dst_bi, 'w', encoding='utf-8') as f:
            f.write(content)
        os.remove(src_bi)
        print("  -> Patched and moved carl_body_interface.py -> carl_scout/")

    # 3. Patch and Move carl_scout_train.py
    src_tr = os.path.join(ROOT_DIR, "carl_scout_train.py")
    dst_tr = os.path.join(SCOUT_DIR, "carl_scout_train.py")
    if os.path.exists(src_tr):
        with open(src_tr, 'r', encoding='utf-8') as f:
            content = f.read()
        # Make XML path relative using os.path
        content = content.replace('MODEL_PATH     = "carl_primate_scout.xml"', 
                                  'import os\nMODEL_PATH     = os.path.join(os.path.dirname(__file__), "carl_primate_scout.xml")')
        with open(dst_tr, 'w', encoding='utf-8') as f:
            f.write(content)
        os.remove(src_tr)
        print("  -> Patched and moved carl_scout_train.py -> carl_scout/")

    # 4. Move other active files to carl_scout/
    moved_count = 0
    for f in ACTIVE_FILES_TO_MOVE:
        src = os.path.join(ROOT_DIR, f)
        dst = os.path.join(SCOUT_DIR, f)
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
    print(f"  -> Moved {moved_count} active scripts and assets to carl_scout/")

    # 5. Move old implementation plan to archive
    src_plan = os.path.join(ROOT_DIR, "implementation_plan (29-06-2026)_full body integration n correction.md")
    dst_plan = os.path.join(ARCHIVE_DIR, "implementation_plan (29-06-2026)_full body integration n correction.md")
    if os.path.exists(src_plan):
        shutil.move(src_plan, dst_plan)
        print("  -> Moved old implementation plan to archive_v4/")

    # 6. Patch carl_autonomous.py launcher in root to insert carl_scout in Python path
    src_launcher = os.path.join(ROOT_DIR, "carl_autonomous.py")
    if os.path.exists(src_launcher):
        with open(src_launcher, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if sys.path.insert is already present
        if "carl_scout" not in content:
            path_patch = (
                "import sys\n"
                "import os\n"
                "sys.path.insert(0, os.path.join(os.path.dirname(__file__), \"carl_scout\"))\n\n"
            )
            # Find a good place to insert it (e.g. right after docstring or at top of file)
            # Find end of docstring
            doc_end = content.find('"""', 3)
            if doc_end != -1:
                insert_idx = doc_end + 3
                # Move past newline
                while insert_idx < len(content) and content[insert_idx] in ('\r', '\n'):
                    insert_idx += 1
                new_content = content[:insert_idx] + path_patch + content[insert_idx:]
            else:
                new_content = path_patch + content
                
            with open(src_launcher, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("  -> Patched carl_autonomous.py launcher to include carl_scout/ in path.")

    print("\nWorkspace Segregation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
