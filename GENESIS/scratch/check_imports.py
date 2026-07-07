"""
check_imports.py — Checks if any files in the root directory try to import
any python files that were moved to archive_v4/.
"""

import os
import re

ROOT_DIR = "D:/carl_simulation/GENESIS"
ARCHIVE_SCRIPTS_DIR = os.path.join(ROOT_DIR, "archive_v4/legacy_scripts")

# Get list of scripts that were archived (without .py extension)
archived_names = []
if os.path.exists(ARCHIVE_SCRIPTS_DIR):
    for f in os.listdir(ARCHIVE_SCRIPTS_DIR):
        if f.endswith(".py"):
            archived_names.append(f[:-3])

# Add old arm demos too
ARCHIVE_ARMS_DIR = os.path.join(ROOT_DIR, "archive_v4/old_arm_demos")
if os.path.exists(ARCHIVE_ARMS_DIR):
    for f in os.listdir(ARCHIVE_ARMS_DIR):
        if f.endswith(".py"):
            archived_names.append(f[:-3])

print("Archived scripts list:", archived_names)
print("\nScanning root files for imports of archived files...")

# Scan root python files
root_files = [f for f in os.listdir(ROOT_DIR) if f.endswith(".py")]

dependencies = {}
for rf in root_files:
    path = os.path.join(ROOT_DIR, rf)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find any imports
    found_imports = []
    for name in archived_names:
        # Match "import name", "from name import", etc.
        pattern = r"\b(import|from)\s+" + re.escape(name) + r"\b"
        if re.search(pattern, content):
            found_imports.append(name)
            
    if found_imports:
        dependencies[rf] = found_imports

if dependencies:
    print("\n[WARN] Dependencies found:")
    for rf, deps in dependencies.items():
        print(f"  {rf} imports: {deps}")
else:
    print("\n[OK] No dependencies found! All root files are self-contained.")
