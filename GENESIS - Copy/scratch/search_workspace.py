import os

workspace_dir = "d:/carl_simulation/GENESIS"
queries = ["ArmPolicy", "ARM_CTRL_", "carl_arm_weights", "carl_arm_train"]

for root, dirs, files in os.walk(workspace_dir):
    # Skip directories we don't want to search
    if any(p in root for p in ["venv", "__pycache__", ".git", "legacy"]):
        continue
    for file in files:
        if not file.endswith(".py"):
            continue
        filepath = os.path.join(root, file)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for q in queries:
                if q in content:
                    print(f"Match found for '{q}' in {filepath}")
        except Exception as e:
            pass
