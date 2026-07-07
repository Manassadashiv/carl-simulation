"""
Run only the two incomplete experiment conditions:
  - no_persistence (had 6/25 generations)
  - no_culture (had 6/25 generations)

These conditions were interrupted by a server restart.
We re-run them from scratch (25 generations each) to get complete data.
"""
import subprocess
import os
import shutil
import time

# Only the incomplete experiments
experiments = [
    ("no_persistence", "True", "False", "True", "True"),
    ("no_culture", "True", "True", "True", "False")
]

MAX_GEN = 25

def clean_file(path):
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"Cleared existing file: {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")

def main():
    print("============================================================")
    print("  RESTARTING INCOMPLETE EXPERIMENTS")
    print("============================================================")
    
    os.makedirs("memory", exist_ok=True)
    
    brain_backup_exists = os.path.exists("memory/carl_brain.npz")
    if brain_backup_exists:
        shutil.copyfile("memory/carl_brain.npz", "memory/carl_brain_experiment_backup.npz")
        print("Backed up core brain model for experiment reset.")
    
    for name, proc, eco, goal, cult in experiments:
        print(f"\n[RUNNING EXPERIMENT: {name.upper()}]")
        print(f"Procedural Inheritance:  {proc}")
        print(f"Ecological Persistence:  {eco}")
        print(f"Goal Crystallization:    {goal}")
        print(f"Culture/Imitation:       {cult}")
        print("------------------------------------------------------------")
        
        # Reset per-run files
        clean_file("evolutionary_fossil_record.csv")
        clean_file("emergence_metrics.log")
        clean_file("metabolic_ecology.log")
        
        # Reset brain to baseline
        if brain_backup_exists:
            shutil.copyfile("memory/carl_brain_experiment_backup.npz", "memory/carl_brain.npz")
            print("Reset brain model to core baseline prior.")
            
        cmd = [
            "python", "-u", "carl_harvest.py",
            "--no-render", "--no-keyboard",
            "--procedural-inheritance", proc,
            "--ecological-persistence", eco,
            "--goal-crystallization", goal,
            "--culture-imitation", cult,
            "--max-generations", str(MAX_GEN)
        ]
        
        t0 = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        for line in iter(process.stdout.readline, ''):
            if "[DEATH]" in line or "[BREEDING]" in line or "[EXPERIMENT]" in line or "[CORTEX] Emergence Status" in line:
                print(line.strip())
        
        process.wait()
        dt = time.time() - t0
        print(f"Experiment finished in {dt:.1f} seconds. Status code: {process.returncode}")
        
        # Save results
        if os.path.exists("evolutionary_fossil_record.csv"):
            shutil.copyfile("evolutionary_fossil_record.csv", f"evolutionary_fossil_{name}.csv")
            print(f"Saved: evolutionary_fossil_{name}.csv")
        if os.path.exists("emergence_metrics.log"):
            shutil.copyfile("emergence_metrics.log", f"emergence_metrics_{name}.csv")
            print(f"Saved: emergence_metrics_{name}.csv")
        if os.path.exists("metabolic_ecology.log"):
            shutil.copyfile("metabolic_ecology.log", f"metabolic_ecology_{name}.csv")
            print(f"Saved: metabolic_ecology_{name}.csv")
            
    # Restore backup
    if brain_backup_exists:
        shutil.copyfile("memory/carl_brain_experiment_backup.npz", "memory/carl_brain.npz")
        try:
            os.remove("memory/carl_brain_experiment_backup.npz")
        except Exception:
            pass
        print("\nRestored original carl_brain.npz file.")
        
    print("\n============================================================")
    print("  REMAINING EXPERIMENTS COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
