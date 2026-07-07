import subprocess
import os
import shutil
import time

# List of experiments to run
# Format: (name, procedural_inheritance, ecological_persistence, goal_crystallization, culture_imitation)
experiments = [
    ("control", "True", "True", "True", "True"),
    ("no_goals", "True", "True", "False", "True"),
    ("no_inheritance", "False", "True", "True", "True"),
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
    # Ensure memory dir exists
    os.makedirs("memory", exist_ok=True)
    
    # We first backup the existing brain so we start each experiment with the same brain model
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
        
        # Reset files
        clean_file("evolutionary_fossil_record.csv")
        clean_file("emergence_metrics.log")
        clean_file("metabolic_ecology.log")
        
        # Reset the brain to the backup version so they all start with identical priors
        if brain_backup_exists:
            shutil.copyfile("memory/carl_brain_experiment_backup.npz", "memory/carl_brain.npz")
            print("Reset brain model to core baseline prior.")
            
        # Command line arguments
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
        # Run process and wait for completion
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        # Stream output in real-time
        for line in iter(process.stdout.readline, ''):
            # Print only key logs to keep output clean
            if "[DEATH]" in line or "[BREEDING]" in line or "[EXPERIMENT]" in line or "[CORTEX] Emergence Status" in line:
                print(line.strip())
        
        process.wait()
        dt = time.time() - t0
        print(f"Experiment finished in {dt:.1f} seconds. Status code: {process.returncode}")
        
        # Backup the generated CSV files
        if os.path.exists("evolutionary_fossil_record.csv"):
            shutil.copyfile("evolutionary_fossil_record.csv", f"evolutionary_fossil_{name}.csv")
            print(f"Saved: evolutionary_fossil_{name}.csv")
        if os.path.exists("emergence_metrics.log"):
            shutil.copyfile("emergence_metrics.log", f"emergence_metrics_{name}.csv")
            print(f"Saved: emergence_metrics_{name}.csv")
        if os.path.exists("metabolic_ecology.log"):
            shutil.copyfile("metabolic_ecology.log", f"metabolic_ecology_{name}.csv")
            print(f"Saved: metabolic_ecology_{name}.csv")
            
    # Restore final backup
    if brain_backup_exists:
        shutil.copyfile("memory/carl_brain_experiment_backup.npz", "memory/carl_brain.npz")
        try:
            os.remove("memory/carl_brain_experiment_backup.npz")
        except Exception:
            pass
        print("\nRestored original carl_brain.npz file.")
        
    print("\n============================================================")
    print("  ALL EXPERIMENTS COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
