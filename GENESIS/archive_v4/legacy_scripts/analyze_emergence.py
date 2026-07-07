"""
CARL Emergence Analysis — Baseline vs Ablation vs Mark IX Subconsciousness
Produces comparative charts and statistics across all available experimental data.
"""
import csv
import os
import math
import statistics

# ─── Configuration ───────────────────────────────────────────────────
BASE_DIR = r"D:\carl_simulation\GENESIS"
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Data Loaders ────────────────────────────────────────────────────
def load_csv(filepath):
    """Load a CSV file into a list of dicts with numeric conversion."""
    if not os.path.exists(filepath):
        return []
    rows = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
    return rows

# ─── Load All Datasets ──────────────────────────────────────────────
print("=" * 70)
print("  CARL EMERGENCE ANALYSIS")
print("  Baseline vs Ablation vs Mark IX Subconsciousness")
print("=" * 70)

# Fossil records (per-generation summary)
fossil_control = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_control.csv"))
fossil_no_persist = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_no_persistence.csv"))
fossil_no_culture = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_no_culture.csv"))
fossil_no_goals = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_no_goals.csv"))
fossil_no_inherit = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_no_inheritance.csv"))
fossil_markix = load_csv(os.path.join(BASE_DIR, "evolutionary_fossil_record.csv"))

# Emergence metrics (per-100-step snapshots)
emergence_control = load_csv(os.path.join(BASE_DIR, "emergence_metrics_control.csv"))
emergence_no_persist = load_csv(os.path.join(BASE_DIR, "emergence_metrics_no_persistence.csv"))
emergence_markix_log = load_csv(os.path.join(BASE_DIR, "emergence_metrics.log"))

# Metabolic ecology
metabolic_control = load_csv(os.path.join(BASE_DIR, "metabolic_ecology_control.csv"))
metabolic_no_persist = load_csv(os.path.join(BASE_DIR, "metabolic_ecology_no_persistence.csv"))

def safe_stats(values):
    """Return (mean, stdev, min, max) with safety for empty lists."""
    if not values:
        return (0, 0, 0, 0)
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0
    return (m, s, min(values), max(values))

def trend_arrow(values):
    """Return an ASCII trend indicator based on first vs last third."""
    if len(values) < 6:
        return "~"
    third = len(values) // 3
    early = statistics.mean(values[:third])
    late = statistics.mean(values[-third:])
    pct = (late - early) / max(abs(early), 0.001) * 100
    if pct > 15:
        return f"↑ +{pct:.0f}%"
    elif pct < -15:
        return f"↓ {pct:.0f}%"
    else:
        return f"→ {pct:+.0f}%"

# ─── Section 1: Survival Analysis ───────────────────────────────────
print("\n" + "─" * 70)
print("  SECTION 1: SURVIVAL ANALYSIS (Lifespan per Generation)")
print("─" * 70)

datasets = {
    "Control (Full Mark VIII)": fossil_control,
    "NO_PERSISTENCE (Ablation)": fossil_no_persist,
    "NO_CULTURE (Ablation)": fossil_no_culture,
    "NO_GOALS (Ablation)": fossil_no_goals,
    "NO_INHERITANCE (Ablation)": fossil_no_inherit,
    "Mark IX (Subconsciousness)": fossil_markix,
}

print(f"\n{'Experiment':<32} {'Gens':>5} {'Mean Life':>10} {'StdDev':>8} {'Min':>8} {'Max':>8} {'Trend':>12}")
print("-" * 92)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>5}")
        continue
    lifespans = [r['lifespan'] for r in data if 'lifespan' in r]
    mean, std, mn, mx = safe_stats(lifespans)
    trend = trend_arrow(lifespans)
    print(f"{name:<32} {len(lifespans):>5} {mean:>10.0f} {std:>8.0f} {mn:>8.0f} {mx:>8.0f} {trend:>12}")

# ─── Section 2: Food Harvesting Efficiency ──────────────────────────
print("\n" + "─" * 70)
print("  SECTION 2: FOOD HARVESTING EFFICIENCY")
print("─" * 70)

print(f"\n{'Experiment':<32} {'Total Food':>10} {'Food/Gen':>10} {'Food/1k Steps':>14} {'Trend':>12}")
print("-" * 85)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>10}")
        continue
    foods = [r.get('food_harvested', 0) for r in data]
    lifespans = [r.get('lifespan', 1) for r in data]
    total_food = sum(foods)
    food_per_gen = statistics.mean(foods) if foods else 0
    total_steps = sum(lifespans)
    food_per_1k = (total_food / max(total_steps, 1)) * 1000
    trend = trend_arrow(foods)
    print(f"{name:<32} {total_food:>10.0f} {food_per_gen:>10.1f} {food_per_1k:>14.2f} {trend:>12}")

# ─── Section 3: Damage & Cause of Death ─────────────────────────────
print("\n" + "─" * 70)
print("  SECTION 3: DAMAGE ANALYSIS & CAUSE OF DEATH")
print("─" * 70)

print(f"\n{'Experiment':<32} {'Mean Dmg':>10} {'Starvation%':>12} {'Collision%':>12} {'Trend':>12}")
print("-" * 85)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>10}")
        continue
    damages = [r.get('damage_accumulated', 0) for r in data]
    mean_dmg = statistics.mean(damages) if damages else 0
    # Estimate cause: damage > 0.8 → collision death, else starvation
    collision_deaths = sum(1 for d in damages if d > 0.8)
    starvation_deaths = len(damages) - collision_deaths
    coll_pct = collision_deaths / max(len(damages), 1) * 100
    starv_pct = starvation_deaths / max(len(damages), 1) * 100
    trend = trend_arrow(damages)
    print(f"{name:<32} {mean_dmg:>10.3f} {starv_pct:>11.0f}% {coll_pct:>11.0f}% {trend:>12}")

# ─── Section 4: Elite Fitness Score ─────────────────────────────────
print("\n" + "─" * 70)
print("  SECTION 4: ELITE FITNESS SCORE (Evolutionary Pressure)")
print("─" * 70)

print(f"\n{'Experiment':<32} {'Mean Elite':>12} {'Peak Elite':>12} {'Final Elite':>12} {'Trend':>12}")
print("-" * 85)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>12}")
        continue
    elites = [r.get('elite_score', 0) for r in data]
    mean_e = statistics.mean(elites) if elites else 0
    peak_e = max(elites) if elites else 0
    final_e = elites[-1] if elites else 0
    trend = trend_arrow(elites)
    print(f"{name:<32} {mean_e:>12.0f} {peak_e:>12.0f} {final_e:>12.0f} {trend:>12}")

# ─── Section 5: Exploration Diversity ────────────────────────────────
print("\n" + "─" * 70)
print("  SECTION 5: EXPLORATION DIVERSITY (Map Coverage %)")
print("─" * 70)

print(f"\n{'Experiment':<32} {'Mean Div':>10} {'Peak Div':>10} {'Final Div':>10} {'Trend':>12}")
print("-" * 85)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>10}")
        continue
    divs = [r.get('exploration_diversity', 0) for r in data]
    mean_d = statistics.mean(divs) if divs else 0
    peak_d = max(divs) if divs else 0
    final_d = divs[-1] if divs else 0
    trend = trend_arrow(divs)
    print(f"{name:<32} {mean_d:>10.1f} {peak_d:>10.1f} {final_d:>10.1f} {trend:>12}")

# ─── Section 6: Goal Crystallization ────────────────────────────────
print("\n" + "─" * 70)
print("  SECTION 6: GOAL CRYSTALLIZATION (Intrinsic Motivation)")
print("─" * 70)

print(f"\n{'Experiment':<32} {'Total Goals':>12} {'Goals/Gen':>10} {'Trend':>12}")
print("-" * 55)

for name, data in datasets.items():
    if not data:
        print(f"{name:<32} {'N/A':>12}")
        continue
    goals = [r.get('goal_count', 0) for r in data]
    total_g = sum(goals)
    mean_g = statistics.mean(goals) if goals else 0
    trend = trend_arrow(goals)
    print(f"{name:<32} {total_g:>12.0f} {mean_g:>10.1f} {trend:>12}")

# ─── Section 7: Mark IX Subconsciousness Metrics ────────────────────
print("\n" + "─" * 70)
print("  SECTION 7: MARK IX SUBCONSCIOUSNESS METRICS")
print("  (Only available for current live run)")
print("─" * 70)

if emergence_markix_log:
    # Workspace mode distribution
    modes = [r.get('workspace_mode', 'unknown') for r in emergence_markix_log if 'workspace_mode' in r]
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    total_modes = len(modes)
    
    print(f"\n  Global Workspace Mode Distribution ({total_modes} samples):")
    print(f"  {'Mode':<20} {'Count':>8} {'Frequency':>10}")
    print(f"  " + "-" * 42)
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / max(total_modes, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"  {str(mode):<20} {count:>8} {pct:>9.1f}% {bar}")
    
    # Workspace entropy (cognitive flexibility)
    entropies = [r.get('workspace_entropy', 0) for r in emergence_markix_log if 'workspace_entropy' in r]
    if entropies:
        mean_ent, std_ent, min_ent, max_ent = safe_stats(entropies)
        trend_ent = trend_arrow(entropies)
        print(f"\n  Workspace Entropy (Cognitive Flexibility):")
        print(f"    Mean: {mean_ent:.3f}  StdDev: {std_ent:.3f}  Range: [{min_ent:.3f}, {max_ent:.3f}]  {trend_ent}")
    
    # Somatic markers
    markers = [r.get('somatic_markers', 0) for r in emergence_markix_log if 'somatic_markers' in r]
    biases = [r.get('somatic_bias', 0) for r in emergence_markix_log if 'somatic_bias' in r]
    if markers:
        mean_m, _, min_m, max_m = safe_stats(markers)
        mean_b, _, min_b, max_b = safe_stats(biases)
        print(f"\n  Somatic Markers (Gut Feelings):")
        print(f"    Active markers: Mean={mean_m:.1f}  Max={max_m:.0f}")
        print(f"    Emotional bias: Mean={mean_b:.3f}  Range=[{min_b:.3f}, {max_b:.3f}]")
        print(f"    Trend: {trend_arrow(markers)}")
    
    # Habit chunks
    habits = [r.get('habit_chunks', 0) for r in emergence_markix_log if 'habit_chunks' in r]
    if habits:
        mean_h, _, min_h, max_h = safe_stats(habits)
        print(f"\n  Basal Ganglia Habits (Motor Programs):")
        print(f"    Active chunks: Mean={mean_h:.1f}  Max={max_h:.0f}")
        print(f"    Trend: {trend_arrow(habits)}")
    
    # DMN activation
    dmn = [r.get('dmn_activation', 0) for r in emergence_markix_log if 'dmn_activation' in r]
    if dmn:
        mean_dmn, _, min_dmn, max_dmn = safe_stats(dmn)
        active_pct = sum(1 for d in dmn if d > 0.1) / max(len(dmn), 1) * 100
        print(f"\n  Default Mode Network (Daydreaming):")
        print(f"    Mean activation: {mean_dmn:.3f}  Max: {max_dmn:.3f}")
        print(f"    Active (>0.1): {active_pct:.1f}% of time")
    
    # Allostatic urgency
    allo = [r.get('allostatic_urgency', 0) for r in emergence_markix_log if 'allostatic_urgency' in r]
    if allo:
        mean_a, _, min_a, max_a = safe_stats(allo)
        urgent_pct = sum(1 for a in allo if a > 0.5) / max(len(allo), 1) * 100
        print(f"\n  Allostatic Predictor (Future Energy Anticipation):")
        print(f"    Mean urgency: {mean_a:.3f}  Max: {max_a:.3f}")
        print(f"    High urgency (>0.5): {urgent_pct:.1f}% of time")
    
    # Circadian phase
    circadian = [r.get('circadian_phase', 0) for r in emergence_markix_log if 'circadian_phase' in r]
    if circadian:
        # Count rest vs active phases
        rest_count = sum(1 for c in circadian if math.cos(c * 2 * math.pi) < -0.5)
        rest_pct = rest_count / max(len(circadian), 1) * 100
        print(f"\n  Circadian Oscillator (Activity/Rest Rhythms):")
        print(f"    Rest phase: {rest_pct:.1f}% of time")
        print(f"    Active phase: {100-rest_pct:.1f}% of time")
        # Check if period is visible
        if len(circadian) > 100:
            phase_diffs = [circadian[i+1] - circadian[i] for i in range(min(100, len(circadian)-1))]
            avg_diff = statistics.mean([abs(d) for d in phase_diffs])
            est_period = 1.0 / max(avg_diff, 0.0001)
            print(f"    Estimated period: ~{est_period:.0f} steps")

else:
    print("\n  [No Mark IX emergence data found yet. Run carl_harvest.py longer.]")

# ─── Section 8: Generation-over-Generation Comparison Table ──────────
print("\n" + "─" * 70)
print("  SECTION 8: GENERATION-BY-GENERATION COMPARISON")
print("─" * 70)

# Find the max number of generations across all experiments
max_gens = max(
    len(fossil_control), len(fossil_no_persist), len(fossil_markix),
    len(fossil_no_goals), len(fossil_no_inherit)
)

print(f"\n{'Gen':>4} │ {'Control':>10} │ {'NoPersist':>10} │ {'NoGoals':>10} │ {'NoInherit':>10} │ {'Mark IX':>10}")
print("─" * 4 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10 + "─┼─" + "─" * 10)

for i in range(min(max_gens, 25)):
    ctrl = f"{fossil_control[i]['lifespan']:.0f}" if i < len(fossil_control) else "—"
    nop = f"{fossil_no_persist[i]['lifespan']:.0f}" if i < len(fossil_no_persist) else "—"
    nog = f"{fossil_no_goals[i]['lifespan']:.0f}" if i < len(fossil_no_goals) else "—"
    noi = f"{fossil_no_inherit[i]['lifespan']:.0f}" if i < len(fossil_no_inherit) else "—"
    mix = f"{fossil_markix[i]['lifespan']:.0f}" if i < len(fossil_markix) else "—"
    print(f"{i:>4} │ {ctrl:>10} │ {nop:>10} │ {nog:>10} │ {noi:>10} │ {mix:>10}")

# ─── Section 9: Summary & Key Findings ──────────────────────────────
print("\n" + "═" * 70)
print("  SUMMARY & KEY FINDINGS")
print("═" * 70)

# Calculate comparative stats
ctrl_lifespans = [r['lifespan'] for r in fossil_control] if fossil_control else [0]
nop_lifespans = [r['lifespan'] for r in fossil_no_persist] if fossil_no_persist else [0]
mix_lifespans = [r['lifespan'] for r in fossil_markix] if fossil_markix else [0]

ctrl_mean = statistics.mean(ctrl_lifespans)
nop_mean = statistics.mean(nop_lifespans)
mix_mean = statistics.mean(mix_lifespans)

print(f"\n  Mean Lifespan Comparison:")
print(f"    Control (Mark VIII):         {ctrl_mean:>10.0f} steps")
print(f"    NO_PERSISTENCE (Ablation):   {nop_mean:>10.0f} steps  ({((nop_mean/max(ctrl_mean,1))-1)*100:+.1f}% vs control)")
print(f"    Mark IX (Subconsciousness):  {mix_mean:>10.0f} steps  ({((mix_mean/max(ctrl_mean,1))-1)*100:+.1f}% vs control)")

# Food efficiency
ctrl_food = [r.get('food_harvested', 0) for r in fossil_control]
nop_food = [r.get('food_harvested', 0) for r in fossil_no_persist]
mix_food = [r.get('food_harvested', 0) for r in fossil_markix]

ctrl_fmean = statistics.mean(ctrl_food) if ctrl_food else 0
nop_fmean = statistics.mean(nop_food) if nop_food else 0
mix_fmean = statistics.mean(mix_food) if mix_food else 0

print(f"\n  Mean Food/Generation:")
print(f"    Control:         {ctrl_fmean:>6.1f}")
print(f"    NO_PERSISTENCE:  {nop_fmean:>6.1f}")
print(f"    Mark IX:         {mix_fmean:>6.1f}")

# Exploration
ctrl_div = [r.get('exploration_diversity', 0) for r in fossil_control]
nop_div = [r.get('exploration_diversity', 0) for r in fossil_no_persist]
mix_div = [r.get('exploration_diversity', 0) for r in fossil_markix]

ctrl_dmean = statistics.mean(ctrl_div) if ctrl_div else 0
nop_dmean = statistics.mean(nop_div) if nop_div else 0
mix_dmean = statistics.mean(mix_div) if mix_div else 0

print(f"\n  Mean Map Coverage (%):")
print(f"    Control:         {ctrl_dmean:>6.1f}%")
print(f"    NO_PERSISTENCE:  {nop_dmean:>6.1f}%")
print(f"    Mark IX:         {mix_dmean:>6.1f}%")

if emergence_markix_log:
    print(f"\n  Subconsciousness Emergence Signatures:")
    markers_final = [r.get('somatic_markers', 0) for r in emergence_markix_log]
    habits_final = [r.get('habit_chunks', 0) for r in emergence_markix_log]
    if markers_final:
        print(f"    Peak somatic markers: {max(markers_final):.0f}")
    if habits_final:
        print(f"    Peak habit chunks:    {max(habits_final):.0f}")
    if modes:
        dominant = max(mode_counts, key=mode_counts.get)
        print(f"    Dominant mode:        {dominant} ({mode_counts[dominant]/total_modes*100:.0f}%)")

print(f"\n" + "═" * 70)
print("  Analysis complete. Data from {0} experiments analyzed.".format(
    sum(1 for d in datasets.values() if d)))
print("═" * 70)
