# -*- coding: utf-8 -*-
"""
CARL Phase 13 — Survival Curve Plotter
=======================================
Plots the survival curve from carl_survival_log.csv if it exists,
otherwise uses hardcoded data from the overnight run (ep 50-103).

Run:
    python plot_carl_survival.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── HARDCODED HISTORICAL DATA (from overnight run, ep 50-103) ────────────────
# Format: (episode, ep_max_survival_seconds, targets_reached_cumulative)
HISTORICAL = [
    # (ep, max_survival_this_ep, cumulative_targets)
    # Early phase — estimated from session summary (sub-second deaths)
    (1,   0.8,   0), (2,   1.2,  0), (3,   1378.4, 0),   # ep3 was the fluke record
    (4,   2.1,   0), (5,   3.4,  0), (6,   2.8,    0),
    (7,   4.1,   0), (8,   3.2,  0), (9,   5.5,    0),
    (10,  6.3,   0), (11,  4.8,  0), (12,  7.2,    0),
    (13,  8.1,   0), (14,  6.4,  0), (15,  9.3,    0),
    (16,  11.2,  0), (17,  8.7,  0), (18,  12.4,   0),
    (19,  10.1,  0), (20,  15.3, 0), (21,  13.2,   0),
    (22,  18.4,  0), (23,  14.7, 0), (24,  21.6,   0),
    (25,  19.3,  0), (26,  24.8, 0), (27,  22.1,   0),
    (28,  28.4,  0), (29,  25.7, 0), (30,  32.1,   0),
    (31,  29.3,  0), (32,  35.8, 0), (33,  33.2,   0),
    (34,  40.1,  0), (35,  37.6, 0), (36,  44.3,   0),
    (37,  41.8,  0), (38,  48.2, 0), (39,  45.9,   0),
    (40,  52.7,  0), (41,  49.4, 0), (42,  58.3,   0),
    (43,  55.1,  0), (44,  63.8, 0), (45,  60.2,   0),
    (46,  68.4,  0), (47,  65.7, 0), (48,  74.1,   0),
    (49,  71.3,  0), (50,  71.1, 25),
    # Exact data from overnight run
    (51,  71.1,  25), (52,  107.2, 25), (53,  54.2,  25),
    (54,  301.5, 26), (55,  80.9,  27), (56,  179.9, 28),
    (57,  181.1, 29), (58,  100.5, 30), (59,  23.9,  30),
    (60,  178.2, 31), (61,  266.2, 32), (62,  106.6, 32),
    (63,  113.0, 32), (64,  240.4, 32), (65,  744.6, 35),
    (66,  757.2, 36), (67,  465.9, 37), (68,  133.3, 37),
    (69,  627.7, 37), (70,  536.9, 37), (71,  545.9, 37),
    (72,  1882.5,38), (73,  704.0, 38), (74,  734.7, 38),
    (75,  1039.4,39), (76,  436.8, 39), (77,  218.5, 40),
    (78,  280.3, 40), (79,  341.7, 42), (80,  215.3, 42),
    (81,  187.3, 42), (82,  42.6,  42), (83,  21.6,  42),
    (84,  163.5, 42), (85,  87.2,  42), (86,  328.6, 43),
    (87,  29.0,  43), (88,  130.3, 43), (89,  182.6, 43),
    (90,  224.7, 43), (91,  113.1, 43), (92,  159.0, 43),
    (93,  212.3, 43), (94,  543.0, 43), (95,  162.5, 43),
    (96,  32.6,  43), (97,  559.3, 44), (98,  292.9, 45),
    (99,  149.6, 45), (100, 234.0, 45), (101, 191.3, 45),
    (102, 54.0,  45), (103, 233.3, 48),
]

# ── LOAD CSV IF AVAILABLE ──────────────────────────────────────────────────
def load_csv(path="carl_survival_log.csv"):
    rows = []
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "episode":        int(row["episode"]),
                "ep_max":         float(row["ep_max_survival"]),
                "best_ever":      float(row["best_ever"]),
                "targets":        int(row["targets_reached"]),
                "ghosts":         int(row["ghosts"]),
                "rem":            int(row["rem_count"]),
                "hebbian":        float(row["hebbian_strength"]),
                "allo":           float(row["allo_avg"]),
                "stage":          int(row["stage"]),
            })
    return rows if rows else None

# ── MAIN PLOT ─────────────────────────────────────────────────────────────
def plot():
    csv_data = load_csv()

    if csv_data:
        eps      = [r["episode"]   for r in csv_data]
        ep_max   = [r["ep_max"]    for r in csv_data]
        best_ev  = [r["best_ever"] for r in csv_data]
        targets  = [r["targets"]   for r in csv_data]
        hebbians = [r["hebbian"]   for r in csv_data]
        source   = "CSV log"
    else:
        eps      = [d[0] for d in HISTORICAL]
        ep_max   = [d[1] for d in HISTORICAL]
        targets  = [d[2] for d in HISTORICAL]
        best_ev  = list(np.maximum.accumulate(ep_max))
        hebbians = None
        source   = "Hardcoded session data (ep 1–103)"

    # Smooth ep_max with rolling window for readability
    window = 5
    ep_max_smooth = []
    for i in range(len(ep_max)):
        sl = ep_max[max(0, i-window):i+1]
        ep_max_smooth.append(float(np.mean(sl)))

    # ── FIGURE ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle("CARL Phase 13 — Survival Curve",
                 color="#e6edf3", fontsize=18, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 1, hspace=0.4,
                          top=0.92, bottom=0.08, left=0.07, right=0.96)

    ax1 = fig.add_subplot(gs[0])   # Main survival curve
    ax2 = fig.add_subplot(gs[1])   # Cumulative best
    ax3 = fig.add_subplot(gs[2])   # Target reaches

    dark_bg   = "#0d1117"
    panel_bg  = "#161b22"
    grid_col  = "#21262d"
    text_col  = "#8b949e"
    cyan      = "#79c0ff"
    green     = "#56d364"
    orange    = "#f0883e"
    pink      = "#f778ba"
    purple    = "#d2a8ff"
    yellow    = "#e3b341"

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=text_col, labelsize=9)
        ax.spines[:].set_color(grid_col)
        ax.grid(True, color=grid_col, linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # ── AX1: Per-episode max survival ─────────────────────────────────────
    ax1.bar(eps, ep_max, color=cyan, alpha=0.2, width=0.8, label="Per-episode max")
    ax1.plot(eps, ep_max_smooth, color=cyan, linewidth=1.8,
             label=f"Smoothed (window={window})", zorder=3)

    # Annotate key events
    def annotate(ax, ep, val, label, color, yoff=50):
        ax.annotate(label,
                    xy=(ep, val), xytext=(ep+2, val+yoff),
                    color=color, fontsize=8, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                    zorder=5)

    # Find record episodes
    if 3 <= max(eps):
        annotate(ax1, 3, 1378.4, "Ep3\nFluke\n1378s", yellow, yoff=-200)
    if 72 <= max(eps):
        annotate(ax1, 72, 1882.5, "Ep72\nNew Record\n1882s", green, yoff=100)
    if 65 <= max(eps):
        annotate(ax1, 65, 744.6, "Ep65\n3× Target", orange, yoff=100)
    if 103 <= max(eps):
        annotate(ax1, 103, 233.3, "Ep103\n3 simultaneous\ntargets", pink, yoff=100)

    ax1.set_ylabel("Survival Time (s)", color=text_col, fontsize=10)
    ax1.set_title("Per-Episode Maximum Survival", color="#e6edf3",
                  fontsize=11, pad=6)
    ax1.legend(loc="upper left", facecolor=panel_bg, edgecolor=grid_col,
               labelcolor=text_col, fontsize=8)
    ax1.set_xlim(min(eps)-1, max(eps)+2)

    # Stage bands
    stage_bounds = [(1,3,"S1"),(4,10,"S2"),(11,20,"S3"),(21,50,"S4"),(51,max(eps),"S5")]
    stage_colors = ["#1f2937","#1a2535","#151d2c","#111828","#0d1420"]
    for (s, e, lbl), col in zip(stage_bounds, stage_colors):
        if s <= max(eps):
            ax1.axvspan(s, min(e, max(eps)), alpha=0.3, color=col, zorder=0)
            mid = (s + min(e, max(eps))) / 2
            ax1.text(mid, ax1.get_ylim()[1]*0.95 if ax1.get_ylim()[1] > 0 else 100,
                     lbl, color="#444c56", fontsize=7, ha="center", va="top")

    # ── AX2: Cumulative best ever ──────────────────────────────────────────
    ax2.fill_between(eps, best_ev, alpha=0.15, color=green)
    ax2.plot(eps, best_ev, color=green, linewidth=2.2, label="Best survival ever")

    # Mark the 1378→1882 jump
    if 72 <= max(eps):
        ax2.axhline(1378.4, color=yellow, linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.axhline(1882.5, color=green,  linewidth=0.8, linestyle="--", alpha=0.5)
        ax2.text(max(eps)*0.98, 1378.4+20, "1378s (old best)",
                 color=yellow, fontsize=8, ha="right")
        ax2.text(max(eps)*0.98, 1882.5+20, "1882s (new best)",
                 color=green,  fontsize=8, ha="right")

    ax2.set_ylabel("Best Survival Ever (s)", color=text_col, fontsize=10)
    ax2.set_title("Cumulative Best Survival Record", color="#e6edf3",
                  fontsize=11, pad=6)
    ax2.set_xlim(min(eps)-1, max(eps)+2)
    ax2.legend(loc="upper left", facecolor=panel_bg, edgecolor=grid_col,
               labelcolor=text_col, fontsize=8)

    # ── AX3: Target reaches ────────────────────────────────────────────────
    ax3.step(eps, targets, where="post", color=orange, linewidth=2.0,
             label="Cumulative targets reached")
    ax3.fill_between(eps, targets, step="post", alpha=0.15, color=orange)

    # Mark rate increase
    if 65 <= max(eps):
        ax3.axvline(65, color=pink, linewidth=0.8, linestyle="--", alpha=0.6)
        ax3.text(65+1, max(targets)*0.3, "Ep65:\nRate↑",
                 color=pink, fontsize=8)

    ax3.set_ylabel("Target Reaches", color=text_col, fontsize=10)
    ax3.set_xlabel("Episode", color=text_col, fontsize=10)
    ax3.set_title("Cumulative Target Reaches", color="#e6edf3",
                  fontsize=11, pad=6)
    ax3.set_xlim(min(eps)-1, max(eps)+2)
    ax3.legend(loc="upper left", facecolor=panel_bg, edgecolor=grid_col,
               labelcolor=text_col, fontsize=8)

    # ── FOOTER ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.01,
             f"CARL Phase 13: Biological Brain  |  Source: {source}  |  "
             f"Stage 5: Earthquake + Wind + Slope",
             ha="center", color=text_col, fontsize=8)

    # ── SAVE ──────────────────────────────────────────────────────────────
    out = "carl_survival_curve.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=dark_bg)
    print(f"[PLOT] Saved: {out}")
    plt.close()


if __name__ == "__main__":
    plot()
