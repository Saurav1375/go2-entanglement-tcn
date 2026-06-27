#!/usr/bin/env python3
"""
Plot thigh torque (tau) for all four legs from each normalized CSV file.
Each file produces one image with 4 subplots (one per leg), saved to plots/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

THIGH_TORQUE_COLS = ["FR_thigh_tau", "FL_thigh_tau", "RR_thigh_tau", "RL_thigh_tau"]
LEG_LABELS = {
    "FR_thigh_tau": "Front Right",
    "FL_thigh_tau": "Front Left",
    "RR_thigh_tau": "Rear Right",
    "RL_thigh_tau": "Rear Left",
}
COLORS = {
    "FR_thigh_tau": "#e74c3c",
    "FL_thigh_tau": "#2ecc71",
    "RR_thigh_tau": "#3498db",
    "RL_thigh_tau": "#f39c12",
}

INPUT_DIR = os.path.join(os.path.dirname(__file__), "csv_normalized")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")


def plot_file(csv_path, out_dir):
    filename = os.path.basename(csv_path)
    stem = os.path.splitext(filename)[0]

    df = pd.read_csv(csv_path)
    t = df["timestamp"].to_numpy()

    duration = float(t[-1])
    # Major tick every 0.5 s; minor tick every 0.1 s — denser for short recordings
    major_step = 0.25 if duration <= 6 else 0.5
    minor_step = 0.05 if duration <= 6 else 0.1

    fig, axes = plt.subplots(4, 1, figsize=(max(18, duration * 1.8), 12), sharex=True)
    fig.suptitle(f"Thigh Torque — {stem}", fontsize=14, fontweight="bold", y=0.98)

    for ax, col in zip(axes, THIGH_TORQUE_COLS):
        ax.plot(t, df[col].to_numpy(), color=COLORS[col], linewidth=0.8)
        ax.set_ylabel(f"{LEG_LABELS[col]}\n(Nm)", fontsize=9)
        ax.grid(True, which="major", linestyle="--", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.25)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(major_step))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
        ax.tick_params(axis="both", which="major", labelsize=7)
        ax.tick_params(axis="x", which="minor", length=3)
        ax.tick_params(labelbottom=True)  # show x labels on every subplot
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.set_xlabel("Time (s)", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved  {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = sorted(
        f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")
    )
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return

    for filename in csv_files:
        plot_file(os.path.join(INPUT_DIR, filename), OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
