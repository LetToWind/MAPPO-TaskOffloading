import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_training_curve(csv_path, title=None, window=200):
    rows = list(csv.reader(open(csv_path)))
    header_comment = rows[0][0] if rows[0][0].startswith("#") else ""
    data_rows = rows[1:] if header_comment else rows
    data_rows = [r for r in data_rows if r and r[0] != "episode"]
    rewards = [float(r[1]) for r in data_rows]
    successes = [int(r[2]) for r in data_rows]
    fails = [int(r[3]) for r in data_rows]
    n = len(rewards)

    reward_ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    success_ma = np.convolve(successes, np.ones(window) / window, mode="valid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(n), rewards, alpha=0.2, color="steelblue", linewidth=0.5)
    ax1.plot(range(window - 1, n), reward_ma, color="steelblue", linewidth=2, label="MA(%d)" % window)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.set_title("Episode Reward")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(range(n), successes, alpha=0.2, color="seagreen", linewidth=0.5)
    ax2.plot(range(window - 1, n), success_ma, color="seagreen", linewidth=2, label="MA(%d)" % window)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success")
    ax2.set_title("Success per Episode")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    if header_comment:
        fig.text(0.5, 0.01, header_comment.replace("# ", ""), ha="center", fontsize=8, color="gray")

    png_path = Path(csv_path).with_suffix(".png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print("Plot saved to", png_path)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        plot_training_curve(path)
