import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')
titles={"e2e-latency": "End-To-End-Latency", "proc-latency": "Processing Latency", "throughput": "Throughput"}


def plot():
    species = ("Seen Queries", "Unseen hardware", "Unseen structures", "Unseen benchmark")
    means = {
           'COSTREAM': (1.34, 1.93,  1.52,   1.45),
        'Flat Vector': (9.92, 15.63, 5.52, 104.75),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(7, 1.5))
    ax.set_yscale("log")

    for attribute, measurement in means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width,  edgecolor="black", label=attribute)
        ax.bar_label(rects, padding=0)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Median\nQ-Error')
    ax.set_xticks(x + width, species)
    ax.axvline(0.65, color="black")
    #ax.text(0.35, 21, "Seen Queries", fontsize=10,  verticalalignment='top')
    #ax.text(2.35, 21, "Unseen Queries", fontsize=10,  verticalalignment='top')
    ax.set_ylim(0, 105)

    bars = ax.patches
    patterns = ('///', '..', 'xxx')
    hatches = [p for p in patterns for i in range(4)]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.legend(loc='upper center', ncols=2, fontsize=8, bbox_to_anchor=(0.64, 1.02))
    fig.tight_layout()
    #plt.show()
    plt.savefig("motivating_plot.pdf")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    plot()
