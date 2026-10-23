import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set publication-quality plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


# PART A
dense_csv = "dense_optimization_results.csv"

if os.path.exists(dense_csv):
    df_dense = pd.read_csv(dense_csv)

    # Compute median and IQR per method and matrix size n
    dense_summary = df_dense.groupby(["method", "n"])["time_seconds"].agg(
        median="median",
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75)
    ).reset_index()
    dense_summary["iqr"] = dense_summary["q75"] - dense_summary["q25"]

    print("SUMMARY STATISTICS: DENSE OPTIMISATIONS (TIME IN SECONDS)")
    print(dense_summary[["method", "n", "median", "iqr"]].to_string(index=False))
    dense_summary.to_csv("summary_dense_optimisation.csv", index=False)

    # Plot 1: Dense Kernel Execution Time vs. Matrix Dimension n (Log-Log)
    plt.figure(figsize=(8, 5.5))
    method_styles = {
        "Naive_IJK": {"color": "#d62728", "marker": "o", "label": "Naive (i-j-k Baseline)"},
        "Reordered_IKJ": {"color": "#2ca02c", "marker": "s", "label": "Loop Reordered (i-k-j)"},
        "Blocked_Tiled": {"color": "#1f77b4", "marker": "^", "label": "Cache Blocked (Tiled 32x32)"}
    }

    for method in df_dense["method"].unique():
        data = dense_summary[dense_summary["method"] == method]
        style = method_styles.get(method, {"color": "black", "marker": "x", "label": method})
        plt.errorbar(
            data["n"],
            data["median"],
            yerr=data["iqr"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            capsize=4,
            linewidth=1.8,
            markersize=6
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Matrix Dimension ($n$)", fontweight="bold")
    plt.ylabel("Execution Time (seconds, log scale)", fontweight="bold")
    plt.title("Dense Multiplication: Baseline vs. Loop Reordering vs. Tiling", fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig("dense_optimization_comparison.png", dpi=300)
    plt.close()
    print("--> Exported: 'dense_optimization_comparison.png'")
else:
    print(f"Warning: '{dense_csv}' not found.")



# PART B
sparse_csv = "sparse_results.csv"

if os.path.exists(sparse_csv):
    df_sparse = pd.read_csv(sparse_csv)

    # Compute median runtime and median memory per method and sparsity
    sparse_summary = df_sparse.groupby(["method", "sparsity"]).agg(
        time_median=("time_seconds", "median"),
        time_iqr=("time_seconds", lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
        mem_median=("memory_mb", "median"),
        nnz_out_median=("nnz_out", "median")
    ).reset_index()

    sparse_summary["sparsity_percent"] = sparse_summary["sparsity"] * 100.0

    print("\n=== SUMMARY STATISTICS: SPARSE VS DENSE (n = 500) ===")
    print(sparse_summary[["method", "sparsity_percent", "time_median", "mem_median", "nnz_out_median"]].to_string(index=False))
    sparse_summary.to_csv("summary_sparse_evaluation.csv", index=False)

    # Plot 2: Dual Plot - Time and Memory vs. Sparsity Level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # --- Subplot 1: Execution Time vs Sparsity ---
    for method, col, mark in [("Dense_IKJ", "#d62728", "o"), ("Sparse_CSR", "#1f77b4", "s")]:
        sub = sparse_summary[sparse_summary["method"] == method]
        ax1.plot(
            sub["sparsity_percent"],
            sub["time_median"],
            label=f"{method}",
            color=col,
            marker=mark,
            linewidth=2,
            markersize=6
        )

    ax1.set_xlabel("Sparsity Level (% of Zeros)", fontweight="bold")
    ax1.set_ylabel("Execution Time (seconds)", fontweight="bold")
    ax1.set_title("Runtime vs. Sparsity Level ($n=500$)", fontweight="bold")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(frameon=True)

    # Add crossover highlight around 98%
    ax1.axvline(x=98.0, color="gray", linestyle=":", label="Crossover threshold (~98%)")

    # --- Subplot 2: Memory Footprint vs Sparsity ---
    for method, col, mark in [("Dense_IKJ", "#d62728", "o"), ("Sparse_CSR", "#1f77b4", "s")]:
        sub = sparse_summary[sparse_summary["method"] == method]
        ax2.plot(
            sub["sparsity_percent"],
            sub["mem_median"],
            label=f"{method}",
            color=col,
            marker=mark,
            linewidth=2,
            markersize=6
        )

    ax2.set_xlabel("Sparsity Level (% of Zeros)", fontweight="bold")
    ax2.set_ylabel("Memory Footprint (MB)", fontweight="bold")
    ax2.set_title("Memory Consumption vs. Sparsity Level ($n=500$)", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(frameon=True)

    plt.tight_layout()
    plt.savefig("sparse_vs_dense_analysis.png", dpi=300)
    plt.close()
    print("--> Exported: 'sparse_vs_dense_analysis.png'")
else:
    print(f"Warning: '{sparse_csv}' not found.")

print("\nAll Assignment 2 analysis plots generated successfully.")