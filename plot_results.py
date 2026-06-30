from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = Path("outputs/results")
CHART_DIR = Path("outputs/charts")
CONV_DIR = Path("outputs/convergence")


def save_bar(df, x, y, title, ylabel, filename):
    ax = df.plot(kind="bar", x=x, y=y, legend=False)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=300)
    plt.close()


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT_DIR / "all_algorithms_summary.csv"
    group_path = RESULT_DIR / "group_average_results.csv"
    if not summary_path.exists():
        print("Missing outputs/results/all_algorithms_summary.csv. Run merge_results.py first.")
        return
    df = pd.read_csv(summary_path)

    algo_avg = df.groupby("algorithm", as_index=False).agg(
        avg_vehicles=("vehicles", "mean"),
        avg_distance=("total_distance", "mean"),
        avg_travel_time=("total_travel_time", "mean"),
        avg_runtime=("runtime_seconds", "mean"),
        avg_improvement_distance=("improvement_vs_bih_distance", "mean"),
    )
    save_bar(algo_avg, "algorithm", "avg_vehicles", "Average Vehicles by Algorithm", "Average vehicles", "avg_vehicles_by_algorithm.png")
    save_bar(algo_avg, "algorithm", "avg_distance", "Average Total Distance by Algorithm", "Average total distance", "avg_distance_by_algorithm.png")
    save_bar(algo_avg, "algorithm", "avg_travel_time", "Average Travel Time by Algorithm", "Average travel time", "avg_travel_time_by_algorithm.png")
    save_bar(algo_avg, "algorithm", "avg_runtime", "Average Runtime by Algorithm", "Average runtime (s)", "avg_runtime_by_algorithm.png")
    save_bar(algo_avg, "algorithm", "avg_improvement_distance", "Average Distance Improvement over BIH", "Improvement (%)", "avg_improvement_vs_bih_distance.png")

    if group_path.exists():
        gdf = pd.read_csv(group_path)
        pivot_v = gdf.pivot(index="group", columns="algorithm", values="avg_vehicles")
        ax = pivot_v.plot(kind="bar")
        ax.set_title("Group-Level Average Vehicles")
        ax.set_xlabel("Group")
        ax.set_ylabel("Average vehicles")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "group_average_vehicles.png", dpi=300)
        plt.close()

        pivot_d = gdf.pivot(index="group", columns="algorithm", values="avg_distance")
        ax = pivot_d.plot(kind="bar")
        ax.set_title("Group-Level Average Distance")
        ax.set_xlabel("Group")
        ax.set_ylabel("Average distance")
        plt.tight_layout()
        plt.savefig(CHART_DIR / "group_average_distance.png", dpi=300)
        plt.close()

    conv_files = sorted(CONV_DIR.glob("*_convergence_seed_*.csv"))
    for p in conv_files:
        cdf = pd.read_csv(p)
        if cdf.empty:
            continue
        for (instance, algorithm, seed), sub in cdf.groupby(["instance", "algorithm", "seed"]):
            ax = sub.plot(x="iteration", y="best_distance", legend=False)
            ax.set_title(f"Convergence Curve - {algorithm} - {instance} - seed {seed}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best distance")
            plt.tight_layout()
            safe = f"convergence_{algorithm}_{instance}_seed_{seed}.png".replace(" ", "_").replace("/", "_")
            plt.savefig(CHART_DIR / safe, dpi=300)
            plt.close()

    print(f"Charts saved to {CHART_DIR}")


if __name__ == "__main__":
    main()
