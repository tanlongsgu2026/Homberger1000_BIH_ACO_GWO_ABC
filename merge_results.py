from pathlib import Path
import pandas as pd
import numpy as np

RESULT_DIR = Path("outputs/results")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in RESULT_DIR.glob("*.csv") if p.name not in {"all_algorithms_summary.csv", "group_average_results.csv", "best_results_by_instance.csv"}])
    if not files:
        print("No result CSV files found in outputs/results/.")
        return

    frames = []
    required = ["instance", "group", "algorithm", "seed", "feasible", "served_customers", "vehicles", "total_distance", "total_travel_time", "runtime_seconds"]
    for p in files:
        try:
            df = pd.read_csv(p)
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Warning: skip {p.name}; missing columns: {missing}")
                continue
            frames.append(df[required])
            print(f"Loaded {p.name}: {len(df)} rows")
        except Exception as e:
            print(f"Warning: cannot read {p}: {e}")

    if not frames:
        print("No valid result files found.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    # Normalize algorithm labels.
    all_df["algorithm"] = all_df["algorithm"].replace({"Standard ACO": "Standard ACO", "BIH-ACO": "BIH-ACO", "GWO": "GWO", "BIH": "BIH"})

    for col in ["vehicles", "total_distance", "total_travel_time", "runtime_seconds", "served_customers"]:
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    bih = all_df[all_df["algorithm"] == "BIH"][["instance", "vehicles", "total_distance", "total_travel_time"]].drop_duplicates("instance")
    bih = bih.rename(columns={"vehicles": "bih_vehicles", "total_distance": "bih_distance", "total_travel_time": "bih_time"})
    all_df = all_df.merge(bih, on="instance", how="left")

    all_df["improvement_vs_bih_vehicles"] = np.where(all_df["bih_vehicles"].notna() & (all_df["bih_vehicles"] != 0), (all_df["bih_vehicles"] - all_df["vehicles"]) / all_df["bih_vehicles"] * 100.0, np.nan)
    all_df["improvement_vs_bih_distance"] = np.where(all_df["bih_distance"].notna() & (all_df["bih_distance"] != 0), (all_df["bih_distance"] - all_df["total_distance"]) / all_df["bih_distance"] * 100.0, np.nan)
    all_df["improvement_vs_bih_time"] = np.where(all_df["bih_time"].notna() & (all_df["bih_time"] != 0), (all_df["bih_time"] - all_df["total_travel_time"]) / all_df["bih_time"] * 100.0, np.nan)
    all_df = all_df.drop(columns=["bih_vehicles", "bih_distance", "bih_time"])

    all_df.to_csv(RESULT_DIR / "all_algorithms_summary.csv", index=False)

    group_avg = all_df.groupby(["group", "algorithm"], dropna=False).agg(
        feasible_rate=("feasible", lambda x: pd.Series(x).astype(str).str.lower().isin(["true", "1"]).mean() * 100.0),
        avg_vehicles=("vehicles", "mean"),
        avg_distance=("total_distance", "mean"),
        avg_travel_time=("total_travel_time", "mean"),
        avg_runtime=("runtime_seconds", "mean"),
        std_vehicles=("vehicles", "std"),
        std_distance=("total_distance", "std"),
        std_travel_time=("total_travel_time", "std"),
        std_runtime=("runtime_seconds", "std"),
    ).reset_index()
    group_avg.to_csv(RESULT_DIR / "group_average_results.csv", index=False)

    sort_df = all_df.copy()
    sort_df["feasible_rank"] = np.where(sort_df["feasible"].astype(str).str.lower().isin(["true", "1"]), 0, 1)
    sort_df = sort_df.sort_values(["instance", "feasible_rank", "vehicles", "total_distance", "total_travel_time", "runtime_seconds"])
    best = sort_df.groupby("instance").head(1)[["instance", "algorithm", "vehicles", "total_distance", "total_travel_time", "runtime_seconds"]]
    best = best.rename(columns={"algorithm": "best_algorithm", "vehicles": "best_vehicles", "total_distance": "best_distance", "total_travel_time": "best_travel_time", "runtime_seconds": "best_runtime"})
    best.to_csv(RESULT_DIR / "best_results_by_instance.csv", index=False)

    print("\nSaved:")
    print("- outputs/results/all_algorithms_summary.csv")
    print("- outputs/results/group_average_results.csv")
    print("- outputs/results/best_results_by_instance.csv")


if __name__ == "__main__":
    main()
