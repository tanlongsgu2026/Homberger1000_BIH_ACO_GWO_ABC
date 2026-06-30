import argparse
from pathlib import Path
from vrptw_core import ensure_output_dirs, export_results_csv, export_routes_csv, read_instances_from_zip
from algorithms import solve_bih_multi_order


def main():
    parser = argparse.ArgumentParser(description="Run BIH on Homberger 1000 VRPTW instances")
    parser.add_argument("--zip-path", default="data/homberger_1000_customer_instances.zip")
    parser.add_argument("--max-instances", type=int, default=None)
    args = parser.parse_args()

    ensure_output_dirs()
    instances = read_instances_from_zip(args.zip_path, args.max_instances)
    results = []
    for k, inst in enumerate(instances, start=1):
        print(f"\n[{k}/{len(instances)}] BIH solving {inst.name} ...")
        sol = solve_bih_multi_order(inst)
        results.append(sol)
        print(f"  feasible={sol.feasible}, served={sol.served_customers}, vehicles={sol.vehicles}, distance={sol.total_distance:.2f}, time={sol.total_travel_time:.2f}, runtime={sol.runtime_seconds:.2f}s")

    export_results_csv(results, Path("outputs/results/bih_results.csv"))
    export_routes_csv(results, Path("outputs/routes/bih_routes.csv"))
    print("\nSaved BIH results to outputs/results/bih_results.csv")


if __name__ == "__main__":
    main()
