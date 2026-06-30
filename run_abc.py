import argparse
from pathlib import Path
from vrptw_core import ensure_output_dirs, export_convergence_csv, export_results_csv, export_routes_csv, read_instances_from_zip
from algorithms import ABCConfig, solve_abc


def main():
    parser = argparse.ArgumentParser(description="Run ABC on Homberger 1000 VRPTW instances")
    parser.add_argument("--zip-path", default="data/homberger_1000_customer_instances.zip")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--bees", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--runtime-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--local-search-trials", type=int, default=100)
    args = parser.parse_args()

    ensure_output_dirs()
    instances = read_instances_from_zip(args.zip_path, args.max_instances)
    config = ABCConfig(colony_size=args.bees, max_iterations=args.iterations, runtime_limit=args.runtime_limit, random_seed=args.seed, limit=args.limit, local_search_trials=args.local_search_trials)
    results = []
    for k, inst in enumerate(instances, start=1):
        print(f"\n[{k}/{len(instances)}] ABC solving {inst.name} ...")
        sol = solve_abc(inst, config)
        results.append(sol)
        print(f"  feasible={sol.feasible}, served={sol.served_customers}, vehicles={sol.vehicles}, distance={sol.total_distance:.2f}, time={sol.total_travel_time:.2f}, runtime={sol.runtime_seconds:.2f}s")

    export_results_csv(results, Path(f"outputs/results/abc_results_seed_{args.seed}.csv"))
    export_routes_csv(results, Path(f"outputs/routes/abc_routes_seed_{args.seed}.csv"))
    export_convergence_csv(results, Path(f"outputs/convergence/abc_convergence_seed_{args.seed}.csv"))
    print("\nSaved ABC outputs.")


if __name__ == "__main__":
    main()
