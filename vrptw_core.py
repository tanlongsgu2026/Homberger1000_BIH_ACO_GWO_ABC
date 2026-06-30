"""
Core utilities for Homberger 1000 VRPTW experiments.
Shared by BIH, Standard ACO, GWO, and BIH-ACO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import csv
import math
import re
import time
import zipfile

import numpy as np


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass
class VRPTWInstance:
    name: str
    depot: Customer
    customers: List[Customer]
    vehicle_capacity: float
    distance_matrix: np.ndarray
    travel_time_matrix: np.ndarray
    id_to_index: Dict[int, int]
    index_to_id: List[int]
    max_vehicles: Optional[int] = None

    @property
    def all_nodes(self) -> List[Customer]:
        return [self.depot] + self.customers

    @property
    def customer_ids(self) -> List[int]:
        return [c.id for c in self.customers]


@dataclass
class Route:
    customer_ids: List[int] = field(default_factory=list)
    load: float = 0.0
    distance: float = 0.0
    travel_time: float = 0.0
    feasible: bool = True


@dataclass
class Solution:
    routes: List[Route] = field(default_factory=list)
    vehicles: int = 0
    total_distance: float = 0.0
    total_travel_time: float = 0.0
    feasible: bool = False
    served_customers: int = 0
    runtime_seconds: float = 0.0
    algorithm: str = ""
    seed: Optional[int] = None
    instance: str = ""
    group: str = ""
    convergence_history: List[Tuple[int, int, float, float]] = field(default_factory=list)


def ensure_output_dirs(base: str | Path = ".") -> None:
    base = Path(base)
    for p in [
        base / "outputs" / "results",
        base / "outputs" / "routes",
        base / "outputs" / "charts",
        base / "outputs" / "convergence",
        base / "outputs" / "logs",
    ]:
        p.mkdir(parents=True, exist_ok=True)


def extract_group_from_instance_name(instance_name: str) -> str:
    stem = Path(instance_name).stem.upper()
    for g in ["RC1", "RC2", "C1", "C2", "R1", "R2"]:
        if stem.startswith(g) or f"/{g}" in stem or f"_{g}" in stem:
            return g
    m = re.search(r"(RC1|RC2|C1|C2|R1|R2)", stem)
    return m.group(1) if m else "UNKNOWN"


def _numbers_from_line(line: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]


def read_instance_from_text(instance_name: str, text: str) -> VRPTWInstance:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    capacity = None
    max_vehicles = None

    # Homberger/Solomon usually has a line with two integers below VEHICLE: NUMBER CAPACITY.
    for i, ln in enumerate(lines):
        if "CAPACITY" in ln.upper() and i + 1 < len(lines):
            nums = _numbers_from_line(lines[i + 1])
            if len(nums) >= 2:
                max_vehicles = int(nums[0])
                capacity = float(nums[1])
                break

    # Fallback: choose first 2-number line before customer table.
    if capacity is None:
        for ln in lines[:30]:
            nums = _numbers_from_line(ln)
            if len(nums) == 2 and nums[0] > 0 and nums[1] > 0:
                max_vehicles = int(nums[0])
                capacity = float(nums[1])
                break

    if capacity is None:
        raise ValueError(f"Cannot find vehicle capacity in {instance_name}")

    rows: List[List[float]] = []
    for ln in lines:
        nums = _numbers_from_line(ln)
        # Customer rows: id x y demand ready due service = at least 7 numbers.
        if len(nums) >= 7:
            # Avoid vehicle-number/capacity lines and headers with accidental numbers.
            cid = int(nums[0])
            row = nums[:7]
            if cid >= 0:
                rows.append(row)

    if not rows:
        raise ValueError(f"Cannot find customer rows in {instance_name}")

    # Keep unique customer ids in first occurrence order. Homberger depot id is usually 0.
    seen = set()
    unique_rows = []
    for r in rows:
        cid = int(r[0])
        if cid not in seen:
            unique_rows.append(r)
            seen.add(cid)

    depot_row = None
    customer_rows = []
    for r in unique_rows:
        if int(r[0]) == 0 and depot_row is None:
            depot_row = r
        elif int(r[0]) != 0:
            customer_rows.append(r)
    if depot_row is None:
        depot_row = unique_rows[0]
        customer_rows = unique_rows[1:]

    depot = Customer(int(depot_row[0]), depot_row[1], depot_row[2], depot_row[3], depot_row[4], depot_row[5], depot_row[6])
    customers = [Customer(int(r[0]), r[1], r[2], r[3], r[4], r[5], r[6]) for r in customer_rows]
    all_nodes = [depot] + customers
    dist = compute_distance_matrix(all_nodes)
    id_to_index = {c.id: idx for idx, c in enumerate(all_nodes)}
    index_to_id = [c.id for c in all_nodes]
    return VRPTWInstance(
        name=Path(instance_name).stem,
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity,
        distance_matrix=dist,
        travel_time_matrix=dist.copy(),
        id_to_index=id_to_index,
        index_to_id=index_to_id,
        max_vehicles=max_vehicles,
    )


def read_instances_from_zip(zip_path: str | Path, max_instances: Optional[int] = None) -> List[VRPTWInstance]:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Data file not found: {zip_path}")
    instances: List[VRPTWInstance] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted([n for n in zf.namelist() if n.lower().endswith(".txt") and not n.endswith("/")])
        if max_instances is not None:
            names = names[:max_instances]
        if not names:
            raise ValueError(f"No .txt instance files found in {zip_path}")
        for name in names:
            print(f"Reading instance: {name}")
            raw = zf.read(name)
            text = raw.decode("utf-8", errors="ignore")
            instances.append(read_instance_from_text(name, text))
    return instances


def compute_distance_matrix(customers: Sequence[Customer]) -> np.ndarray:
    coords = np.array([(c.x, c.y) for c in customers], dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _customer_by_id(instance: VRPTWInstance, customer_id: int) -> Customer:
    return instance.all_nodes[instance.id_to_index[customer_id]]


def evaluate_route(instance: VRPTWInstance, customer_sequence: Sequence[int]) -> Route:
    load = 0.0
    distance = 0.0
    current_time = max(0.0, instance.depot.ready_time)
    feasible = True
    prev = instance.depot.id

    for cid in customer_sequence:
        c = _customer_by_id(instance, cid)
        i = instance.id_to_index[prev]
        j = instance.id_to_index[cid]
        travel = float(instance.travel_time_matrix[i, j])
        dist = float(instance.distance_matrix[i, j])
        arrival = current_time + travel
        service_start = max(arrival, c.ready_time)
        if service_start > c.due_date + 1e-9:
            feasible = False
        load += c.demand
        if load > instance.vehicle_capacity + 1e-9:
            feasible = False
        distance += dist
        current_time = service_start + c.service_time
        prev = cid

    # Return to depot.
    i = instance.id_to_index[prev]
    j = instance.id_to_index[instance.depot.id]
    distance += float(instance.distance_matrix[i, j])
    current_time += float(instance.travel_time_matrix[i, j])
    if current_time > instance.depot.due_date + 1e-9:
        feasible = False

    return Route(list(customer_sequence), load, distance, current_time, feasible)


def evaluate_solution(instance: VRPTWInstance, solution: Solution) -> Solution:
    evaluated_routes = [evaluate_route(instance, r.customer_ids) for r in solution.routes if len(r.customer_ids) > 0]
    solution.routes = evaluated_routes
    solution.vehicles = len(evaluated_routes)
    solution.total_distance = float(sum(r.distance for r in evaluated_routes))
    solution.total_travel_time = float(sum(r.travel_time for r in evaluated_routes))
    solution.served_customers = sum(len(r.customer_ids) for r in evaluated_routes)
    solution.feasible = all(r.feasible for r in evaluated_routes) and check_all_customers_served(instance, solution)
    solution.instance = instance.name
    solution.group = extract_group_from_instance_name(instance.name)
    return solution


def check_all_customers_served(instance: VRPTWInstance, solution: Solution) -> bool:
    expected = set(instance.customer_ids)
    served = []
    for r in solution.routes:
        served.extend(r.customer_ids)
    return len(served) == len(expected) and set(served) == expected and len(served) == len(set(served))


def is_feasible_solution(instance: VRPTWInstance, solution: Solution) -> bool:
    return evaluate_solution(instance, solution).feasible


def solution_cost_tuple(solution: Solution) -> Tuple[int, float, float, float, int]:
    feasible_rank = 0 if solution.feasible else 1
    served_penalty = -solution.served_customers
    return (feasible_rank, solution.vehicles, solution.total_distance, solution.total_travel_time, solution.runtime_seconds, served_penalty)


def compare_solutions(sol_a: Optional[Solution], sol_b: Optional[Solution]) -> int:
    """Return -1 if sol_a is better, 1 if sol_b is better, 0 if tie."""
    if sol_a is None and sol_b is None:
        return 0
    if sol_a is None:
        return 1
    if sol_b is None:
        return -1
    a = solution_cost_tuple(sol_a)
    b = solution_cost_tuple(sol_b)
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def clone_solution(sol: Solution) -> Solution:
    return Solution(
        routes=[Route(list(r.customer_ids), r.load, r.distance, r.travel_time, r.feasible) for r in sol.routes],
        vehicles=sol.vehicles,
        total_distance=sol.total_distance,
        total_travel_time=sol.total_travel_time,
        feasible=sol.feasible,
        served_customers=sol.served_customers,
        runtime_seconds=sol.runtime_seconds,
        algorithm=sol.algorithm,
        seed=sol.seed,
        instance=sol.instance,
        group=sol.group,
        convergence_history=list(sol.convergence_history),
    )


def compute_improvement_vs_bih(bih_solution: Solution, algo_solution: Solution) -> Dict[str, Optional[float]]:
    def imp(base: float, val: float) -> Optional[float]:
        if base is None or abs(base) < 1e-12:
            return None
        return ((base - val) / base) * 100.0
    return {
        "improvement_vs_bih_vehicles": imp(float(bih_solution.vehicles), float(algo_solution.vehicles)),
        "improvement_vs_bih_distance": imp(float(bih_solution.total_distance), float(algo_solution.total_distance)),
        "improvement_vs_bih_time": imp(float(bih_solution.total_travel_time), float(algo_solution.total_travel_time)),
    }


def result_row(sol: Solution) -> Dict[str, object]:
    return {
        "instance": sol.instance,
        "group": sol.group,
        "algorithm": sol.algorithm,
        "seed": "" if sol.seed is None else sol.seed,
        "feasible": sol.feasible,
        "served_customers": sol.served_customers,
        "vehicles": sol.vehicles,
        "total_distance": round(sol.total_distance, 6),
        "total_travel_time": round(sol.total_travel_time, 6),
        "runtime_seconds": round(sol.runtime_seconds, 6),
    }


def export_results_csv(results: Sequence[Solution], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance", "group", "algorithm", "seed", "feasible", "served_customers", "vehicles", "total_distance", "total_travel_time", "runtime_seconds"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sol in results:
            writer.writerow(result_row(sol))


def export_routes_csv(results: Sequence[Solution], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance", "group", "algorithm", "seed", "route_id", "customer_sequence", "load", "distance", "travel_time", "feasible"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sol in results:
            for rid, r in enumerate(sol.routes, start=1):
                writer.writerow({
                    "instance": sol.instance,
                    "group": sol.group,
                    "algorithm": sol.algorithm,
                    "seed": "" if sol.seed is None else sol.seed,
                    "route_id": rid,
                    "customer_sequence": " ".join(map(str, r.customer_ids)),
                    "load": round(r.load, 6),
                    "distance": round(r.distance, 6),
                    "travel_time": round(r.travel_time, 6),
                    "feasible": r.feasible,
                })


def export_convergence_csv(results: Sequence[Solution], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance", "group", "algorithm", "seed", "iteration", "best_vehicles", "best_distance", "best_travel_time"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sol in results:
            for it, veh, dist, tt in sol.convergence_history:
                writer.writerow({
                    "instance": sol.instance,
                    "group": sol.group,
                    "algorithm": sol.algorithm,
                    "seed": "" if sol.seed is None else sol.seed,
                    "iteration": it,
                    "best_vehicles": veh,
                    "best_distance": round(dist, 6),
                    "best_travel_time": round(tt, 6),
                })


def timed() -> float:
    return time.perf_counter()
