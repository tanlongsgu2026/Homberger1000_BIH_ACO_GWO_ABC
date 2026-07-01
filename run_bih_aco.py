#!/usr/bin/main/env python
# -*- coding: utf-8 -*-
"""
Chương trình chạy thuật toán BIH-ACO cho bài toán VRPTW sử dụng nghiệm BIH đã tính toán sẵn từ file CSV.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time

# Import các thành phần cốt lõi từ hệ thống quản lý VRPTW của dự án
from vrptw_core import (
    Route,
    Solution,
    ensure_output_dirs,
    export_convergence_csv,
    export_results_csv,
    export_routes_csv,
    read_instances_from_zip,
    evaluate_solution,
    evaluate_route,
    clone_solution
)

# Import các hàm tính toán lõi của thuật toán ACO từ file algorithms.py
from algorithms import (
    ACOConfig, 
    _initialize_pheromone, 
    construct_ant_solution, 
    improve_solution, 
    update_pheromone, 
    compare_solutions
)

def load_bih_solution(instance, results_df, routes_df):
    """
    Đọc nghiệm BIH đã chạy sẵn từ dữ liệu CSV lịch sử hành trình và
    tái cấu trúc lại thành đối tượng Solution chuẩn hóa của hệ thống.
    """
    inst_name = instance.name
    
    # Kiểm tra xem dữ liệu tuyến đường (routes) của instance này có tồn tại không
    if inst_name not in routes_df['instance'].values:
        return None
        
    # Lọc và sắp xếp các tuyến đường của instance hiện tại theo thứ tự route_id
    inst_routes = routes_df[routes_df['instance'] == inst_name].sort_values('route_id')
    
    routes = []
    for _, row in inst_routes.iterrows():
        # Chuỗi khách hàng lưu trong cột customer_sequence dạng chuỗi cách nhau bởi khoảng trắng
        seq_str = str(row['customer_sequence']).strip()
        if not seq_str:
            continue
            
        # Chuyển đổi chuỗi ID khách hàng thành danh sách kiểu số nguyên
        customer_ids = [int(cid) for cid in seq_str.split()]
        
        # Đánh giá và tạo lại đối tượng Route hợp lệ dựa trên cấu trúc bài toán gốc
        route_obj = evaluate_route(instance, customer_ids)
        routes.append(route_obj)
        
    if not routes:
        return None
        
    # Tạo đối tượng Solution baseline từ danh sách các Route đã nạp
    sol = Solution(routes=routes, algorithm="BIH")
    return evaluate_solution(instance, sol)


def solve_bih_aco_with_cached_bih(instance, config, bih_sol):
    """
    Hàm tối ưu hóa chính của BIH-ACO:
    Thay vì tính lại BIH, hàm sử dụng trực tiếp nghiệm mồi `bih_sol` được truyền vào,
    khởi tạo ma trận Pheromone và tiến hành chạy giải thuật đàn kiến ACO.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)
    
    # Thiết lập thuật toán và seed, gán nghiệm BIH làm nghiệm tốt nhất ban đầu toàn cục
    bih_sol.algorithm = "BIH-ACO"
    bih_sol.seed = config.random_seed
    global_best = clone_solution(bih_sol)
    
    # BƯỚC QUAN TRỌNG: Khởi tạo vết pheromone trên các cạnh dựa trên cấu trúc nghiệm BIH có sẵn
    pheromone = _initialize_pheromone(instance, bih_sol)
    
    # Ghi nhận trạng thái hội tụ ban đầu tại vòng lặp thứ 0
    history = [(0, global_best.vehicles, global_best.total_distance, global_best.total_travel_time)]
    
    # Vòng lặp tối ưu hóa của thuật toán đàn kiến (Ant Colony Optimization)
    for it in range(1, config.max_iterations + 1):
        if time.perf_counter() - start >= config.runtime_limit:
            break
            
        iteration_best = None
        for _ in range(config.number_of_ants):
            if time.perf_counter() - start >= config.runtime_limit:
                break
                
            # Tạo lời giải mới từ kiến dựa trên ma trận xác suất và pheromone
            sol = construct_ant_solution(instance, pheromone, config, rng)
            
            # Áp dụng Local Search (Relocate, Swap, 2-opt) để tối ưu cục bộ lời giải của kiến
            sol = improve_solution(instance, sol, config.local_search_trials, rng)
            sol.algorithm = "BIH-ACO"
            sol.seed = config.random_seed
            
            # Cập nhật nghiệm tốt nhất trong vòng lặp hiện tại
            if compare_solutions(sol, iteration_best) < 0:
                iteration_best = sol
                
            # Cập nhật nghiệm tốt nhất toàn cục (Global Best) nếu tìm thấy phương án tốt hơn
            if compare_solutions(sol, global_best) < 0:
                global_best = clone_solution(sol)
                
        # Bay hơi và cập nhật lượng Pheromone mới dựa trên Iteration Best và Global Best
        update_pheromone(pheromone, iteration_best, global_best, instance, config)
        
        # Lưu lại tiến trình hội tụ của thuật toán
        history.append((it, global_best.vehicles, global_best.total_distance, global_best.total_travel_time))
        
    # Cập nhật thông tin tổng hợp cuối cùng của lời giải tối ưu tìm được
    global_best.algorithm = "BIH-ACO"
    global_best.seed = config.random_seed
    global_best.runtime_seconds = time.perf_counter() - start
    global_best.convergence_history = history
    return evaluate_solution(instance, global_best)


def main():
    parser = argparse.ArgumentParser(description="Run BIH-ACO using pre-calculated BIH results to skip re-computation")
    parser.add_argument("--zip-path", default="data/homberger_1000_customer_instances.zip")
    parser.add_argument("--bih-results", default="outputs/results/bih_results.csv", help="Đường dẫn đến file kết quả tổng quát BIH")
    parser.add_argument("--bih-routes", default="outputs/routes/bih_routes.csv", help="Đường dẫn đến file lưu hành trình chi tiết BIH")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--ants", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--runtime-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-search-trials", type=int, default=100)
    args = parser.parse_args()

    # Tạo các thư mục outputs cần thiết (results, routes, convergence) nếu chưa có
    ensure_output_dirs()
    
    # Xác định đường dẫn file kết quả BIH hợp lệ (Ưu tiên cấu hình truyền vào hoặc file ở thư mục gốc)
    bih_results_path = Path(args.bih_results)
    bih_routes_path = Path(args.bih_routes)
    
    if not bih_results_path.exists() and Path("bih_results.csv").exists():
        bih_results_path = Path("bih_results.csv")
    if not bih_routes_path.exists() and Path("bih_routes.csv").exists():
        bih_routes_path = Path("bih_routes.csv")
        
    print(f"[*] Loading BIH historical solutions from:\n    -> {bih_results_path}\n    -> {bih_routes_path}")
    results_df = pd.read_csv(bih_results_path)
    routes_df = pd.read_csv(bih_routes_path)

    # Đọc tập hợp bài toán VRPTW từ file zip cấu hình
    instances = read_instances_from_zip(args.zip_path, args.max_instances)
    config = ACOConfig(
        number_of_ants=args.ants, 
        max_iterations=args.iterations, 
        runtime_limit=args.runtime_limit, 
        random_seed=args.seed, 
        local_search_trials=args.local_search_trials, 
        mode="bih_aco"
    )
    
    results = []
    for k, inst in enumerate(instances, start=1):
        print(f"\n[{k}/{len(instances)}] BIH-ACO processing instance: {inst.name} ...")
        
        # Bước 1: Nạp trực tiếp nghiệm BIH đã có từ file CSV lưu trữ
        bih_sol = load_bih_solution(inst, results_df, routes_df)
        
        if bih_sol is None:
            print(f"  [Warning] Không tìm thấy nghiệm chạy sẵn của BIH cho {inst.name}. Bỏ qua bài toán này.")
            continue
            
        print(f"  -> BIH Baseline Loaded: vehicles={bih_sol.vehicles}, distance={bih_sol.total_distance:.2f}")
        
        # Bước 2: Kích hoạt thuật toán ACO giải dựa trên cấu trúc nghiệm mồi BIH vừa nạp
        sol = solve_bih_aco_with_cached_bih(inst, config, bih_sol)
        results.append(sol)
        print(f"  [Result] feasible={sol.feasible}, served={sol.served_customers}, vehicles={sol.vehicles}, distance={sol.total_distance:.2f}, time={sol.total_travel_time:.2f}, runtime={sol.runtime_seconds:.2f}s")

    # Xuất các file báo cáo kết quả đầu ra của thuật toán đề xuất BIH-ACO
    export_results_csv(results, Path(f"outputs/results/bih_aco_results_seed_{args.seed}.csv"))
    export_routes_csv(results, Path(f"outputs/routes/bih_aco_routes_seed_{args.seed}.csv"))
    export_convergence_csv(results, Path(f"outputs/convergence/bih_aco_convergence_seed_{args.seed}.csv"))
    print("\n[✔] Saved BIH-ACO optimization outputs successfully.")

if __name__ == "__main__":
    main()