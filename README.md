# Homberger1000_BIH_ACO_GWO_ABC

Bộ code này dùng để thực nghiệm bài toán **Vehicle Routing Problem with Time Windows (VRPTW)** trên bộ dữ liệu **Homberger 1000**. Project được thiết kế để so sánh thuật toán đề xuất **BIH-ACO** với các thuật toán đối chứng, gồm BIH, Standard ACO, GWO và ABC.

## 1. Các thuật toán

Project so sánh 5 thuật toán:

1. **BIH** — Best Insertion Heuristic, thuật toán chèn tốt nhất, dùng làm baseline.
2. **Standard ACO** — Standard Ant Colony Optimization, thuật toán tối ưu đàn kiến tiêu chuẩn, không khởi tạo bằng BIH.
3. **GWO** — Grey Wolf Optimizer, thuật toán tối ưu sói xám, dùng random-key representation.
4. **ABC** — Artificial Bee Colony, thuật toán đàn ong nhân tạo, dùng random-key representation.
5. **BIH-ACO** — thuật toán đề xuất, ACO được khởi tạo bằng nghiệm BIH.

Lưu ý: Kết quả chỉ nên gọi là **feasible solution**, **baseline solution**, **best-found solution**, hoặc **improved solution**. Không gọi là optimal solution nếu chưa chứng minh tối ưu.

## 2. Cấu trúc project

```text
Homberger1000_BIH_ACO_GWO_ABC/
│
├── data/
│   └── homberger_1000_customer_instances.zip
│
├── outputs/
│   ├── results/
│   ├── routes/
│   ├── charts/
│   └── convergence/
│
├── vrptw_core.py
├── algorithms.py
│
├── run_bih.py
├── run_standard_aco.py
├── run_gwo.py
├── run_abc.py
├── run_bih_aco.py
├── merge_results.py
├── plot_results.py
├── requirements.txt
└── README.md
```

## 3. Cài đặt môi trường

Mở CMD hoặc PowerShell tại thư mục project, sau đó chạy:

```bash
pip install -r requirements.txt
```

Project dùng các thư viện:

```text
numpy
pandas
matplotlib
```

## 4. Chạy thử 2 instance

### BIH

```bash
python run_bih.py --max-instances 2
```

### Standard ACO

```bash
python run_standard_aco.py --max-instances 2 --runtime-limit 30 --seed 2026
```

### GWO

```bash
python run_gwo.py --max-instances 2 --runtime-limit 30 --seed 2026
```

### ABC

```bash
python run_abc.py --max-instances 2 --runtime-limit 30 --seed 2026
```

### BIH-ACO

```bash
python run_bih_aco.py --max-instances 2 --runtime-limit 30 --seed 2026
```

## 5. Chạy toàn bộ 60 instance

Sau khi chạy thử ổn định, bỏ tham số `--max-instances`:

```bash
python run_bih.py
python run_standard_aco.py --runtime-limit 90 --seed 2026
python run_gwo.py --runtime-limit 90 --seed 2026
python run_abc.py --runtime-limit 90 --seed 2026
python run_bih_aco.py --runtime-limit 90 --seed 2026
```

Nếu muốn chạy nhiều seed:

```bash
python run_standard_aco.py --runtime-limit 90 --seed 2027
python run_gwo.py --runtime-limit 90 --seed 2027
python run_abc.py --runtime-limit 90 --seed 2027
python run_bih_aco.py --runtime-limit 90 --seed 2027
```

## 6. Gộp kết quả và vẽ biểu đồ

Sau khi có đủ CSV trong `outputs/results/`, chạy:

```bash
python merge_results.py
python plot_results.py
```

Kết quả gộp:

```text
outputs/results/all_algorithms_summary.csv
outputs/results/group_average_results.csv
outputs/results/best_results_by_instance.csv
```

Biểu đồ được lưu tại:

```text
outputs/charts/
```

## 7. Kết quả từng thuật toán

BIH:

```text
outputs/results/bih_results.csv
outputs/routes/bih_routes.csv
```

Standard ACO:

```text
outputs/results/standard_aco_results_seed_2026.csv
outputs/routes/standard_aco_routes_seed_2026.csv
outputs/convergence/standard_aco_convergence_seed_2026.csv
```

GWO:

```text
outputs/results/gwo_results_seed_2026.csv
outputs/routes/gwo_routes_seed_2026.csv
outputs/convergence/gwo_convergence_seed_2026.csv
```

ABC:

```text
outputs/results/abc_results_seed_2026.csv
outputs/routes/abc_routes_seed_2026.csv
outputs/convergence/abc_convergence_seed_2026.csv
```

BIH-ACO:

```text
outputs/results/bih_aco_results_seed_2026.csv
outputs/routes/bih_aco_routes_seed_2026.csv
outputs/convergence/bih_aco_convergence_seed_2026.csv
```

## 8. Định dạng CSV kết quả

Tất cả file kết quả riêng của từng thuật toán đều có các cột:

```text
instance, group, algorithm, seed, feasible, served_customers, vehicles, total_distance, total_travel_time, runtime_seconds
```

File route có các cột:

```text
instance, group, algorithm, seed, route_id, customer_sequence, load, distance, travel_time, feasible
```

File convergence có các cột:

```text
instance, group, algorithm, seed, iteration, best_vehicles, best_distance, best_travel_time
```

## 9. Gợi ý viết trong bài báo

Có thể viết:

> The proposed BIH-ACO algorithm is compared with four baseline methods, including BIH, Standard ACO, GWO, and ABC.

Tiếng Việt:

> Thuật toán BIH-ACO đề xuất được so sánh với bốn phương pháp đối chứng gồm BIH, ACO tiêu chuẩn, GWO và ABC.

Trong bài báo, nên nhấn mạnh rằng tất cả thuật toán dùng chung:

- bộ đọc dữ liệu,
- bộ kiểm tra ràng buộc VRPTW,
- hàm đánh giá nghiệm,
- quy tắc so sánh nghiệm,
- định dạng CSV đầu ra.
