import csv
import os
import time
import multiprocessing
import json 
import itertools
from typing import List, Tuple, Any, Dict, Union, Callable

# 类型定义
# Scheme 结构: (data_transmission, compute_cycles, resource_consumption, mapping_design)
Scheme = Tuple[float, float, float, Any] 
ResultRow = List[Any] 

# ==============================================================================
# DLBP 辅助函数
# ==============================================================================

def _calculate_min_resource_suffix_sum(valid_schemes_per_layer: List[List[Scheme]]) -> List[float]:
    """
    计算从当前层到最后一层所有层中，最小资源消耗的总和（后缀和）。
    min_resource_suffix_sum[i] 存储的是 i 层到最后一层 (Layer N-1) 的最小资源总和。
    用于 DLBP 剪枝。
    """
    total_layers = len(valid_schemes_per_layer)
    # 数组长度为 total_layers + 1，索引 total_layers 处为 0
    min_resource_suffix_sum = [0.0] * (total_layers + 1)
    
    current_min_sum = 0.0
    # 从后往前计算，索引 total_layers 处为 0
    for i in range(total_layers - 1, -1, -1):
        # 找到当前层 i 的最小资源消耗
        min_layer_resource = min(scheme[2] for scheme in valid_schemes_per_layer[i])
        current_min_sum += min_layer_resource
        min_resource_suffix_sum[i] = current_min_sum
        
    return min_resource_suffix_sum

# ==============================================================================
# 核心工作函数 (Worker Functions)
# 这些函数在子进程中运行，负责执行实际的搜索逻辑。
# ==============================================================================

def _dlbp_search_worker(worker_input: Tuple) -> List[ResultRow]:
    """
    DLBP (动态下限剪枝) 搜索的工作进程。
    使用 Dynamic Lower Bound Pruning (资源) 和 Upper Bound Pruning (Cycles, Transmission)。
    """
    (
        start_scheme,
        valid_schemes_per_layer,
        min_resource_suffix_sum, # DLBP 独有的参数
        resource_limit,
        resource_min,
        cycle_limit,
        transmission_limit,
        max_rows_per_process 
    ) = worker_input
    
    total_layers = len(valid_schemes_per_layer)
    found_rows = [] 
    local_count = 0

    def search_scheme(
        layer_index: int, 
        current_combo: List[Scheme],
        current_transmission: float,
        current_cycles: float,
        current_resource: float
    ) -> None:
        nonlocal local_count
        
        # 1. 上限剪枝 (Upper Bound Pruning)
        if (current_resource > resource_limit or
            current_cycles > cycle_limit or
            current_transmission > transmission_limit):
            return 

        # 2. 递归终止条件
        if layer_index == total_layers:
            # 最终检查 resource_min
            if current_resource >= resource_min:
                if local_count >= max_rows_per_process:
                    return 

                local_count += 1
                row = [0] # Scheme ID 占位符
                for scheme in current_combo:
                    row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                row.extend([current_transmission, current_cycles, current_resource])
                found_rows.append(row)
            return

        # 3. 动态下限剪枝 (DLBP)
        # min_resource_suffix_sum[layer_index] 存储的是从当前层开始到最后一层的最小总和
        # min_resource_suffix_sum[layer_index+1] 存储的是从下一层开始到最后一层的最小总和
        remaining_min_resource = min_resource_suffix_sum[layer_index]
        
        # 核心优化剪枝：如果当前累积资源 + 剩余层的最小总和 > 资源上限，则整个分支无解。
        if current_resource + remaining_min_resource > resource_limit:
            return 
            
        # 4. 递归步骤
        for scheme in valid_schemes_per_layer[layer_index]:
            
            new_resource = current_resource + scheme[2]
            
            # 辅助 DLBP 剪枝: 检查选择当前方案后，是否会触发下一层的 DLBP 剪枝条件
            if layer_index + 1 < total_layers and \
               new_resource + min_resource_suffix_sum[layer_index + 1] > resource_limit:
                 continue

            search_scheme(
                layer_index + 1,
                current_combo + [scheme], 
                current_transmission + scheme[0],
                current_cycles + scheme[1],
                new_resource
            )

    # 启动搜索：从第二层 (index 1) 开始，第一层的方案已作为 start_scheme
    dt_start, cc_start, rc_start, md_start = start_scheme
    initial_combo = [start_scheme]
    search_scheme(1, initial_combo, dt_start, cc_start, rc_start)
    
    return found_rows

def _backtracking_search_worker(worker_input: Tuple) -> List[ResultRow]:
    """
    简单剪枝回溯搜索的工作进程。只使用上限剪枝。
    """
    # 仅使用非 DLBP 相关的输入参数
    (
        start_scheme,
        valid_schemes_per_layer,
        _, # DLBP占位符，忽略
        resource_limit,
        resource_min,
        cycle_limit,
        transmission_limit,
        max_rows_per_process 
    ) = worker_input
    
    total_layers = len(valid_schemes_per_layer)
    found_rows = [] 
    local_count = 0

    def search_scheme(
        layer_index: int, 
        current_combo: List[Scheme],
        current_transmission: float,
        current_cycles: float,
        current_resource: float
    ) -> None:
        nonlocal local_count
        
        # 1. 上限剪枝 (Pruning): 实时检查当前的累积值是否超限。
        if (current_resource > resource_limit or
            current_cycles > cycle_limit or
            current_transmission > transmission_limit):
            return 

        # 2. 递归终止条件
        if layer_index == total_layers:
            # 最终检查 resource_min
            if current_resource >= resource_min:
                if local_count >= max_rows_per_process:
                    return 

                local_count += 1
                row = [0] 
                for scheme in current_combo:
                    row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                row.extend([current_transmission, current_cycles, current_resource])
                found_rows.append(row)
            return

        # 3. 递归步骤
        for scheme in valid_schemes_per_layer[layer_index]:
            
            search_scheme(
                layer_index + 1,
                current_combo + [scheme], 
                current_transmission + scheme[0],
                current_cycles + scheme[1],
                current_resource + scheme[2]
            )

    # 启动搜索：从第二层 (index 1) 开始
    dt_start, cc_start, rc_start, md_start = start_scheme
    initial_combo = [start_scheme]
    search_scheme(1, initial_combo, dt_start, cc_start, rc_start)
    
    return found_rows

def _brute_force_search_worker(worker_input: Tuple) -> List[ResultRow]:
    """
    暴力搜索的工作进程。使用 itertools.product 遍历子空间，效率最低。
    """
    (
        start_scheme,
        valid_schemes_per_layer,
        _, # DLBP占位符，忽略
        resource_limit,
        resource_min,
        cycle_limit,
        transmission_limit,
        max_rows_per_process 
    ) = worker_input

    # 暴力搜索需要剩余层的方案列表，而不是整个列表
    remaining_layer_schemes = valid_schemes_per_layer[1:]
    total_layers = len(valid_schemes_per_layer)
    found_rows = [] 
    local_count = 0

    dt_start, cc_start, rc_start, md_start = start_scheme
    
    # 使用 itertools.product 生成剩余层的组合
    for remaining_combo in itertools.product(*remaining_layer_schemes):
        if local_count >= max_rows_per_process:
            break

        # 组合完整方案：start_scheme + remaining_combo
        full_combo = [start_scheme] + list(remaining_combo)

        # 实时计算总和
        total_resource = rc_start + sum(scheme[2] for scheme in remaining_combo)
        total_transmission = dt_start + sum(scheme[0] for scheme in remaining_combo)
        total_cycles = cc_start + sum(scheme[1] for scheme in remaining_combo)
        
        # 检查是否满足所有限制条件
        if (total_resource <= resource_limit and
            total_resource >= resource_min and
            total_cycles <= cycle_limit and
            total_transmission <= transmission_limit):

            local_count += 1
            # 格式化为 ResultRow (Scheme ID 占位符设为 0)
            row = [0] 
            for scheme in full_combo:
                row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
            
            row.extend([total_transmission, total_cycles, total_resource])
            found_rows.append(row)
            
    return found_rows


# ==============================================================================
# 通用调度函数 (Parallel Dispatcher)
# ==============================================================================

def _prepare_and_dispatch(
    search_name: str,
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str, 
    max_rows: int,
    num_processes: int,
    worker_function: Callable,
    is_dlbp: bool
) -> None:
    """
    通用调度器，负责预处理、任务分配、多进程执行、结果合并和写入。
    """
    start_time = time.time()
    print(f"\n--- 运行 {search_name} 并行搜索 ---")
    
    # 步骤 1: 单层预筛选
    valid_schemes_per_layer = []
    for i, layer in enumerate(design_schemes):
        valid_layer_schemes = [scheme for scheme in layer if scheme[2] <= resource_limit]
            
        if not valid_layer_schemes:
            print(f"警告: 第 {i + 1} 层没有符合 'resource_limit' 的方案。搜索中止。")
            return 
                
        valid_schemes_per_layer.append(valid_layer_schemes)

    total_layers = len(valid_schemes_per_layer)
    first_layer_schemes = valid_schemes_per_layer[0]

    # 步骤 2: DLBP 专用预计算
    min_resource_suffix_sum = None
    if is_dlbp:
        min_resource_suffix_sum = _calculate_min_resource_suffix_sum(valid_schemes_per_layer)
        # 检查 DLBP 是否提前剪枝掉整个搜索空间
        if min_resource_suffix_sum[0] > resource_limit:
             print(f"资源上限 {resource_limit} 低于最小资源总和 {min_resource_suffix_sum[0]:.2f}。搜索空间为空。")
             return

    # 步骤 3: 多进程并行搜索调度设置
    num_processes = num_processes if num_processes is not None else os.cpu_count()
    if num_processes is None or num_processes < 1: 
        num_processes = 4
    
    num_tasks = len(first_layer_schemes)
    # 11.7进行修改，将min改成max
    num_processes = max(num_processes, num_tasks)
    max_rows_per_process = max_rows // num_processes if num_processes > 0 else max_rows
    
    # 构建 worker 输入数据列表
    worker_inputs = []
    for start_scheme in first_layer_schemes:
        # 所有 worker 输入遵循统一的结构，不相关的参数设为 None
        worker_inputs.append((
            start_scheme, 
            valid_schemes_per_layer, 
            min_resource_suffix_sum, # DLBP worker 会使用，其他忽略
            resource_limit, 
            resource_min, 
            cycle_limit, 
            transmission_limit,
            max_rows_per_process 
        ))
    
    print(f"启动 {num_processes} 个进程进行并行搜索，第一层共有 {num_tasks} 个任务。")
    
    all_results: List[List[ResultRow]] = []
    try:
        # 步骤 4: 执行并行任务
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_results = pool.map(worker_function, worker_inputs)
    except Exception as e:
        print(f"并行搜索过程中发生错误: {e}")
        # 如果是 Ctrl+C 中断，这里可能捕获到 KeyboardInterrupt
        return
        
    # 步骤 5: 合并、截断和写入结果
    final_rows = [row for worker_results in all_results for row in worker_results]
    final_rows = final_rows[:max_rows]
    total_found_count = len(final_rows)
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # 写入 Header
            header = ["Scheme ID"]
            for i in range(total_layers):
                header.extend([
                    f"Layer{i+1}_data_transmission", 
                    f"Layer{i+1}_compute_cycles", 
                    f"Layer{i+1}_resource_consumption", 
                    f"Layer{i+1}_mapping_design" 
                ])
            header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
            writer.writerow(header)

            # 写入 Data Rows (序列化 Mapping Design)
            for i, row in enumerate(final_rows):
                row[0] = i + 1 # 更新 Scheme ID
                
                # 序列化 Mapping Design
                for j in range(total_layers):
                    # mapping_design 所在的索引位置 (4, 8, 12, ...)
                    mapping_design_index = (j * 4) + 4
                    item_to_serialize = row[mapping_design_index]
                    
                    # 只有当它是 list 或 dict 时才需要序列化
                    if isinstance(item_to_serialize, (list, dict)):
                        try:
                            # 确保 JSON 字符串不包含 ASCII 以外的字符，方便中文显示
                            row[mapping_design_index] = json.dumps(item_to_serialize, ensure_ascii=False)
                        except TypeError:
                            # 如果序列化失败，保持原样（虽然不太可能发生）
                            pass 

                writer.writerow(row)
                
    except Exception as e:
        print(f"写入 CSV 文件 {output_file} 时发生错误: {e}")
        return
        
    end_time = time.time()
    
    print(f"结果已写入 {output_file} 文件中! 共找到 {total_found_count} 条记录 (最多 {max_rows} 条)。")
    print(f"总执行时间: {end_time - start_time:.2f} 秒。")

# ==============================================================================
# 用户调用接口
# ==============================================================================

def process_parallel_dlbp(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_dlbp_parallel.csv", 
    max_rows: int = 1000000,
    num_processes: int = None
) -> None:
    """
    DLBP (动态下限剪枝) 并行搜索调度函数。对资源约束有很强的剪枝效果。
    """
    _prepare_and_dispatch(
        "DLBP 并行搜索 (动态下限剪枝)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        max_rows,
        num_processes,
        _dlbp_search_worker,
        is_dlbp=True
    )


def process_parallel_backtracking(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_backtracking_parallel.csv", 
    max_rows: int = 1000000,
    num_processes: int = None
) -> None:
    """
    简单回溯剪枝并行搜索调度函数。只使用上限剪枝，性能低于 DLBP。
    """
    _prepare_and_dispatch(
        "Backtracking 并行搜索 (简单上限剪枝)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        max_rows,
        num_processes,
        _backtracking_search_worker,
        is_dlbp=False
    )

def process_parallel_brute_force(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_brute_force_parallel.csv", 
    max_rows: int = 1000000,
    num_processes: int = None
) -> None:
    """
    暴力搜索并行调度函数。遍历所有组合，无剪枝，仅靠多进程加速。
    """
    _prepare_and_dispatch(
        "Brute Force 并行搜索 (暴力/Product)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        max_rows,
        num_processes,
        _brute_force_search_worker,
        is_dlbp=False
    )
