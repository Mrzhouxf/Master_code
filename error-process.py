import multiprocessing
import csv
import os
import itertools
import time
from typing import List, Tuple, Any

# --- 辅助函数：核心递归搜索逻辑 (用于 DLBP 优化搜索) ---

# 必须将递归函数定义在外部，才能被多进程 Pool 调用
def _search_worker_recursive(
    layer_index: int, 
    current_combo: List[Tuple[float, float, float, Any]],
    current_transmission: float,
    current_cycles: float,
    current_resource: float,
    valid_schemes_per_layer: List[List[Tuple[float, float, float, Any]]],
    min_resource_suffix_sum: List[float],
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float,
    writer, # 传入 CSV writer 对象
    max_rows: int,
    global_count_lock, # 传入锁对象
    global_count_value # 传入可共享计数值
):
    """
    核心递归搜索逻辑，由多进程工作池调用。
    使用动态下限剪枝 (DLBP) 来优化回溯搜索。
    """
    
    # 检查是否已达到全局最大记录数
    with global_count_lock:
        if global_count_value.value >= max_rows:
            return

    # **上界剪枝 (Upper Bound Pruning):**
    if (current_cycles > cycle_limit or
        current_transmission > transmission_limit):
        return

    # 递归终止条件: 已经为所有层选择了方案
    if layer_index == len(valid_schemes_per_layer):
        
        # 最终检查 resource_min
        if current_resource >= resource_min:
            # 写入数据行
            # 注意：这里需要在递归调用处检查 resource_limit，并在这里检查 resource_min
            with global_count_lock:
                if global_count_value.value < max_rows:
                    global_count_value.value += 1
                    
                    row = [global_count_value.value]
                    for scheme in current_combo:
                        # scheme[0]=transmission, scheme[1]=cycles, scheme[2]=resource, scheme[3]=design
                        row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                    row.extend([current_transmission, current_cycles, current_resource])
                    # writer 是由父进程 Manager 创建的，确保线程安全写入
                    writer.writerow(row) 
        return

    # **动态下限剪枝 (DLBP)**
    # 计算剩余所有层的最小资源总和
    remaining_min_resource = min_resource_suffix_sum[layer_index]
    
    # 递归步骤: 遍历当前层的每一个可行方案
    for scheme in valid_schemes_per_layer[layer_index]:
        
        new_resource = current_resource + scheme[2]
        
        # 核心优化剪枝 1: 检查当前方案是否会导致 Resource 上限超限
        if new_resource > resource_limit:
            continue
            
        # 核心优化剪枝 2: 检查当前资源 + 剩余层的最小资源总和是否会超限
        # 这里的 min_resource_suffix_sum[layer_index + 1] 是指从下一层开始的最小后缀和
        if new_resource + min_resource_suffix_sum[layer_index + 1] > resource_limit:
            continue

        # 递归调用自身，进入下一层
        _search_worker_recursive(
            layer_index + 1,
            current_combo + [scheme], 
            current_transmission + scheme[0],
            current_cycles + scheme[1],
            new_resource,
            valid_schemes_per_layer,
            min_resource_suffix_sum,
            resource_limit, resource_min, cycle_limit, transmission_limit,
            writer,
            max_rows,
            global_count_lock,
            global_count_value
        )

# --- 辅助函数：核心暴力搜索逻辑 (用于纯暴力搜索) ---

def _bruteforce_worker(worker_data):
    """
    多进程暴力搜索的工作函数。每个进程负责处理一个或多个第一层方案。
    """
    # worker_data = (start_scheme, remaining_schemes, resource_limit, resource_min, cycle_limit, transmission_limit, temp_file_path)
    start_scheme, remaining_schemes, resource_limit, resource_min, cycle_limit, transmission_limit, temp_file = worker_data
    
    # 每个进程独立写入其临时文件
    with open(temp_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 迭代剩余层的所有组合
        remaining_product = itertools.product(*remaining_schemes)
        
        for combo_tail in remaining_product:
            # 构建完整的方案组合
            combo = [start_scheme] + list(combo_tail)
            
            # 计算总和
            total_transmission = sum(scheme[0] for scheme in combo)
            total_cycles = sum(scheme[1] for scheme in combo)
            total_resource = sum(scheme[2] for scheme in combo) # Resource calculation here
            
            # 检查是否满足所有限制条件
            if (total_resource <= resource_limit and
                total_resource >= resource_min and
                total_cycles <= cycle_limit and
                total_transmission <= transmission_limit):

                # 写入数据行 (Scheme ID 占位，最终在合并时统一编号)
                # row structure: [Placeholder ID] + [L1 data, L1 cycles, L1 resource, L1 design] + ... + [Totals]
                row = [0] # Placeholder for Scheme ID
                for scheme in combo:
                    row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                row.extend([total_transmission, total_cycles, total_resource])
                writer.writerow(row)
                
    return temp_file # 返回临时文件名

# --- 辅助函数：核心简单回溯逻辑 (仅上界剪枝) ---

def _simple_backtrack_worker_recursive(
    layer_index: int, 
    current_combo: List[Tuple[float, float, float, Any]],
    current_transmission: float,
    current_cycles: float,
    current_resource: float,
    valid_schemes_per_layer: List[List[Tuple[float, float, float, Any]]],
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float,
    writer, # 传入 CSV writer 对象
    max_rows: int,
    global_count_lock, # 传入锁对象
    global_count_value # 传入可共享计数值
):
    """
    核心递归搜索逻辑，仅使用上界剪枝，由多进程工作池调用。
    """
    
    # 检查是否已达到全局最大记录数
    with global_count_lock:
        if global_count_value.value >= max_rows:
            return

    # **上界剪枝 (Upper Bound Pruning):**
    if (current_resource > resource_limit or
        current_cycles > cycle_limit or
        current_transmission > transmission_limit):
        return

    # 递归终止条件: 已经为所有层选择了方案
    if layer_index == len(valid_schemes_per_layer):
        
        # 最终检查 resource_min
        if current_resource >= resource_min:
            # 写入数据行
            with global_count_lock:
                if global_count_value.value < max_rows:
                    global_count_value.value += 1
                    
                    row = [global_count_value.value]
                    for scheme in current_combo:
                        # scheme[0]=transmission, scheme[1]=cycles, scheme[2]=resource, scheme[3]=design
                        row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                    row.extend([current_transmission, current_cycles, current_resource])
                    writer.writerow(row)
        return

    # 递归步骤: 遍历当前层的每一个可行方案
    for scheme in valid_schemes_per_layer[layer_index]:
        
        # 递归调用自身，进入下一层
        _simple_backtrack_worker_recursive(
            layer_index + 1,
            current_combo + [scheme], 
            current_transmission + scheme[0],
            current_cycles + scheme[1],
            current_resource + scheme[2],
            valid_schemes_per_layer,
            resource_limit, resource_min, cycle_limit, transmission_limit,
            writer,
            max_rows,
            global_count_lock,
            global_count_value
        )


# --- 主并行函数 1: 优化回溯搜索 (DLBP) ---

def process_neural_network_design_optimized_parallel(
    design_schemes: List[List[Tuple[float, float, float, Any]]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_optimized.csv", 
    max_rows: int = 1000000,
    num_processes: int = None # 默认使用 CPU 核心数
) -> None:
    """
    使用多进程（multiprocessing）和 DLBP 优化回溯搜索，以加速大型网络的运行。
    """
    
    print(f"\n--- 1. 开始优化回溯搜索 (DLBP) ---")
    
    # 步骤 1: 单层预筛选 & 预计算最小资源
    valid_schemes_per_layer = []
    min_resource_per_layer = [] 
    
    for i, layer in enumerate(design_schemes):
        valid_layer_schemes = []
        current_min_resource = float('inf')
        
        for scheme in layer:
            # 只预筛选单个方案的资源上限
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
                if scheme[2] < current_min_resource:
                    current_min_resource = scheme[2]
        
        if not valid_layer_schemes:
            print(f"警告：第 {i + 1} 层没有可用的方案。")
            return
            
        valid_schemes_per_layer.append(valid_layer_schemes)
        min_resource_per_layer.append(current_min_resource)

    total_layers = len(valid_schemes_per_layer)
    
    # 步骤 2: 计算最小资源后缀和 (R_min_k -> N)
    # min_resource_suffix_sum[i] 存储的是从第 i 层到最后一层 (N-1) 的最小资源总和。
    min_resource_suffix_sum = [0.0] * (total_layers + 1)
    for i in range(total_layers - 1, -1, -1):
        min_resource_suffix_sum[i] = min_resource_suffix_sum[i+1] + min_resource_per_layer[i]
        
    if min_resource_suffix_sum[0] > resource_limit:
        print(f"警告：最小资源需求总和 ({min_resource_suffix_sum[0]:.2f}) 超限。")
        return

    # ----------------------------------------------------
    # 步骤 3: 多进程并行搜索
    # ----------------------------------------------------
    
    # 使用 Manager 来管理进程间的共享状态
    manager = multiprocessing.Manager()
    global_count_value = manager.Value('i', 0) # 共享的记录计数器
    global_count_lock = manager.Lock()         # 共享的锁，用于控制写入和计数
    
    # 获取第一层的所有方案，作为并行任务的起点
    first_layer_schemes = valid_schemes_per_layer[0]

    # 定义 worker 包装函数，每个进程将从一个特定的 'start_scheme' 开始搜索
    def worker_wrapper_optimized(start_scheme):
        # 创建临时 CSV 文件，文件名包含进程 ID
        temp_file = f"{output_file}.temp.{os.getpid()}.csv"
        
        # 每个进程独立打开文件，并使用共享计数器和锁进行写入控制
        with open(temp_file, 'w', newline='') as csvfile:
            # 需要在进程内创建 csv.writer，因为 writer 对象不能跨进程共享
            writer = csv.writer(csvfile) 
            
            # 启动递归搜索，从第一层 (索引 0) 的单个方案开始
            _search_worker_recursive(
                layer_index=1, # 从第二层开始递归
                current_combo=[start_scheme], # 初始方案组合包含第一个方案
                current_transmission=start_scheme[0],
                current_cycles=start_scheme[1],
                current_resource=start_scheme[2],
                valid_schemes_per_layer=valid_schemes_per_layer,
                min_resource_suffix_sum=min_resource_suffix_sum,
                resource_limit=resource_limit, 
                resource_min=resource_min, 
                cycle_limit=cycle_limit, 
                transmission_limit=transmission_limit,
                writer=writer,
                max_rows=max_rows,
                global_count_lock=global_count_lock,
                global_count_value=global_count_value
            )
        return temp_file # 返回临时文件名

    # 创建进程池
    num_processes = num_processes if num_processes is not None else os.cpu_count()
    if num_processes is None:
        num_processes = 4 # Fallback
    
    print(f"启动 {num_processes} 个进程进行并行搜索...")
    
    temp_files = []
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 将第一层的所有方案作为参数，分配给进程池
            # 注意：如果第一层方案过多，可能需要分批处理
            temp_files = pool.map(worker_wrapper_optimized, first_layer_schemes)
    except Exception as e:
        print(f"并行搜索过程中发生错误: {e}")
        return
    finally:
        # 清理可能产生的空临时文件
        temp_files = [f for f in temp_files if os.path.exists(f) and os.path.getsize(f) > 0]
        
    # ----------------------------------------------------
    # 步骤 4: 合并结果并写入最终文件
    # ----------------------------------------------------
    
    total_found_count = 0
    
    # 写入最终的 CSV 文件（包括表头）
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 写入表头 
        header = ["Scheme ID"]
        for i in range(total_layers):
            header.extend([
                f"Layer{i+1}_data_transmission", 
                f"Layer{i+1}_compute_cycles", 
                f"Layer{i+1}_resource", 
                f"Layer{i+1}_mapping_design"
            ])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 合并所有临时文件
        for temp_file in temp_files:
            if total_found_count >= max_rows:
                break # 达到最大行数限制，停止合并

            try:
                with open(temp_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        if total_found_count < max_rows:
                            total_found_count += 1
                            # 更新 Scheme ID 并写入主文件
                            row[0] = total_found_count 
                            writer.writerow(row)
                        else:
                            break
            except Exception as e:
                print(f"合并临时文件 {temp_file} 时出错: {e}")
            finally:
                # 完成后删除临时文件
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                    
    print(f"结果已写入 {output_file} 文件中！共找到 {total_found_count} 条记录（最多 {max_rows} 条）。")


# --- 主并行函数 2: 暴力搜索 (基于 itertools.product) ---

def process_neural_network_design_bruteforce_parallel(
    design_schemes: List[List[Tuple[float, float, float, Any]]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_bruteforce_parallel.csv", 
    max_rows: int = 1000000,
    num_processes: int = None # 默认使用 CPU 核心数
) -> None:
    """
    使用多进程（multiprocessing）并行化暴力搜索（基于 itertools.product）。
    """
    
    print(f"\n--- 2. 开始并行暴力搜索 (Bruteforce) ---")
    
    # 步骤 1: 单层预筛选
    valid_schemes_per_layer = []
    
    for i, layer in enumerate(design_schemes):
        valid_layer_schemes = []
        for scheme in layer:
            # 仅预筛选单个方案的资源上限
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
        
        if not valid_layer_schemes:
            print(f"警告：第 {i + 1} 层没有可用的方案。")
            return
            
        valid_schemes_per_layer.append(valid_layer_schemes)

    total_layers = len(valid_schemes_per_layer)
    
    # ----------------------------------------------------
    # 步骤 2: 多进程并行搜索
    # ----------------------------------------------------
    
    # 获取第一层的所有方案，作为并行任务的起点
    first_layer_schemes = valid_schemes_per_layer[0]
    # 获取剩余所有层的方案列表，用于 itertools.product
    remaining_schemes = valid_schemes_per_layer[1:]
    
    # 构建 worker 的输入数据列表
    worker_inputs = [
        (
            start_scheme, 
            remaining_schemes, 
            resource_limit, 
            resource_min, 
            cycle_limit, 
            transmission_limit, 
            f"{output_file}.temp.{i}_{multiprocessing.current_process().pid}.csv" # 确保临时文件名唯一
        ) 
        for i, start_scheme in enumerate(first_layer_schemes)
    ]
    
    # 创建进程池
    num_processes = num_processes if num_processes is not None else os.cpu_count()
    if num_processes is None:
        num_processes = 4 # Fallback
    
    print(f"启动 {num_processes} 个进程进行并行暴力搜索...")
    
    temp_files = []
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            temp_files = list(pool.map(_bruteforce_worker, worker_inputs))
    except Exception as e:
        print(f"并行搜索过程中发生错误: {e}")
        # 即使出错也尝试清理
    finally:
        # 清理可能产生的空临时文件
        temp_files = [f for f in temp_files if os.path.exists(f) and os.path.getsize(f) > 0]
        
    # ----------------------------------------------------
    # 步骤 3: 合并结果
    # ----------------------------------------------------
    
    total_found_count = 0
    
    # 写入最终的 CSV 文件（包括表头）
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 写入表头
        header = ["Scheme ID"]
        for i in range(total_layers):
            header.extend([
                f"Layer{i+1}_data_transmission", 
                f"Layer{i+1}_compute_cycles", 
                f"Layer{i+1}_resource", 
                f"Layer{i+1}_mapping_design"
            ])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 合并所有临时文件
        for temp_file in temp_files:
            if total_found_count >= max_rows:
                break # 达到最大行数限制，停止合并

            try:
                with open(temp_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        # row[0] 是 Scheme ID 的占位符 '0'
                        if total_found_count < max_rows:
                            total_found_count += 1
                            # 更新 Scheme ID 并写入主文件
                            row[0] = total_found_count 
                            writer.writerow(row)
                        else:
                            break
            except Exception as e:
                print(f"合并临时文件 {temp_file} 时出错: {e}")
            finally:
                # 完成后删除临时文件
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                    
    print(f"结果已写入 {output_file} 文件中！共找到 {total_found_count} 条记录（最多 {max_rows} 条）。")


# --- 主并行函数 3: 简单回溯搜索 (仅上界剪枝) ---

def process_neural_network_design_backtrack_parallel(
    design_schemes: List[List[Tuple[float, float, float, Any]]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_backtrack_parallel.csv", 
    max_rows: int = 1000000,
    num_processes: int = None # 默认使用 CPU 核心数
) -> None:
    """
    使用多进程（multiprocessing）并行化简单上界剪枝的回溯搜索。
    """
    
    print(f"\n--- 3. 开始并行简单回溯搜索 ---")
    
    # 步骤 1: 单层预筛选
    valid_schemes_per_layer = []
    
    for i, layer in enumerate(design_schemes):
        valid_layer_schemes = []
        for scheme in layer:
            # scheme[2] 是单个方案的资源消耗
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
        
        if not valid_layer_schemes:
            print(f"警告：第 {i + 1} 层没有可用的方案。")
            return
            
        valid_schemes_per_layer.append(valid_layer_schemes)

    total_layers = len(valid_schemes_per_layer)
    
    # ----------------------------------------------------
    # 步骤 2: 多进程并行搜索
    # ----------------------------------------------------
    
    # 使用 Manager 来管理进程间的共享状态
    manager = multiprocessing.Manager()
    global_count_value = manager.Value('i', 0) # 共享的记录计数器
    global_count_lock = manager.Lock()         # 共享的锁，用于控制写入和计数
    
    # 获取第一层的所有方案，作为并行任务的起点
    first_layer_schemes = valid_schemes_per_layer[0]

    # 定义 worker 包装函数
    def worker_wrapper_simple_backtrack(start_scheme):
        # 创建临时 CSV 文件，文件名包含进程 ID
        temp_file = f"{output_file}.temp.{os.getpid()}.csv"
        
        # 每个进程独立打开文件，并使用共享计数器和锁进行写入控制
        with open(temp_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 启动递归搜索，从第二层开始
            _simple_backtrack_worker_recursive(
                layer_index=1, # 从第二层开始递归
                current_combo=[start_scheme], # 初始方案组合包含第一个方案
                current_transmission=start_scheme[0],
                current_cycles=start_scheme[1],
                current_resource=start_scheme[2],
                valid_schemes_per_layer=valid_schemes_per_layer,
                resource_limit=resource_limit, 
                resource_min=resource_min, 
                cycle_limit=cycle_limit, 
                transmission_limit=transmission_limit,
                writer=writer,
                max_rows=max_rows,
                global_count_lock=global_count_lock,
                global_count_value=global_count_value
            )
        return temp_file # 返回临时文件名

    # 创建进程池
    num_processes = num_processes if num_processes is not None else os.cpu_count()
    if num_processes is None:
        num_processes = 4 # Fallback
    
    print(f"启动 {num_processes} 个进程进行并行回溯搜索 (仅上界剪枝)...")
    
    temp_files = []
    try:
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 将第一层的所有方案作为参数，分配给进程池
            temp_files = pool.map(worker_wrapper_simple_backtrack, first_layer_schemes)
    except Exception as e:
        print(f"并行搜索过程中发生错误: {e}")
        return
    finally:
        # 清理可能产生的空临时文件
        temp_files = [f for f in temp_files if os.path.exists(f) and os.path.getsize(f) > 0]
        
    # ----------------------------------------------------
    # 步骤 3: 合并结果
    # ----------------------------------------------------
    
    total_found_count = 0
    
    # 写入最终的 CSV 文件（包括表头）
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # 写入表头
        header = ["Scheme ID"]
        for i in range(total_layers):
            header.extend([
                f"Layer{i+1}_data_transmission", 
                f"Layer{i+1}_compute_cycles", 
                f"Layer{i+1}_resource", 
                f"Layer{i+1}_mapping_design"
            ])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 合并所有临时文件
        for temp_file in temp_files:
            if total_found_count >= max_rows:
                break # 达到最大行数限制，停止合并

            try:
                with open(temp_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        if total_found_count < max_rows:
                            total_found_count += 1
                            # 更新 Scheme ID 并写入主文件
                            row[0] = total_found_count 
                            writer.writerow(row)
                        else:
                            break
            except Exception as e:
                print(f"合并临时文件 {temp_file} 时出错: {e}")
            finally:
                # 完成后删除临时文件
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
                    
    print(f"结果已写入 {output_file} 文件中！共找到 {total_found_count} 条记录（最多 {max_rows} 条）。")


