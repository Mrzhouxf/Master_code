import csv
import os
import time
import multiprocessing
import json 
import itertools
import math
from typing import List, Tuple, Any, Dict, Union, Callable, Optional, Set

# 类型定义
# Scheme 结构: (data_transmission, compute_cycles, resource_consumption, mapping_design)
Scheme = Tuple[float, float, float, Any] 
ResultRow = List[Any] # 用于列表模式的结果行
# OptimizationResult 结构: (total_transmission, total_cycles, total_resource, full_scheme_combo)
OptimizationResult = Tuple[float, float, float, List[Scheme]]

# 优化目标索引
TRANSMISSION_IDX = 0
CYCLES_IDX = 1
RESOURCE_IDX = 2

# ==============================================================================
# DLBP 辅助函数 (后缀和计算)
# ==============================================================================

def _calculate_min_suffix_sum(
    valid_schemes_per_layer: List[List[Scheme]], 
    metric_index: int
) -> List[float]:
    """
    计算从当前层到最后一层所有层中，指定指标的最小总和（后缀和）。
    用于 DLBP (Resource) 和 Cycles/Transmission 的下限剪枝。
    """
    total_layers = len(valid_schemes_per_layer)
    # 数组长度为 total_layers + 1，min_suffix_sum[i] 存储从 i 层开始的最小总和
    min_suffix_sum = [0.0] * (total_layers + 1)
    
    # 确保方案列表非空，否则 min() 会报错
    if not valid_schemes_per_layer:
        return min_suffix_sum

    # 从后往前计算
    current_min_sum = 0.0
    for i in range(total_layers - 1, -1, -1):
        if not valid_schemes_per_layer[i]:
            # 如果某层没有方案，则最小总和计算中断，但这不是常态
            min_layer_metric = 0.0
        else:
            # 找到当前层 i 的最小指标值
            min_layer_metric = min(scheme[metric_index] for scheme in valid_schemes_per_layer[i])
        
        current_min_sum += min_layer_metric
        min_suffix_sum[i] = current_min_sum
        
    return min_suffix_sum

# ==============================================================================
# 核心工作函数 (Worker Functions) - 优化模式 (Optimizer Modes)
# ==============================================================================

def _dlbp_opt_flexible_worker(worker_input: Tuple) -> Optional[OptimizationResult]:
    """
    灵活优化模式的工作进程：支持 H, L, U, C 模式，使用 DLBP 和动态上界剪枝。
    """
    (
        start_scheme,
        valid_schemes_per_layer,
        min_resource_suffix_sum,
        min_cycles_suffix_sum,
        min_transmission_suffix_sum,
        optimization_target, # 'H', 'L', 'U', 'C', 'R'
        norm_max_cycles, # U Mode 专用
        norm_max_transmission, # U Mode 专用
        weights, # U Mode 专用 (W_T, W_C)
        resource_limit,
        resource_min,
        cycle_limit,
        transmission_limit,
    ) = worker_input
    
    total_layers = len(valid_schemes_per_layer)
    
    # 局部最优解追踪
    best_value = float('inf') 
    best_scheme_combo: Optional[List[Scheme]] = None
    best_totals: Optional[Tuple[float, float, float]] = None # (transmission, cycles, resource)

    # 定义目标函数索引
    primary_idx = -1 
    if optimization_target in ('H', 'C'): # 最小化 Cycles
        primary_idx = CYCLES_IDX
    elif optimization_target == 'L': # 最小化 Transmission
        primary_idx = TRANSMISSION_IDX
    elif optimization_target == 'R': # 最小化 Resource
        primary_idx = RESOURCE_IDX
    # U 模式没有简单的 primary_idx

    def calculate_utility(current_transmission: float, current_cycles: float) -> float:
        """计算 U 模式的加权效用分数 (需要先归一化)。"""
        # U 模式的归一化基准由调度器传入
        if norm_max_cycles == 0 or norm_max_transmission == 0:
            return float('inf') 
        
        # 归一化：将值映射到 [0, 1] 范围内
        norm_cycles = current_cycles / norm_max_cycles
        norm_transmission = current_transmission / norm_max_transmission
        
        # 计算加权分数
        wt, wc = weights
        return wt * norm_transmission + wc * norm_cycles
        
    def check_and_update_best(
        current_transmission: float, 
        current_cycles: float, 
        current_resource: float,
        current_combo: List[Scheme]
    ) -> bool:
        """检查是否是更优解，并更新局部最优解"""
        nonlocal best_value, best_scheme_combo, best_totals

        # 1. 计算当前方案的比较值 (主目标)
        if optimization_target == 'U':
            current_value = calculate_utility(current_transmission, current_cycles)
        else:
            current_value = [current_transmission, current_cycles, current_resource][primary_idx]
        
        # 2. 检查是否为新最优值
        is_better = False
        if current_value < best_value:
            is_better = True
        elif current_value == best_value:
            # 平局判断 (次要目标优化)
            if best_totals is None: # 第一次找到解
                is_better = True
            elif optimization_target in ('H', 'C'):
                # H/C: Cycles 相同，比较 Transmission (次要目标: 最小化 Transmission)
                if current_transmission < best_totals[TRANSMISSION_IDX]:
                    is_better = True
            elif optimization_target == 'L':
                # L: Transmission 相同，比较 Cycles (次要目标: 最小化 Cycles)
                if current_cycles < best_totals[CYCLES_IDX]:
                    is_better = True
            # R/U 模式平局不作处理，保留第一个找到的解
        
        if is_better:
            best_value = current_value
            best_scheme_combo = current_combo
            best_totals = (current_transmission, current_cycles, current_resource)
            return True
        return False

    def search_scheme(
        layer_index: int, 
        current_combo: List[Scheme],
        current_transmission: float,
        current_cycles: float,
        current_resource: float
    ) -> None:
        
        # 1. 静态上限检查 (立即剪枝)
        if (current_resource > resource_limit or
            current_cycles > cycle_limit or
            current_transmission > transmission_limit):
            return 

        # 2. 递归终止条件
        if layer_index == total_layers:
            # 找到一个可行解，检查是否为局部最优
            # 资源下限检查理论上在 DLBP 剪枝中已覆盖，这里仅作最终验证
            check_and_update_best(current_transmission, current_cycles, current_resource, current_combo)
            return
            
        # 3. 动态/下限剪枝 (在递归进入前检查)
        
        # 注意：这里的后缀和是从 layer_index 开始的 (包含 layer_index)
        # 在 _calculate_min_suffix_sum 中，min_suffix_sum[i] 存储从 i 层开始的最小总和
        # 但在 DLBP 剪枝中，我们检查的是当前层选择 scheme 后的**剩余路径**
        
        # 剩余层的最小总和 (从下一层开始计算)
        remaining_min_resource = min_resource_suffix_sum[layer_index + 1]
        remaining_min_cycles = min_cycles_suffix_sum[layer_index + 1]
        remaining_min_transmission = min_transmission_suffix_sum[layer_index + 1]


        # 4. 递归步骤
        for scheme in valid_schemes_per_layer[layer_index]:
            
            new_transmission = current_transmission + scheme[TRANSMISSION_IDX]
            new_cycles = current_cycles + scheme[CYCLES_IDX]
            new_resource = current_resource + scheme[RESOURCE_IDX]
            
            # --- 剪枝检查 (DLBP 和硬约束启发式) ---
            
            # A. DLBP 资源下限剪枝 (硬约束剪枝)
            if new_resource + remaining_min_resource > resource_limit:
                 continue
                 
            # B. Cycles/Transmission 下限剪枝 (硬约束剪枝 - 优化新增)
            # 如果当前累计值加上剩余层的最小可能值，就已经超出硬性限制，则剪枝
            if new_cycles + remaining_min_cycles > cycle_limit:
                continue
            if new_transmission + remaining_min_transmission > transmission_limit:
                continue
                 
            # C. 动态上界剪枝 (优化目标软剪枝) - 只有找到第一个解后才生效
            if best_totals is not None and optimization_target != 'U':
                
                # C1. 计算主目标下限
                current_opt_value = [new_transmission, new_cycles, new_resource][primary_idx]
                
                remaining_min_metric = 0.0
                if optimization_target in ('H', 'C'): # 最小化 Cycles
                    remaining_min_metric = remaining_min_cycles
                elif optimization_target == 'L': # 最小化 Transmission
                    remaining_min_metric = remaining_min_transmission
                elif optimization_target == 'R': # 最小化 Resource
                    remaining_min_metric = remaining_min_resource
                    
                # 【H 模式核心逻辑修复】
                # 如果当前累积值 + 剩余层最小指标 > 当前找到的最佳值，则剪枝
                # 注意：这里必须是严格大于 (>), 因为如果等于 (=)，则该路径有可能通过次要目标找到更优解。
                if current_opt_value + remaining_min_metric > best_value:
                    continue
            
            # 递归调用自身，进入下一层
            search_scheme(
                layer_index + 1,
                current_combo + [scheme], 
                new_transmission,
                new_cycles,
                new_resource
            )

    # 启动搜索：从第二层 (index 1) 开始
    dt_start, cc_start, rc_start, md_start = start_scheme
    initial_combo = [start_scheme]
    
    # 初始化 best_value 的上界
    if optimization_target == 'U':
         best_value = calculate_utility(cycle_limit, transmission_limit) + 1.0
    elif optimization_target == 'R':
         best_value = resource_limit + 1.0
    else: # H, L, C modes
         # 使用硬性上限作为初始上界
         best_value = [transmission_limit, cycle_limit, resource_limit][primary_idx] + 1.0
        
    # 由于第一层已经被选中 (start_scheme)，我们从第二层开始搜索 (index 1)
    search_scheme(1, initial_combo, dt_start, cc_start, rc_start)
    
    if best_scheme_combo is not None:
        return (best_totals[0], best_totals[1], best_totals[2], best_scheme_combo)
    return None

# ==============================================================================
# 通用调度函数 - 优化模式 (Optimizer Dispatcher)
# ==============================================================================

def _prepare_and_dispatch_optimizer(
    search_name: str,
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str, 
    num_processes: int,
    optimization_target: str, # 'H', 'L', 'C', 'U', 'R'
    weights: Tuple[float, float] = (0.5, 0.5) # (W_T, W_C)
) -> None:
    """
    通用调度器，用于优化模式（查找单个最优解）。
    """
    start_time = time.time()
    print(f"\n--- 运行 {search_name} 并行搜索 (优化模式) ---")

    # 步骤 1: 单层预筛选
    valid_schemes_per_layer = []
    max_cycles_global = 0.0 
    max_transmission_global = 0.0 

    for i, layer in enumerate(design_schemes):
        # 过滤掉资源超限的方案
        valid_layer_schemes = [scheme for scheme in layer if scheme[RESOURCE_IDX] <= resource_limit]
        if not valid_layer_schemes:
            print(f"警告: 第 {i + 1} 层没有符合 'resource_limit' 的方案。搜索中止。")
            return 
        valid_schemes_per_layer.append(valid_layer_schemes)
        
        # 计算归一化基准 (U mode only)
        if optimization_target == 'U':
            max_cycles_global += max(scheme[CYCLES_IDX] for scheme in valid_layer_schemes)
            max_transmission_global += max(scheme[TRANSMISSION_IDX] for scheme in valid_layer_schemes)

    total_layers = len(valid_schemes_per_layer)
    if total_layers == 0:
        print("网络结构为空，搜索中止。")
        return
        
    first_layer_schemes = valid_schemes_per_layer[0]

    # 【优化点】为了让动态上界剪枝尽快生效，对第一层方案按主目标升序排序
    primary_idx = -1 
    if optimization_target in ('H', 'C'): 
        primary_idx = CYCLES_IDX
    elif optimization_target == 'L': 
        primary_idx = TRANSMISSION_IDX
    elif optimization_target == 'R': 
        primary_idx = RESOURCE_IDX
        
    if primary_idx != -1:
        # 仅对 H/L/C/R 模式按主目标排序
        first_layer_schemes.sort(key=lambda s: s[primary_idx])

    # 步骤 2: DLBP & 优化下限剪枝专用预计算
    min_resource_suffix_sum = _calculate_min_suffix_sum(valid_schemes_per_layer, RESOURCE_IDX)
    min_cycles_suffix_sum = _calculate_min_suffix_sum(valid_schemes_per_layer, CYCLES_IDX)
    min_transmission_suffix_sum = _calculate_min_suffix_sum(valid_schemes_per_layer, TRANSMISSION_IDX)
    
    # 检查 DLBP 是否提前剪枝掉整个搜索空间
    if min_resource_suffix_sum[0] > resource_limit:
         print(f"资源上限 {resource_limit} 低于最小资源总和 {min_resource_suffix_sum[0]:.2f}。搜索空间为空。")
         return
    if min_cycles_suffix_sum[0] > cycle_limit:
         print(f"周期上限 {cycle_limit} 低于最小周期总和 {min_cycles_suffix_sum[0]:.2f}。搜索空间为空。")
         return
    if min_transmission_suffix_sum[0] > transmission_limit:
         print(f"传输上限 {transmission_limit} 低于最小传输总和 {min_transmission_suffix_sum[0]:.2f}。搜索空间为空。")
         return


    # 步骤 3: 多进程并行搜索调度设置
    num_processes = num_processes if num_processes is not None else os.cpu_count()
    if num_processes is None or num_processes < 1: 
        num_processes = 4
    
    num_tasks = len(first_layer_schemes)
    num_processes = min(num_processes, num_tasks)
    
    # 构建 worker 输入数据列表
    worker_inputs = []
    for start_scheme in first_layer_schemes:
        # Optimization worker 需要所有下限后缀和
        worker_inputs.append((
            start_scheme, 
            valid_schemes_per_layer, 
            min_resource_suffix_sum,
            min_cycles_suffix_sum,
            min_transmission_suffix_sum,
            optimization_target,
            max_cycles_global if optimization_target == 'U' else 0.0,
            max_transmission_global if optimization_target == 'U' else 0.0,
            weights if optimization_target == 'U' else (0.0, 0.0),
            resource_limit, 
            resource_min, 
            cycle_limit, 
            transmission_limit,
        ))
    
    print(f"启动 {num_processes} 个进程进行并行搜索，第一层共有 {num_tasks} 个任务。")
    
    all_results: List[Optional[OptimizationResult]] = []
    try:
        # 步骤 4: 执行并行任务
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 统一使用灵活工作函数
            all_results = pool.map(_dlbp_opt_flexible_worker, worker_inputs)
    except Exception as e:
        print(f"并行搜索过程中发生错误: {e}")
        return
        
    # 步骤 5: 合并并选择全局最优解
    valid_results = [res for res in all_results if res is not None]
    
    if not valid_results:
        print(f"未在约束条件下找到任何可行方案。")
        return
        
    # 比较局部最优解，找出全局最优解
    best_result: OptimizationResult = valid_results[0]
    best_value = float('inf')
    
    for res in valid_results:
        current_transmission, current_cycles, current_resource, _ = res
        current_value = float('inf')
        
        # 重新计算比较值
        if optimization_target == 'U':
            if max_cycles_global > 0 and max_transmission_global > 0:
                wt, wc = weights
                norm_cycles = current_cycles / max_cycles_global
                norm_transmission = current_transmission / max_transmission_global
                current_value = wt * norm_transmission + wc * norm_cycles
        elif optimization_target == 'R':
            current_value = current_resource
        else: # H, L, C
            current_value = [current_transmission, current_cycles, current_resource][CYCLES_IDX if optimization_target in ('H', 'C') else TRANSMISSION_IDX]
        
        # 比较逻辑 (使用与 check_and_update_best 相同的逻辑)
        if current_value < best_value:
            best_value = current_value
            best_result = res
        elif current_value == best_value:
             # 平局处理
            if optimization_target in ('H', 'C'):
                if current_transmission < best_result[TRANSMISSION_IDX]:
                    best_result = res
            elif optimization_target == 'L':
                if current_cycles < best_result[CYCLES_IDX]:
                    best_result = res
            
    # 步骤 6: 格式化结果并写入
    final_row = _format_optimization_result(best_result, total_layers)
    
    # 写入结果
    _write_lister_results(output_file, [final_row], total_layers, 1)

    end_time = time.time()
    
    # 打印最终结果
    if optimization_target == 'U':
         print(f"全局最优方案已写入 {output_file} 文件中。目标: 最小加权效用 (W_T={weights[0]}, W_C={weights[1]})。")
         print(f"最优效用分数: {best_value:.4f}")
    elif optimization_target in ('H', 'C'):
         print(f"全局最优方案已写入 {output_file} 文件中。目标: 最小 Cycles (次要目标: 最小 Transmission)。")
         print(f"最优 Cycles: {best_result[CYCLES_IDX]:.4f}, 最优 Transmission: {best_result[TRANSMISSION_IDX]:.4f}")
    elif optimization_target == 'L':
         print(f"全局最优方案已写入 {output_file} 文件中。目标: 最小 Transmission (次要目标: 最小 Cycles)。")
         print(f"最优 Transmission: {best_result[TRANSMISSION_IDX]:.4f}, 最优 Cycles: {best_result[CYCLES_IDX]:.4f}")
    elif optimization_target == 'R':
         print(f"全局最优方案已写入 {output_file} 文件中。目标: 最小 Resource。")
         print(f"最优 Resource: {best_result[RESOURCE_IDX]:.4f}, 对应的 Cycles: {best_result[CYCLES_IDX]:.4f}")
         
    print(f"总执行时间: {end_time - start_time:.2f} 秒。")

# ==============================================================================
# 辅助写入函数 (与之前相同)
# ==============================================================================

def _format_optimization_result(result: OptimizationResult, total_layers: int) -> ResultRow:
    """将 OptimizationResult 格式化为 ResultRow"""
    total_transmission, total_cycles, total_resource, full_combo = result
    
    row = [1] # Scheme ID 设为 1
    for scheme in full_combo:
        row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
    row.extend([total_transmission, total_cycles, total_resource])
    return row

def _write_lister_results(output_file: str, final_rows: List[ResultRow], total_layers: int, max_rows: int) -> None:
    """
    负责将列表模式或优化模式的结果写入 CSV 文件。
    """
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

            # 写入 Data Rows
            for i, row in enumerate(final_rows):
                row[0] = i + 1 # 更新 Scheme ID
                
                # 序列化 Mapping Design (索引 4, 8, 12, ...)
                for j in range(total_layers):
                    mapping_design_index = (j * 4) + 4
                    item_to_serialize = row[mapping_design_index]
                    
                    if isinstance(item_to_serialize, (list, dict)):
                        try:
                            # 确保 JSON 字符串不包含 ASCII 以外的字符
                            row[mapping_design_index] = json.dumps(item_to_serialize, ensure_ascii=False)
                        except TypeError:
                            # 无法序列化时保留原始值
                            pass 

                writer.writerow(row)
                
    except Exception as e:
        print(f"写入 CSV 文件 {output_file} 时发生错误: {e}")
        return

# ==============================================================================
# 用户调用接口 (优化模式) - 查找单个最优解
# ==============================================================================

def process_parallel_opt_H(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_opt_H_min_cycles.csv", 
    num_processes: int = None
) -> None:
    """
    H 模式 (High Priority Cycle)：在现有约束下，最小化 Cycles，然后最小化 Transmission。
    """
    _prepare_and_dispatch_optimizer(
        "H 模式 (最小 Cycles 优先)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        num_processes,
        optimization_target='H'
    )
    
def process_parallel_opt_L(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_opt_L_min_transmission.csv", 
    num_processes: int = None
) -> None:
    """
    L 模式 (High Priority Transmission)：在现有约束下，最小化 Transmission，然后最小化 Cycles。
    """
    _prepare_and_dispatch_optimizer(
        "L 模式 (最小 Transmission 优先)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        num_processes,
        optimization_target='L'
    )

def process_parallel_opt_C(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_opt_C_constrained_cycles.csv", 
    num_processes: int = None
) -> None:
    """
    C 模式 (Constrained Resource)：在指定资源约束下，最小化 Cycles，然后最小化 Transmission。
    """
    _prepare_and_dispatch_optimizer(
        "C 模式 (限定资源下的最小 Cycles)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        num_processes,
        optimization_target='C'
    )
    
def process_parallel_opt_U(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    weights: Tuple[float, float], # (W_T, W_C)
    output_file: str = "result_opt_U_weighted_utility.csv", 
    num_processes: int = None
) -> None:
    """
    U 模式 (Utility)：在现有约束下，最小化 Cycles 和 Transmission 的加权效用分数。
    """
    if sum(weights) == 0:
         print("错误: 权重 (W_T, W_C) 的总和不能为零。")
         return
         
    _prepare_and_dispatch_optimizer(
        f"U 模式 (加权效用搜索 W_T={weights[0]}, W_C={weights[1]})",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        num_processes,
        optimization_target='U',
        weights=weights
    )

def process_parallel_opt_R(
    design_schemes: List[List[Scheme]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_opt_R_min_resource.csv", 
    num_processes: int = None
) -> None:
    """
    R 模式 (Min Resource)：在 Cycles/Transmission 约束下，最小化 Resource。
    """
    _prepare_and_dispatch_optimizer(
        "R 模式 (最小 Resource 搜索)",
        design_schemes, 
        resource_limit, 
        resource_min, 
        cycle_limit, 
        transmission_limit, 
        output_file, 
        num_processes,
        optimization_target='R'
    )

