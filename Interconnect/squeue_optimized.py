import random
from copy import deepcopy
import os
import csv
import math
import numpy as np
# 在 squeue_optimized.py 文件的最开始部分，添加以下导入语句
from typing import List, Any, Dict, Union
import glob
import sys
import re
import subprocess
# --------------------------
# 1. 核心工具函数：资源块坐标与跳数计算
# --------------------------
def coords_to_res_id(x, y, mesh_size=4):
    """(x,y)坐标→资源块编号（行优先：(0,0)=0, (0,1)=1, ..., (3,3)=15）"""
    return x * mesh_size + y

def res_id_to_coords(res_id, mesh_size=4):
    """资源块编号→(x,y)坐标"""
    x = res_id // mesh_size
    y = res_id % mesh_size
    return (x, y)

def calculate_xy_hop(res1, res2):
    """计算单个资源块对的XY路由跳数（曼哈顿距离）"""
    (x1, y1) = res_id_to_coords(res1)
    (x2, y2) = res_id_to_coords(res2)
    return abs(x1 - x2) + abs(y1 - y2)

# --------------------------
# 2. 输入解析：提取层-资源映射与传输对
# --------------------------
def parse_input_layout(input_layout):
    """
    解析布局字典，返回每个芯片的层信息：
    - layer_res_map: {层号: [资源块列表]}
    - layer_count: {层号: 资源块数量}
    - transmission_pairs: 连续层传输对
    """
    chip_info = {}
    for chip_id, layout in input_layout.items():
        layer_res_map = {}
        for x in range(4):
            for y in range(4):
                layer = layout[x][y]
                if layer == 0:
                    continue
                res_id = coords_to_res_id(x, y)
                if layer not in layer_res_map:
                    layer_res_map[layer] = []
                layer_res_map[layer].append(res_id)
        
        layer_count = {layer: len(res_list) for layer, res_list in layer_res_map.items()}
        sorted_layers = sorted(layer_res_map.keys())
        transmission_pairs = [(sorted_layers[i], sorted_layers[i+1]) 
                             for i in range(len(sorted_layers)-1)]
        
        chip_info[chip_id] = {
            "layer_res_map": layer_res_map,
            "layer_count": layer_count,
            "transmission_pairs": transmission_pairs,
            "all_layers": set(sorted_layers)  # 新增：保存所有有效层号
        }
    return chip_info

# --------------------------
# 3. 遗传算法类：单芯片布局优化
# --------------------------
class GeneticLayoutOptimizer:
    def __init__(self, chip_id, chip_info, initial_layout):
        """
        初始化优化器
        参数：
            chip_id: 芯片编号
            chip_info: 由parse_input_layout解析的芯片信息
            initial_layout: 当前芯片的初始布局（4x4二维数组）
        """
        self.chip_id = chip_id
        self.layer_count = chip_info["layer_count"]
        self.transmission_pairs = chip_info["transmission_pairs"]
        self.all_res_ids = list(range(16))  # 0-15资源块
        self.initial_layout = initial_layout
        self.all_layers = chip_info["all_layers"]  # 新增：所有有效层号集合

    def create_individual(self):
        """创建有效个体（确保每层资源块数符合约束）"""
        individual = {}
        layer_pool = []
        for layer, count in self.layer_count.items():
            layer_pool.extend([layer] * count)
        layer_pool += [0] * (16 - len(layer_pool))
        random.shuffle(layer_pool)
        for res_id, layer in zip(self.all_res_ids, layer_pool):
            individual[res_id] = layer
        return individual

    def initialize_population(self, pop_size=50):
        """初始化种群"""
        return [self.create_individual() for _ in range(pop_size)]

    def calculate_total_hops(self, individual):
        """计算个体的跳数总和"""
        total_hops = 0
        for (src_layer, dest_layer) in self.transmission_pairs:
            src_res_list = [res for res, layer in individual.items() if layer == src_layer]
            dest_res_list = [res for res, layer in individual.items() if layer == dest_layer]
            for src_res in src_res_list:
                for dest_res in dest_res_list:
                    total_hops += calculate_xy_hop(src_res, dest_res)
        return total_hops

    def fitness(self, individual):
        """适应度函数（跳数越小，适应度越高）"""
        total_hops = self.calculate_total_hops(individual)
        return 1.0 / (total_hops + 1e-6)

    def select_parents(self, population, fitness_scores, k=3):
        """锦标赛选择"""
        parents = []
        pop_size = len(population)
        for _ in range(pop_size // 2):
            candidates = random.sample(list(zip(population, fitness_scores)), k)
            candidates.sort(key=lambda x: x[1], reverse=True)
            parents.append(candidates[0][0])
        return parents

    def crossover(self, parent1, parent2):
        """两点交叉（确保约束）"""
        cross1 = random.randint(0, 7)
        cross2 = random.randint(8, 15)
        offspring = {}

        # 1. 复制parent1的[0, cross1)段
        for res_id in self.all_res_ids[:cross1]:
            offspring[res_id] = parent1[res_id]

        # 初始化层使用计数
        layer_used = {}
        for layer in offspring.values():
            layer_used[layer] = layer_used.get(layer, 0) + 1

        # 2. 处理[cross1, cross2]段（使用parent2的基因）
        for res_id in self.all_res_ids[cross1:cross2+1]:
            candidate_layer = parent2[res_id]
            
            # 关键修复：验证层号是否有效
            if candidate_layer not in self.all_layers and candidate_layer != 0:
                # 如果是无效层号，使用parent1的层号
                offspring[res_id] = parent1[res_id]
                layer_used[parent1[res_id]] = layer_used.get(parent1[res_id], 0) + 1
                continue
                
            # 计算最大允许数量
            max_allowed = self.layer_count.get(candidate_layer, float('inf'))
            
            # 检查是否可以使用parent2的这个层
            if layer_used.get(candidate_layer, 0) < max_allowed:
                offspring[res_id] = candidate_layer
                layer_used[candidate_layer] = layer_used.get(candidate_layer, 0) + 1
            else:
                offspring[res_id] = parent1[res_id]
                layer_used[parent1[res_id]] = layer_used.get(parent1[res_id], 0) + 1

        # 3. 复制parent1的[cross2+1, 15]段
        for res_id in self.all_res_ids[cross2+1:]:
            offspring[res_id] = parent1[res_id]
            layer_used[parent1[res_id]] = layer_used.get(parent1[res_id], 0) + 1

        # 验证约束
        for layer, required in self.layer_count.items():
            if list(offspring.values()).count(layer) != required:
                return self.create_individual()
        return offspring

    def mutate(self, individual, mutation_rate=0.1):
        """变异（确保约束）"""
        mutate_count = int(len(self.all_res_ids) * mutation_rate)
        for _ in range(mutate_count):
            res1, res2 = random.sample(self.all_res_ids, 2)
            layer1 = individual[res1]
            layer2 = individual[res2]

            # 检查层号有效性
            if (layer1 not in self.all_layers and layer1 != 0) or \
               (layer2 not in self.all_layers and layer2 != 0):
                continue  # 跳过无效层号的变异
            
            valid = True
            for layer in [layer1, layer2]:
                if layer in self.layer_count:
                    if list(individual.values()).count(layer) != self.layer_count[layer]:
                        valid = False
                        break
            if valid:
                individual[res1], individual[res2] = layer2, layer1

        # 验证约束
        for layer, required in self.layer_count.items():
            if list(individual.values()).count(layer) != required:
                return self.create_individual()
        return individual

    def optimize(self, pop_size=50, generations=100, mutation_rate=0.1):
        """执行优化，返回最优布局、最小跳数、优化前跳数"""
        # 计算优化前的跳数
        initial_individual = {}
        for res_id in self.all_res_ids:
            x, y = res_id_to_coords(res_id)
            initial_individual[res_id] = self.initial_layout[x][y]
        initial_hops = self.calculate_total_hops(initial_individual)

        population = self.initialize_population(pop_size)
        best_hops = float('inf')
        best_individual = None

        print(f"\n=== 芯片{self.chip_id}优化过程 ===")
        print(f"优化前跳数总和：{initial_hops}")
        for gen in range(generations):
            fitness_scores = [self.fitness(ind) for ind in population]
            current_hops_list = [self.calculate_total_hops(ind) for ind in population]
            current_min = min(current_hops_list)
            current_best_idx = current_hops_list.index(current_min)

            if current_min < best_hops:
                best_hops = current_min
                best_individual = deepcopy(population[current_best_idx])

            # if (gen + 1) % 10 == 0:
            #     print(f"第{gen+1:3d}代 | 当前最小跳数：{current_min:6d} | 全局最小跳数：{best_hops:6d}")

            parents = self.select_parents(population, fitness_scores)
            offspring = []
            while len(offspring) < pop_size - len(parents):
                p1, p2 = random.sample(parents, 2)
                offspring.append(self.crossover(p1, p2))
            offspring = [self.mutate(ind, mutation_rate) for ind in offspring]
            population = parents + offspring

        # 转换为4x4布局
        best_layout = []
        for x in range(4):
            row = []
            for y in range(4):
                res_id = coords_to_res_id(x, y)
                row.append(best_individual[res_id])
            best_layout.append(row)

        print(f"优化完成 | 最小跳数总和：{best_hops} | 跳数减少：{initial_hops - best_hops}")
        return best_layout, best_hops, initial_hops

# --------------------------
# 4. 主函数：优化所有芯片布局
# --------------------------
def optimize_all_chips(input_layout, pop_size=50, generations=100, mutation_rate=0.1):
    """优化输入字典中所有芯片的布局"""
    chip_info = parse_input_layout(input_layout)
    optimized_layouts = {}
    hops_summary = {}

    for chip_id in input_layout.keys():
        optimizer = GeneticLayoutOptimizer(
            chip_id=chip_id,
            chip_info=chip_info[chip_id],
            initial_layout=input_layout[chip_id]
        )
        best_layout, best_hops, initial_hops = optimizer.optimize(
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        optimized_layouts[chip_id] = best_layout
        hops_summary[chip_id] = {
            "优化前跳数": initial_hops,
            "优化后跳数": best_hops,
            "减少跳数": initial_hops - best_hops
        }



    return optimized_layouts, hops_summary




# 读取神经网络和神经网络映射策略，后续通信优化需要
def read_net_mapping_strategy(network_name, param1, param2):
    """
    按行读取文件,将每行内容拆分并转换为数字(尽可能)
    
    参数:
        network_name (str):网络名称,如"Resnet20"
        param1 (int):第一个数字参数,如512
        param2 (int):第二个数字参数,如512
    
    返回:
        list:处理后的内容列表,每个元素为一行的拆分结果(列表)
              其中可转换为数字的元素会被转为int或float,其余保持字符串
        None:若文件不存在或读取失败
    """
    try:
        # 拼接文件路径
        resource = 0
        filename = f"NetWork_{network_name}_{param1}_{param2}_cof.csv"
        file_path = os.path.join("/home/zxf1/master_code/", network_name, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"Error: File does not exist - {file_path}")
            return None
        with open('/home/zxf1/master_code/NetWork_'+network_name+'.csv', newline='', encoding='utf-8') as f:
            net = list(csv.reader(f, delimiter=',', quotechar='"'))
        
        result = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # 去除首尾空白字符(包括换行符)
                cleaned_line = line.strip()
                
                # 按常见分隔符(逗号、空格、制表符)拆分
                # 优先按逗号拆分(适合CSV文件),其次按空白字符
                if ',' in cleaned_line:
                    parts = [p.strip() for p in cleaned_line.split(',')]
                else:
                    parts = cleaned_line.split()
                
                # 尝试将每个部分转换为数字
                processed_parts = []
                for part in parts:
                    # 尝试转换为整数
                    try:
                        processed_parts.append(int(part))
                        continue
                    except ValueError:
                        pass
                    
                    # 尝试转换为浮点数
                    try:
                        processed_parts.append(float(part))
                        continue
                    except ValueError:
                        pass
                    
                    # 无法转换则保留原始字符串
                    processed_parts.append(part)
                resource = resource + processed_parts[2]
                result.append(processed_parts)
        
        print(f"Successfully read and processed the file:{file_path},共{len(result)}行")
        return result,net,resource
    
    except Exception as e:
        print(f"An error occurred while reading the file:{str(e)}")
        return None

#record data transmission path
def calculate_transferpath(map_strategy,net,quantization_bit):
    all_transfer_path = []
    # path_record = []
    for i in range(len(map_strategy)-1):
        all_transmission = int(net[i][0])*int(net[i][1])*int(net[i][2])*quantization_bit

        all_transfer_path.append([all_transmission,map_strategy[i][2],map_strategy[i+1][2]])

    return all_transfer_path



def allocate_chips(mapping_stratepy, num_tile, num_chiplet):
    """
    修正起始层映射状态判断的芯片分配函数：
    拆分到新芯片的起始层(如18层在第三个芯片)不算完全映射
    
    参数:
        mapping_stratepy (list): 每层数据列表(0-based),第三项为资源需求
        num_tile (int): 单个芯片的总资源容量
        num_chiplet (int): 最大可用芯片数量
    
    返回:
        list: 芯片分配详情，包含正确的起始层和结束层映射信息
    """
    chips = []                  # 存储芯片分配结果
    current_chip_id = 0         # 当前芯片ID
    current_start_layer = None  # 当前芯片起始层（1-based）
    current_remaining_tile = num_tile  # 当前芯片剩余资源
    current_layer_idx = 0       # 当前处理的层索引（0-based）
    total_layers = len(mapping_stratepy)
    prev_chip = None            # 上一个芯片信息，用于处理起始层映射
    
    while current_layer_idx < total_layers and current_chip_id < num_chiplet:
        # 获取当前层信息（1-based层号 = 索引 + 1）
        layer_1based = current_layer_idx + 1
        layer_res = int(mapping_stratepy[current_layer_idx][2])  # 第三项为资源需求
        
        # 初始化当前芯片的起始层
        if current_start_layer is None:
            current_start_layer = layer_1based
        
        # 计算起始层映射信息（仅在芯片首次创建时计算）
        if len(chips) <= current_chip_id:  # 尚未初始化当前芯片信息
            # 起始层总资源需求
            start_layer_total = int(mapping_stratepy[current_start_layer - 1][2])
            # 判断起始层是否为上一个芯片的结束层（拆分层）
            if prev_chip and prev_chip['end_layer'] == current_start_layer:
                # 关键修正：起始层是拆分层，即使包含剩余全部资源，也不算完全映射
                start_res = start_layer_total - prev_chip['end_layer_resources']
                start_complete = False  # 拆分层作为起始层，不算完全映射
            else:
                # 起始层是全新层，未被拆分，算完全映射
                start_res = start_layer_total
                start_complete = True
        
        # 尝试分配当前层
        if layer_res <= current_remaining_tile:
            # 完全分配当前层
            current_remaining_tile -= layer_res
            current_layer_idx += 1  # 处理下一层
            
            # 检查是否需要封板当前芯片（资源用尽或处理完所有层）
            if current_remaining_tile == 0 or current_layer_idx == total_layers:
                # 确定结束层信息
                end_layer_1based = current_layer_idx if current_layer_idx == total_layers else current_layer_idx
                # 结束层总资源需求（最后一层特殊处理）
                end_layer_total = int(mapping_stratepy[end_layer_1based - 1][2]) if end_layer_1based <= total_layers else 0
                
                # 添加当前芯片信息
                chips.append({
                    'chip_id': current_chip_id,
                    'start_layer': current_start_layer,
                    'start_layer_complete': start_complete,
                    'start_layer_resources': start_res,
                    'end_layer': end_layer_1based,
                    'end_layer_complete': True,
                    'end_layer_resources': end_layer_total
                })
                
                # 更新状态，准备下一个芯片
                prev_chip = chips[-1]
                current_chip_id += 1
                current_start_layer = end_layer_1based + 1  # 新芯片起始层
                current_remaining_tile = num_tile  # 重置芯片资源
        else:
            # 部分分配当前层（当前芯片资源不足）
            end_layer_1based = layer_1based
            allocated_res = current_remaining_tile  # 当前芯片可分配的资源
            
            # 添加当前芯片信息
            chips.append({
                'chip_id': current_chip_id,
                'start_layer': current_start_layer,
                'start_layer_complete': start_complete,
                'start_layer_resources': start_res,
                'end_layer': end_layer_1based,
                'end_layer_complete': False,
                'end_layer_resources': allocated_res
            })
            
            # 更新状态，准备下一个芯片
            prev_chip = chips[-1]
            current_chip_id += 1
            current_start_layer = end_layer_1based  # 新芯片起始层为当前结束层（拆分层）
            current_remaining_tile = num_tile - (layer_res - allocated_res)  # 新芯片剩余资源
            current_layer_idx += 1  # 处理下一层
    
    return chips


def allocate_chips_new(mapping_stratepy: List[List[Any]], num_tile: int, num_chiplet: int) -> List[Dict[str, Union[int, float, bool]]]:
    """
    基于资源限制和芯片数量，将网络层分配到芯片上，并生成摘要信息。
    
    核心修复: 确保 chip_record 在芯片资源用尽或分配完毕前不会被重复初始化。
    
    参数:
        mapping_stratepy (list): 每层数据列表(0-based),第三项为资源需求。
                                 格式: [[...], [..., 资源需求, ...], ...]
        num_tile (int): 单个芯片的总资源容量(如16)。
        num_chiplet (int): 最大可用芯片数量(如4)。
    
    返回:
        list: 芯片分配详情，包含正确的起始层和结束层映射信息。
    """
    
    chips = [] # 存储芯片分配结果
    current_chip_id = 0 # 当前芯片ID(0-based)
    current_start_layer = 1 # 当前芯片起始层(1-based)
    current_remaining_tile = num_tile # 当前芯片剩余资源
    current_layer_idx = 0 # 当前处理的层索引(0-based)
    total_layers = len(mapping_stratepy)
    prev_chip = None # 上一个芯片信息，用于处理拆分层的延续
    
    # 核心修复变量: 存储当前正在构建的芯片记录，如果为 None 则需要初始化新芯片。
    chip_record = None 
    
    # 追踪变量：记录被部分分配的层，它在所有前续芯片上已经消耗了多少资源
    layer_resources_used_so_far = {} 
    
    # 在分配开始前，预先计算所有层的总资源需求，避免多次查询
    layer_total_resources = [int(layer[2]) for layer in mapping_stratepy]
    
    while current_layer_idx < total_layers and current_chip_id < num_chiplet:
        layer_1based = current_layer_idx + 1
        layer_res_total = layer_total_resources[current_layer_idx] # 该层总共需要的资源
        
        # 1. 确定该层在当前芯片的资源需求(layer_res_to_use)
        res_allocated_on_prev_chips = layer_resources_used_so_far.get(layer_1based, 0)
        layer_res_to_use = layer_res_total - res_allocated_on_prev_chips
            
        # 2. 确定当前芯片的Start Layer属性并初始化记录 (仅在 chip_record 为 None 时执行)
        if chip_record is None:
            # 确定 start_layer_complete 状态
            # 如果起始层与上一个芯片的结束层相同，且上一个芯片的结束映射不完全(即拆分层)
            if prev_chip and prev_chip['end_layer'] == current_start_layer and not prev_chip['end_layer_complete']:
                start_complete = False
            else:
                # 否则，是新的未拆分的层的开始
                start_complete = True
                
            # 初始化当前芯片的记录
            chip_record = {
                'chip_id': current_chip_id,
                'start_layer': current_start_layer,
                'start_layer_complete': start_complete,
                'start_layer_resources': 0, # 待分配后更新为实际使用量
                'end_layer': 0,
                'end_layer_complete': False,
                'end_layer_resources': 0
            }
        
        # 3. 尝试分配当前层到当前芯片
        
        # 实际分配的资源量：取 '层所需剩余资源' 和 '芯片剩余资源' 中的最小值
        allocated_res_on_current_chip = min(layer_res_to_use, current_remaining_tile)
        
        # 核心修复点: 当且仅当当前层是该芯片的起始层时，更新 start_layer_resources
        # 确保它等于该层在当前芯片上实际分配到的资源量
        if chip_record['start_layer'] == layer_1based:
            chip_record['start_layer_resources'] = allocated_res_on_current_chip
            
        # 更新剩余资源
        current_remaining_tile -= allocated_res_on_current_chip
        
        # 更新该层总共被使用的资源(用于下一个芯片的计算)
        layer_resources_used_so_far[layer_1based] = res_allocated_on_prev_chips + allocated_res_on_current_chip
        
        # 4. 判断当前层的映射状态：是否完成？
        is_layer_fully_mapped_on_this_step = (allocated_res_on_current_chip == layer_res_to_use)
        
        
        if is_layer_fully_mapped_on_this_step:
            # 情况A: 当前层在该芯片上**完成**映射 (可能芯片还有剩余资源)

            if current_remaining_tile == 0 or current_layer_idx == total_layers - 1:
                # 芯片资源用尽 或 已经分配完所有层->封板
                chip_record['end_layer'] = layer_1based
                chip_record['end_layer_complete'] = True
                chip_record['end_layer_resources'] = allocated_res_on_current_chip
                
                chips.append(chip_record)
                
                # 准备下一个芯片/层
                prev_chip = chips[-1]
                current_chip_id += 1
                current_start_layer = layer_1based + 1 
                current_remaining_tile = num_tile 
                current_layer_idx += 1 
                chip_record = None # CRITICAL: 重置当前芯片记录
            else:
                # 芯片还有剩余资源，继续下一层
                current_layer_idx += 1 
                
        else:
            # 情况B: 当前层在该芯片上**未完成**映射(芯片资源用尽)->必须封板
            
            # 封板当前芯片
            chip_record['end_layer'] = layer_1based
            chip_record['end_layer_complete'] = False
            chip_record['end_layer_resources'] = allocated_res_on_current_chip
            
            chips.append(chip_record)
            
            # 准备下一个芯片/层
            prev_chip = chips[-1]
            current_chip_id += 1
            current_start_layer = layer_1based # 下一个芯片从当前层开始(延续)
            current_remaining_tile = num_tile # 新芯片重置资源
            chip_record = None # CRITICAL: 重置当前芯片记录
            # layer_idx 不变，下一轮循环将再次处理当前层
            
    return chips


def create_folder(folder_name, path=None):
    """
    在指定路径创建文件夹并进入，若文件夹已存在则直接进入
    
    参数:
        folder_name (str): 要创建的文件夹名称
        path (str, optional): 文件夹所在的路径，默认为当前路径
    
    返回:
        str: 最终进入的文件夹的绝对路径
        None: 若操作失败
    """
    try:
        # 确定目标路径
        if path is None:
            target_path = os.path.join(os.getcwd(), folder_name)
        else:
            # 检查指定路径是否存在
            if not os.path.exists(path):
                print(f"错误：指定路径不存在 - {path}")
                return None
            target_path = os.path.join(path, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(target_path):
            # 创建文件夹
            os.makedirs(target_path, exist_ok=True)  # exist_ok=True 避免多线程等场景的竞争问题
            print(f"已创建文件夹：{target_path}")
        else:
            print(f"文件夹已存在：{target_path}")
        
        
        return os.getcwd()
    
    except PermissionError:
        print(f"error: no permission - {folder_name}")
        return None
    except Exception as e:
        print(f"operation error: {str(e)}")
        return None


def split_transmissions(chip_mappings, transmission_data):
    """
    通用传输数据拆分函数，根据芯片映射关系动态处理片内和片间传输
    
    参数:
        chip_mappings (list): 芯片映射信息列表，每个元素包含:
            - chip_id: 芯片ID
            - start_layer: 起始层(1-based)
            - end_layer: 结束层(1-based)
            - end_layer_complete: 结束层是否完全映射
            - end_layer_resources: 结束层在本芯片的资源数(未完全映射时有效)
        transmission_data (list): 层间传输数据列表，每个元素为 [数据量, 源层资源, 目的层资源]
                                 其中第i个元素对应 (i+1)→(i+2) 层的传输
    
    返回:
        dict: 拆分结果，键为芯片ID，值包含:
            - intra: 片内传输列表，每个元素为 (源层, 目的层, 传输数据)
            - inter: 片间传输列表，每个元素为 (源层, 目的层, 目标芯片ID, 传输数据)
    """
    # 初始化结果字典
    result = {
        chip['chip_id']: {
            'intra': [],
            'inter': []
        } for chip in chip_mappings
    }
    
    # 构建层与芯片的映射关系：记录每个层属于哪些芯片
    layer_chips = {}
    for chip in chip_mappings:
        # 处理起始层到结束层-1（这些层完全属于当前芯片）
        for layer in range(chip['start_layer'], chip['end_layer']):
            if layer not in layer_chips:
                layer_chips[layer] = []
            layer_chips[layer].append(chip['chip_id'])
        
        # 处理结束层（可能跨芯片）
        end_layer = chip['end_layer']
        if end_layer not in layer_chips:
            layer_chips[end_layer] = []
        layer_chips[end_layer].append(chip['chip_id'])
    
    # 处理每一项传输数据
    for trans_idx, trans in enumerate(transmission_data):
        src_layer = trans_idx + 1    # 源层(1-based)
        dest_layer = trans_idx + 2   # 目的层(1-based)
        data_size, src_total_res, dest_total_res = trans
        
        # 获取源层和目的层所在的芯片
        src_possible_chips = layer_chips.get(src_layer, [])
        dest_possible_chips = layer_chips.get(dest_layer, [])
        
        # 处理源层在各个芯片上的传输
        for src_chip in src_possible_chips:
            # 获取源层在当前芯片的资源比例
            src_chip_info = next(c for c in chip_mappings if c['chip_id'] == src_chip)
            if src_layer == src_chip_info['end_layer'] and not src_chip_info['end_layer_complete']:
                src_res = src_chip_info['end_layer_resources']
                real_data = data_size*(src_res/dest_total_res)
            else:
                if src_layer == src_chip_info['start_layer'] and not src_chip_info['start_layer_complete']:
                    src_res = src_chip_info['start_layer_resources']
                    real_data = data_size*(src_res/dest_total_res)
                else:
                    src_res = src_total_res  # 完全映射的层使用全部资源
                    real_data = data_size
            
            # 处理目的层在各个芯片上的传输
            for dest_chip in dest_possible_chips:
                # 获取目的层在当前芯片的资源比例
                dest_chip_info = next(c for c in chip_mappings if c['chip_id'] == dest_chip)
                if dest_layer == dest_chip_info['end_layer'] and not dest_chip_info['end_layer_complete']:
                    dest_res = dest_chip_info['end_layer_resources']
                else:
                    if dest_layer == dest_chip_info['start_layer'] and not dest_chip_info['start_layer_complete']:
                        dest_res = dest_chip_info['start_layer_resources']
                    else:
                        dest_res = dest_total_res  # 完全映射的层使用全部资源
                
                # 构建传输数据
                trans_data = [real_data, src_res, dest_res]
                
                # 判断是片内还是片间传输
                if src_chip == dest_chip:
                    # 片内传输
                    result[src_chip]['intra'].append((src_layer, dest_layer, trans_data))
                else:
                    # 片间传输（从源芯片视角记录）
                    result[src_chip]['inter'].append((src_layer, dest_layer, dest_chip, trans_data))
    
    return result


# Complete the file for data transfer

def Sequential_mapping(transmission_data):
    """
    1. Allocate independent on-chip resource blocks (ID 0-15) for each chip (fixed 16 resources per chip)
    2. Generate intra-chip (NoC) transmission records: Format [[src_resource_list], [dest_resource_list], data_volume]
    3. Generate inter-chip (NoP) transmission records: Format [[src_chip_list], [dest_chip_list], [total_data_volume]]
    
    Args:
        transmission_data (dict): Input chip transmission distribution data, including 'intra' (intra-chip) and 'inter' (inter-chip)
        
    Returns:
        tuple: (chip_layer_resource, noc_records, nop_records)
            - chip_layer_resource: Dict of layer-resource block mapping for each chip
            - noc_records: Dict of intra-chip transmission records (grouped by chip)
            - nop_records: List of inter-chip transmission records
    """
    # --------------------------
    # Step 1: Allocate on-chip resource blocks for each chip (independent ID 0-15)
    # --------------------------
    chip_layer_resource = {}  # Structure: {chip_id: {layer_num: [resource_block_list]}}
    total_resources_per_chip = 16  # Fixed 16 resource blocks per chip

    for chip_id in transmission_data:
        current_res_counter = 0  # Resource block ID starts from 0
        chip_layer_resource[chip_id] = {}
        # Collect all layers involved in current chip (deduplicate and sort by layer number)
        layers = set()
        # Extract layers from intra-chip transmission (source layer + destination layer)
        for (src_layer, dest_layer, _) in transmission_data[chip_id]['intra']:
            layers.add(src_layer)
            layers.add(dest_layer)
        # Extract local layers from inter-chip transmission (only source layer; dest layer is in other chips)
        for (src_layer, _, _, _) in transmission_data[chip_id]['inter']:
            layers.add(src_layer)
        # Sort layers by number to ensure continuous resource allocation
        sorted_layers = sorted(layers)

        for layer in sorted_layers:
            # Get resource demand of current layer (extract from transmission data)
            resource_demand = None
            # 1. Priority: Extract from intra-chip transmission (source or destination layer)
            for (s, d, data) in transmission_data[chip_id]['intra']:
                if s == layer:
                    resource_demand = data[1]
                    break
                if d == layer:
                    resource_demand = data[2]
                    break
            # 2. If not found in intra-chip, extract from inter-chip transmission (only source layer)
            if resource_demand is None:
                for (s, _, _, data) in transmission_data[chip_id]['inter']:
                    if s == layer:
                        resource_demand = data[1]
                        break
            resource_demand = int(resource_demand)  # Ensure resource demand is integer

            # Allocate resource blocks and record (generate continuous resource block list)
            start_res = current_res_counter
            end_res = current_res_counter + resource_demand - 1
            chip_layer_resource[chip_id][layer] = list(range(start_res, end_res + 1))
            # Update resource counter (ensure no exceed 16 resources limit)
            current_res_counter += resource_demand
            if current_res_counter > total_resources_per_chip:
                raise ValueError(f"Error: Insufficient resources for chip {chip_id}! Allocated {current_res_counter}, max supported 16")

    # --------------------------
    # Step 2: Generate intra-chip transmission records (NoC)
    # --------------------------
    noc_records = {}  # Structure: {chip_id: [intra_chip_transmission_records]}
    for chip_id in transmission_data:
        noc_records[chip_id] = []
        # Traverse all intra-chip transmission data of current chip
        for (src_layer, dest_layer, data) in transmission_data[chip_id]['intra']:
            data_volume = data[0]
            # Get resource block lists of source and destination layers
            src_res_list = chip_layer_resource[chip_id][src_layer]
            dest_res_list = chip_layer_resource[chip_id][dest_layer]
            # Generate record in specified format
            noc_record = [src_res_list, dest_res_list, data_volume]
            noc_records[chip_id].append(noc_record)

    # --------------------------
    # Step 3: Generate inter-chip transmission records (NoP)
    # --------------------------
    nop_records = []  # List to store inter-chip transmission records
    # First accumulate total data volume between the same pair of chips
    inter_total_map = {}  # Temporary storage: {(src_chip_id, dest_chip_id): total_data_volume}
    for src_chip in transmission_data:
        for (_, _, dest_chip, data) in transmission_data[src_chip]['inter']:
            single_data_volume = data[0]
            key = (src_chip, dest_chip)
            # Accumulate data volume (preserve original precision for float)
            if key in inter_total_map:
                inter_total_map[key] += single_data_volume
            else:
                inter_total_map[key] = single_data_volume
    # Convert to final records in specified format
    for (src_chip, dest_chip), total_data_volume in inter_total_map.items():
        nop_record = [[src_chip], [dest_chip], total_data_volume]
        nop_records.append(nop_record)
    
    create_folder('to_Interconnect')

    create_folder('chiplet_perlayer_resource','./to_Interconnect')

    output_dir = './to_Interconnect/chiplet_perlayer_resource'

    for chip_id in sorted(chip_layer_resource.keys()):
        # 获取当前芯片的层-资源块映射
        layer_resources = chip_layer_resource[chip_id]
        # 定义CSV文件名：Chiplet + 芯片ID
        filename = f"Chiplet{chip_id}.csv"
        
        file_path = os.path.join(output_dir,filename)
        # 手动写入CSV文件，避免内置csv模块的格式问题
        with open(file_path, 'w', encoding='utf-8') as file:
            # 写入表头
            file.write("Layer,Resource Blocks\n")
            
            # 按层号升序写入数据（确保层顺序正确）
            for layer in sorted(layer_resources.keys()):
                # 获取当前层的资源块列表
                resources = layer_resources[layer]
                # 转换为字符串格式（保持[1, 2]样式）
                resources_str = str(resources)
                # 写入一行数据（层号,资源块列表）
                file.write(f"{layer},{resources_str}\n")
    

    return chip_layer_resource, noc_records, nop_records

#Generate trace file

def generate_traces_noc(bus_width, netname, noc_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    Interconnect_path = '/home/zxf1/master_code/Interconnect'                                    # 根目录
    create_folder(netname + '_NoC_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
    file_path = Interconnect_path + '/' + netname + '_NoC_traces'           # trace 总目录

    # ---------------- 按 Chiplet 遍历 ----------------
    for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
        create_folder('Chiplet_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
        chiplet_dir_name = file_path + '/Chiplet_' + str(chip_id)           # 当前 Chiplet 目录

        # ---------------- 按层遍历 ----------------
        for i in range(len(noc_records[chip_id])):
            # 初始化：trace 第一行占位，后续会删除
            trace = np.array([[0, 0, 0]])
            timestamp = 1                                                     # 时间戳从 1 开始

            # 计算本层需要生成的 packet 数量
            num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
            num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

            # 提取源/目的 tile 区间
            src_tile_begin = noc_records[chip_id][i][0][0]
            src_tile_end   = noc_records[chip_id][i][0][-1]
            dest_tile_begin = noc_records[chip_id][i][1][0]
            dest_tile_end   = noc_records[chip_id][i][1][-1]

            # ---------------- 三重循环生成 trace ----------------
            for pack_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in range(dest_tile_begin, dest_tile_end + 1):
                    for src_tile_idx in range(src_tile_begin, src_tile_end + 1):
                        # 追加一行：源tile，目的tile，时间戳
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                    # 同目的不同源之间时间戳+1（除最后一个目的）
                    if dest_tile_idx != dest_tile_end:
                        timestamp += 1
                # 完成一个 packet 后时间戳再+1
                timestamp += 1

            # ---------------- 文件写出 ----------------
            filename = 'trace_file_layer_' + str(i) + '.txt'
            trace = np.delete(trace, 0, 0)                      # 删除初始占位行
            os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
            np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
            # 回到顶层，准备下一层
            os.chdir("..")
            os.chdir("..")
            os.chdir("..")


def generate_traces_noc_GA(bus_width, netname, noc_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    create_folder('/home/zxf1/master_code/Genetic_A')
    Interconnect_path = '/home/zxf1/master_code/Genetic_A'                                    # 根目录
    create_folder(netname + '_NoC_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
    file_path = Interconnect_path + '/' + netname + '_NoC_traces'           # trace 总目录

    # ---------------- 按 Chiplet 遍历 ----------------
    for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
        create_folder('Chiplet_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
        chiplet_dir_name = file_path + '/Chiplet_' + str(chip_id)           # 当前 Chiplet 目录

        # ---------------- 按层遍历 ----------------
        for i in range(len(noc_records[chip_id])):
            # 初始化：trace 第一行占位，后续会删除
            trace = np.array([[0, 0, 0]])
            timestamp = 1                                                     # 时间戳从 1 开始

            # 计算本层需要生成的 packet 数量
            num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
            num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

            # 提取源/目的 tile 区间
            src_tile = noc_records[chip_id][i][0]
            # src_tile_end   = noc_records[chip_id][i][0][-1]
            dest_tile = noc_records[chip_id][i][1]
            dest_tile_end   = noc_records[chip_id][i][1][-1]

            # ---------------- 三重循环生成 trace ----------------
            for pack_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in dest_tile:
                    for src_tile_idx in src_tile:
                        # 追加一行：源tile，目的tile，时间戳
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                    # 同目的不同源之间时间戳+1（除最后一个目的）
                    if dest_tile_idx != dest_tile_end:
                        timestamp += 1
                # 完成一个 packet 后时间戳再+1
                timestamp += 1

            # ---------------- 文件写出 ----------------
            filename = 'trace_file_layer_' + str(i) + '.txt'
            trace = np.delete(trace, 0, 0)                      # 删除初始占位行
            os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
            np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
            # 回到顶层，准备下一层
            os.chdir("..")
            os.chdir("..")
            os.chdir("..")


# def generate_traces_noc_ours(bus_width, netname, noc_records, scale):
#     """
#     为片上网络(NoC)生成通信 trace 文件
#     每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
#     """
#     # ---------------- 目录准备 ----------------
#     create_folder('Ours')
#     Interconnect_path = './Ours'                                    # 根目录
#     create_folder(netname + '_NoC_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
#     file_path = Interconnect_path + '/' + netname + '_NoC_traces'           # trace 总目录

#     # ---------------- 按 Chiplet 遍历 ----------------
#     for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
#         create_folder('Chiplet_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
#         chiplet_dir_name = file_path + '/Chiplet_' + str(chip_id)           # 当前 Chiplet 目录

#         # ---------------- 按层遍历 ----------------
#         for i in range(len(noc_records[chip_id])):
#             # 初始化：trace 第一行占位，后续会删除
#             trace = np.array([[0, 0, 0]])
#             timestamp = 1                                                     # 时间戳从 1 开始

#             # 计算本层需要生成的 packet 数量
#             num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
#             num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

#             # 提取源/目的 tile 区间
         
#             src_tile = noc_records[chip_id][i][0]
#             # src_tile_end   = noc_records[chip_id][i][0][-1]
#             dest_tile = noc_records[chip_id][i][1]
#             dest_tile_end   = noc_records[chip_id][i][1][-1]

#             # ---------------- 三重循环生成 trace ----------------
#             for pack_idx in range(0, num_packets_this_layer):
#                 for dest_tile_idx in dest_tile:
#                     for src_tile_idx in src_tile:
#                         # 追加一行：源tile，目的tile，时间戳
#                         trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

#                     # 同目的不同源之间时间戳+1（除最后一个目的）
#                     if dest_tile_idx != dest_tile_end:
#                         timestamp += 1
#                 # 完成一个 packet 后时间戳再+1
#                 timestamp += 1

#             # ---------------- 文件写出 ----------------
#             filename = 'trace_file_layer_' + str(i) + '.txt'
#             trace = np.delete(trace, 0, 0)                      # 删除初始占位行
#             os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
#             np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
#             # 回到顶层，准备下一层
#             os.chdir("..")
#             os.chdir("..")
#             os.chdir("..")



def convert_to_mesh_layout(input_lists,mesh_size):
    """
    将输入的列表转换为4x4网格布局的字典
    
    参数:
        input_lists: 包含3个子列表的列表，每个子列表对应一个芯片的资源块
        
    返回:
        字典，键为芯片编号(0,1,2)，值为4x4的二维数组
    """ 
    total_elements = mesh_size * mesh_size  # 16个元素
    result = {}
    
    for chip_id, layer_list in enumerate(input_lists):
        # 确保列表长度为16，不足则用0填充
        padded_list = layer_list.copy()
        if len(padded_list) < total_elements:
            padded_list += [0] * (total_elements - len(padded_list))
        
        # 转换为4x4二维数组（每4个元素一行）
        mesh_layout = []
        for i in range(mesh_size):
            start_idx = i * mesh_size
            end_idx = start_idx + mesh_size
            row = padded_list[start_idx:end_idx]
            mesh_layout.append(row)
        
        result[chip_id] = mesh_layout
    
    return result


def block_mapping(mapping,mesh):


    all_chip = []
    chip = []
    for i in range(len(mapping)):
        for resourece in range(mapping[i][2]):
            chip.append(i+1)

    num_tile = mesh*mesh

    num_chip = math.ceil(len(chip)/num_tile)
    
    for j in range(num_chip):
        if j == num_chip - 1:
            all_chip.append(chip[j*num_tile:])

        else:
            all_chip.append(chip[j*num_tile:(j+1)*num_tile])
    
    input_layout = convert_to_mesh_layout(all_chip,mesh)
        

    return input_layout, all_chip


# GA
def GA_mapping(layer_resource_layout, transmission_data):
    """
    生成完整传输记录：
    - 片内：保持原格式（资源块级传输）
    - 片间：将芯片视为单块资源，格式为[[源芯片编号列表], [目的芯片编号列表], 总传输量]
    
    参数:
        layer_resource_layout (dict): 第一份数据（层-资源块布局）
        transmission_data (dict): 第三份数据（含intra/inter传输信息）
    
    返回:
        dict: {
            'intra_records': 片内记录（资源块级）,
            'inter_records': 片间记录（芯片级，传输量求和）
        }
    """
    # --------------------------
    # 步骤1：生成片内（intra）记录（资源块级，保持原逻辑）
    # --------------------------
    # 先预生成「芯片→层→资源块列表」映射（片内记录依赖）
    chip_layer_to_res = {}
    for chip_id, layout in layer_resource_layout.items():
        layer_to_res = {}
        for x in range(4):
            for y in range(4):
                layer = layout[x][y]
                res_id = x * 4 + y  # 资源块编号（0-15）
                if layer not in layer_to_res:
                    layer_to_res[layer] = []
                layer_to_res[layer].append(res_id)
        chip_layer_to_res[chip_id] = layer_to_res

    # 生成片内记录
    intra_records = {}
    for chip_id in transmission_data.keys():
        intra_list = transmission_data[chip_id].get('intra', [])
        current_records = []
        for intra in intra_list:
            src_layer = intra[0]
            dest_layer = intra[1]
            data_volume = intra[2][0]
            # 提取源/目的层的资源块列表
            src_res = chip_layer_to_res[chip_id].get(src_layer, [])
            dest_res = chip_layer_to_res[chip_id].get(dest_layer, [])
            current_records.append([src_res, dest_res, data_volume])
        intra_records[chip_id] = current_records

    # --------------------------
    # 步骤2：生成片间（inter）记录（芯片级，传输量求和）
    # --------------------------
    # 1. 先按「源芯片-目标芯片」分组，累加传输量
    inter_sum_map = {}  # 键：(源芯片ID, 目标芯片ID)，值：总传输量
    for src_chip_id in transmission_data.keys():
        inter_list = transmission_data[src_chip_id].get('inter', [])
        for inter in inter_list:
            dest_chip_id = inter[2]  # 目标芯片ID（inter的第三个元素）
            data_volume = inter[3][0]  # 当前inter的传输量（inter第四个元素的第一个值）
            # 按「源-目标」芯片对分组求和
            key = (src_chip_id, dest_chip_id)
            if key not in inter_sum_map:
                inter_sum_map[key] = 0.0
            inter_sum_map[key] += data_volume

    # 2. 转换为用户要求的片间记录格式：[[源芯片列表], [目的芯片列表], 总传输量]
    inter_records = []
    for (src_chip, dest_chip), total_volume in inter_sum_map.items():
        # 芯片视为单块资源，源/目的均为单元素列表（如[0]、[1]）
        inter_record = [[src_chip], [dest_chip], total_volume]
        inter_records.append(inter_record)

    return {
        'intra_records': intra_records,  # 片内：资源块级传输
        'inter_records': inter_records   # 片间：芯片级传输（求和后）
    }



def process_network_traces(
    network_name: str,
    mapping_mode: str,  # "Ours", "Sqm", or "GA"
    measurement_method: str,  # "NoC" or "NoP"
    mesh: int
) -> Dict[str, List[str]]:
    """
    处理网络跟踪数据的函数 (从您提供的代码中提取并优化)
    
    参数:
        network_name: 网络名称字符串
        mapping_mode: 映射模式 ("Ours", "Sqm", "GA")
        measurement_method: 测量方法 ("NoC", "NoP")
        mesh: NoC Mesh 的 k 值 (例如 4x4 mesh 的 k=4)
    
    返回:
        一个空字典 (为了保持原函数签名，实际结果已写入CSV)
    """
    total_area = 0.0
    total_latency = 0.0
    total_power = 0.0
    
    # 1. 构建基础路径
    if mapping_mode == 'Ours':
        base_dir = "/home/zxf1/master_code/Ours/"
    elif mapping_mode == "GA":
        # 注意: 这里的 os.chdir("/home/zxf1/master_code/Genetic_A/") 会改变主程序的当前工作目录
        # 如果需要，应该在调用 Booksim 前再执行一次
        base_dir = "/home/zxf1/master_code/Genetic_A/"
    else: # 默认为 "Sqm" 或其他模式
        base_dir = "/home/zxf1/master_code/Interconnect/"
        
    base_path = os.path.join(
        base_dir, 
        f"{network_name}_{measurement_method}_traces"
    )
        
    # 检查路径是否存在
    if not os.path.exists(base_path):
        print(f"错误: 路径不存在: {base_path}")
        return {} # 遇到错误提前返回
    
    print(f"\n--- 正在处理跟踪文件: {base_path} ---")
    
    # 3. 准备输出目录
    logs_dir = os.path.join(base_path, 'logs')
    create_folder('logs', base_path)
    
    # 初始化总结果文件
    latency_csv_path = os.path.join(logs_dir, 'booksim_latency.csv')
    power_csv_path = os.path.join(logs_dir, 'booksim_power.csv')
    area_csv_path = os.path.join(logs_dir, 'booksim_area.csv')
    
    # 确保总结果文件清空或开始写入
    # 注意: 原始代码中 power 和 area 写入了同一个文件 'booksim_area.csv'，这里进行了修正
    # 并且使用 'w' 模式清空旧数据
    with open(latency_csv_path, 'w') as f: f.write(f"Trace,Latency(cycles)\n")
    with open(power_csv_path, 'w') as f: f.write(f"Metric,Value,Unit\n")
    with open(area_csv_path, 'w') as f: f.write(f"Metric,Value,Unit\n")


    # 2. 获取所有子文件夹
    # 遍历base_path下的所有文件夹
    for fold_idx, folder_name in enumerate(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder_name)
        
        if not os.path.isdir(folder_path):
            continue # 跳过文件

        # 3. 遍历文件夹中的所有txt文件
        files = glob.glob(os.path.join(folder_path, '*txt'))
        print(f"  - 芯片 {fold_idx} ({folder_name}): 找到 {len(files)} 个跟踪文件.")

        # 配置文件的路径
        config_file_name = f"chiplet_{fold_idx}_mesh_config"
        config_file_path = os.path.join(logs_dir, config_file_name)
        
        # 4. 准备 Booksim 配置
        try:
            with open('/home/zxf1/master_code/Genetic_A/mesh_config_trace_based', 'r') as fp, \
                 open(config_file_path, 'w') as outfile:
                for line in fp:
                    line = line.strip()
                    # 匹配并修改 k= 值
                    matchobj = re.match(r'^k=', line)
                    if matchobj:
                        line = f'k={mesh};'
                    outfile.write(line + '\n')
        except FileNotFoundError:
            print("错误: 找不到 Booksim 配置文件 'mesh_config_trace_based'。请确保其在当前工作目录。")
            return {}
        
        # 5. 运行 Booksim 模拟
        for file_idx, file in enumerate(files):
            # Booksim 日志文件路径
            log_file = os.path.join(logs_dir, f"{folder_name}_layer_{file_idx}.log")
            
            # 将跟踪文件复制到 Booksim 预期位置 (假设 Booksim 在 base_dir 运行且需要 trace_file.txt)
            trace_file_target = os.path.join(base_dir, 'trace_file.txt')
            try:
                 # 使用 shutil.copy 而非 os.system, 更安全
                import shutil
                shutil.copy(file, trace_file_target)
            except Exception as e:
                print(f"复制文件失败: {e}")
                
            
            # 运行 Booksim 命令
            booksim_command = f'./booksim {config_file_path} > {log_file}'
            
            # 切换到 Booksim 可执行文件所在的目录 (假设是 base_dir)
            original_cwd = os.getcwd()
            try:
                os.chdir(base_dir)
                print(f"    -> 模拟 Layer {file_idx}, 运行命令: {booksim_command}")
                # 使用 subprocess.run 替代 os.system，更安全且可捕获返回值
                subprocess.run(booksim_command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"Booksim 运行失败 (命令: {booksim_command}): {e}")
                os.chdir(original_cwd)
                continue
            except FileNotFoundError:
                print(f"错误: 找不到 Booksim 可执行文件 './booksim'。请确保其在 {base_dir} 目录。")
                os.chdir(original_cwd)
                return {}
            finally:
                os.chdir(original_cwd) # 切换回原始目录
            
            
            # 6. 解析结果 (从日志文件中提取 Latency, Power, Area)
            
            def safe_grep_and_extract(log_path, pattern, index):
                """使用 grep/awk 安全提取值"""
                try:
                    # 使用 subprocess.run 替代 os.popen，更现代和安全
                    command = f'grep "{pattern}" {log_path} | tail -1 | awk \'{{print ${index}}}\''
                    result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
                    return result.stdout.strip()
                except Exception:
                    return None

            latency_str = safe_grep_and_extract(log_file, "Trace is finished in", 5)
            power_str = safe_grep_and_extract(log_file, "Total Power", 4)
            area_str = safe_grep_and_extract(log_file, "Total Area", 4)

            try:
                latency = int(latency_str) if latency_str else 0
                power = float(power_str) if power_str else 0.0
                area = float(area_str) if area_str else 0.0
            except ValueError:
                print(f"警告: 无法解析 Booksim 结果, Log: {log_file}")
                latency, power, area = 0, 0.0, 0.0


            total_latency += latency
            total_power += power
            total_area += area
            
            # 写入单层延迟
            with open(latency_csv_path, 'a') as outfile_latency:
                outfile_latency.write(f"{folder_name}_layer_{file_idx},{latency}\n")
            
    # 7. 写入总计结果
    with open(latency_csv_path, 'a') as outfile_latency:
        # 原始代码使用了 1e-9 的缩放因子，表示将周期转换为秒
        outfile_latency.write(f"Total NoC latency is\t{total_latency * 1e-9:.6f}\ts\n")

    with open(power_csv_path, 'a') as outfile_power:
        outfile_power.write(f"Total NoC power is\t{total_power:.4f}\tmW\n")

    with open(area_csv_path, 'a') as outfile_area:
        outfile_area.write(f"Total NoC area is\t{total_area:.4f}\tum^2\n")

    print(f"\n--- 模拟结果汇总 ---")
    print(f"总延迟: {total_latency * 1e-9:.6e} s")
    print(f"总功耗: {total_power:.4f} mW")
    print(f"总面积: {total_area:.4f} um^2")
    print(f"详细结果已保存到 {logs_dir} 目录下。")
    
    return {}

def generate_traces_nop(bus_width, netname, nop_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    create_folder('/home/zxf1/master_code/Interconnect')
    Interconnect_path = '/home/zxf1/master_code/Interconnect'                                    # 根目录
    create_folder(netname + '_NoP_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
    file_path = Interconnect_path + '/' + netname + '_NoP_traces'           # trace 总目录

    # ---------------- 按 Chiplet 遍历 ----------------
    # for chip_id in sorted(nop_records.keys()):                              # 保证 Chiplet 顺序
    create_folder('Chiplet_NoP', file_path)                 # 创建 ./Chiplet_<id>/
    chiplet_dir_name = file_path + '/Chiplet_NoP'          # 当前 Chiplet 目录

    # ---------------- 按层遍历 ----------------
    for i in range(len(nop_records)):
        # 初始化：trace 第一行占位，后续会删除
        trace = np.array([[0, 0, 0]])
        timestamp = 1                                                     # 时间戳从 1 开始

        # 计算本层需要生成的 packet 数量
        num_packets_this_layer = math.ceil(nop_records[i][2] / bus_width)
        num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

        # 提取源/目的 tile 区间
        src_tile = nop_records[i][0]
        # src_tile_end   = noc_records[chip_id][i][0][-1]
        dest_tile = nop_records[i][1]
        dest_tile_end   = nop_records[i][1][-1]

        # ---------------- 三重循环生成 trace ----------------
        for pack_idx in range(0, num_packets_this_layer):
            for dest_tile_idx in dest_tile:
                for src_tile_idx in src_tile:
                    # 追加一行：源tile，目的tile，时间戳
                    trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                # 同目的不同源之间时间戳+1（除最后一个目的）
                if dest_tile_idx != dest_tile_end:
                    timestamp += 1
            # 完成一个 packet 后时间戳再+1
            timestamp += 1

        # ---------------- 文件写出 ----------------
        filename = 'NoP_trace_file_layer_' + str(i) + '.txt'
        trace = np.delete(trace, 0, 0)                      # 删除初始占位行
        os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
        np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
        # 回到顶层，准备下一层
        os.chdir("..")
        os.chdir("..")
        os.chdir("..")