import random
import numpy as np
from copy import deepcopy

class LayerMappingOptimizer:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.tile_count = grid_size * grid_size
        # 预计算曼哈顿距离矩阵
        self.manhattan_matrix = self._precompute_manhattan_matrix()
        
    def _precompute_manhattan_matrix(self):
        """预计算所有tile对之间的曼哈顿距离"""
        matrix = np.zeros((self.tile_count, self.tile_count), dtype=int)
        for i in range(self.tile_count):
            x1, y1 = i // self.grid_size, i % self.grid_size
            for j in range(self.tile_count):
                x2, y2 = j // self.grid_size, j % self.grid_size
                matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)
        return matrix
    
    def create_layer_mapping_from_chromosome(self, chromosome, original_layer_to_tiles):
        """
        从染色体创建层到tile的映射
        
        参数:
            chromosome: 染色体，是一个tile排列，如 [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            original_layer_to_tiles: 原始层到tile的映射
        
        返回:
            new_mapping: 新的层到tile的映射 {layer_id: [tile_ids]}
        """
        new_mapping = {}
        
        # 获取所有层和每层需要的tile数量
        layers = sorted(original_layer_to_tiles.keys())
        tile_counts = {layer: len(original_layer_to_tiles[layer]) for layer in layers}
        
        # 按照染色体顺序和每层的tile数量分配tile
        idx = 0
        for layer in layers:
            count = tile_counts[layer]
            new_mapping[layer] = chromosome[idx:idx+count]
            idx += count
        
        return new_mapping
    
    def calculate_cost(self, chromosome, noc_records, chiplet_id, original_layer_to_tiles):
        """
        计算给定染色体的总代价
        
        参数:
            chromosome: 染色体，tile排列
            noc_records: NoC记录
            chiplet_id: chiplet ID
            original_layer_to_tiles: 原始层到tile的映射
        
        返回:
            总代价
        """
        # 从染色体创建层映射
        layer_mapping = self.create_layer_mapping_from_chromosome(chromosome, original_layer_to_tiles)
        
        # 验证映射：确保没有重复的tile
        all_tiles = []
        for tiles in layer_mapping.values():
            all_tiles.extend(tiles)
        if len(set(all_tiles)) != len(all_tiles):
            return float('inf')  # 无效映射，返回极大代价
        
        total_cost = 0
        
        if chiplet_id not in noc_records:
            return total_cost
        
        records = noc_records[chiplet_id]
        
        for record in records:
            src_tiles, dst_tiles, data_volume = record
            
            # 找出这些tile属于哪些层
            src_layer = None
            dst_layer = None
            
            # 查找源层和目的层
            for layer_id, tiles in original_layer_to_tiles.items():
                # 检查src_tiles是否是该层的tile
                if set(src_tiles) == set(tiles):
                    src_layer = layer_id
                # 检查dst_tiles是否是该层的tile
                if set(dst_tiles) == set(tiles):
                    dst_layer = layer_id
            
            if src_layer is None or dst_layer is None:
                # 如果找不到对应的层，跳过这条记录
                continue
            
            # 获取新的tile映射
            new_src_tiles = layer_mapping[src_layer]
            new_dst_tiles = layer_mapping[dst_layer]
            
            # 计算总跳数：源层每个tile到目的层每个tile的距离总和
            total_hops = 0
            for src_tile in new_src_tiles:
                for dst_tile in new_dst_tiles:
                    total_hops += self.manhattan_matrix[src_tile][dst_tile]
            
            total_cost += total_hops * data_volume
        
        return total_cost
    
    def create_initial_population(self, pop_size, original_layer_to_tiles):
        """创建初始种群 - 确保每个tile只被一个层使用"""
        population = []
        
        # 获取所有tile
        all_tiles = list(range(self.tile_count))
        
        for _ in range(pop_size):
            # 随机打乱所有tile
            chromosome = all_tiles.copy()
            random.shuffle(chromosome)
            
            # 确保染色体有效（每个tile只出现一次）
            if len(set(chromosome)) == self.tile_count:
                population.append(chromosome)
            else:
                # 如果无效，重新生成
                chromosome = all_tiles.copy()
                random.shuffle(chromosome)
                population.append(chromosome)
        
        return population
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """锦标赛选择"""
        selected = []
        pop_size = len(population)
        
        for _ in range(pop_size):
            # 随机选择参赛者
            tournament_indices = random.sample(range(pop_size), tournament_size)
            # 找到适应度最高的（代价最低的）
            best_idx = tournament_indices[0]
            for idx in tournament_indices[1:]:
                if fitness_scores[idx] > fitness_scores[best_idx]:
                    best_idx = idx
            selected.append(population[best_idx])
        
        return selected
    
    def ordered_crossover(self, parent1, parent2):
        """有序交叉 - 确保每个tile只出现一次"""
        size = len(parent1)
        
        # 选择两个交叉点
        point1, point2 = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        
        # 复制中间段
        child[point1:point2] = parent1[point1:point2]
        
        # 填充剩余部分
        idx = point2
        for i in range(size):
            pos = (point2 + i) % size
            gene = parent2[pos]
            if gene not in child:
                child[idx % size] = gene
                idx += 1
        
        return child
    
    def swap_mutation(self, chromosome):
        """交换变异 - 随机交换两个tile"""
        size = len(chromosome)
        i, j = random.sample(range(size), 2)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome
    
    def genetic_algorithm_optimize(self, noc_records, chiplet_id, original_layer_to_tiles,
                                   pop_size=20, generations=50, crossover_rate=0.8, mutation_rate=0.1):
        """遗传算法优化"""
        
        # 创建初始种群
        population = self.create_initial_population(pop_size, original_layer_to_tiles)
        
        # 评估初始种群
        costs = []
        for chrom in population:
            cost = self.calculate_cost(chrom, noc_records, chiplet_id, original_layer_to_tiles)
            costs.append(cost)
        
        best_idx = np.argmin(costs)
        best_chromosome = deepcopy(population[best_idx])
        best_cost = costs[best_idx]
        
        history = [{
            'generation': 0,
            'best_cost': best_cost,
            'avg_cost': np.mean(costs),
            'worst_cost': np.max(costs)
        }]
        
        # 进化循环
        for gen in range(1, generations + 1):
            # 计算适应度（负代价）
            fitness_scores = [-cost for cost in costs]
            
            # 选择
            selected = self.tournament_selection(population, fitness_scores)
            
            # 交叉和变异
            offspring = []
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size:
                    parent1, parent2 = selected[i], selected[i+1]
                    
                    if random.random() < crossover_rate:
                        child1 = self.ordered_crossover(parent1, parent2)
                        child2 = self.ordered_crossover(parent2, parent1)
                    else:
                        child1 = parent1.copy()
                        child2 = parent2.copy()
                    
                    # 变异
                    if random.random() < mutation_rate:
                        child1 = self.swap_mutation(child1)
                    if random.random() < mutation_rate:
                        child2 = self.swap_mutation(child2)
                    
                    offspring.extend([child1, child2])
                else:
                    child = selected[i].copy()
                    if random.random() < mutation_rate:
                        child = self.swap_mutation(child)
                    offspring.append(child)
            
            # 更新种群
            population = offspring
            
            # 评估新种群
            costs = []
            for chrom in population:
                cost = self.calculate_cost(chrom, noc_records, chiplet_id, original_layer_to_tiles)
                costs.append(cost)
            
            # 精英保留
            current_best_idx = np.argmin(costs)
            current_best_cost = costs[current_best_idx]
            
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_chromosome = deepcopy(population[current_best_idx])
            
            # 记录历史
            if gen % 10 == 0 or gen == generations:
                history.append({
                    'generation': gen,
                    'best_cost': best_cost,
                    'avg_cost': np.mean(costs),
                    'worst_cost': np.max(costs)
                })
                if gen % 10 == 0:
                    print(f"  Gen {gen}: Best = {best_cost:.2f}, Avg = {np.mean(costs):.2f}")
        
        print(f"  Final: Best cost = {best_cost:.2f}")
        return best_chromosome, best_cost, history

def parse_original_layout(original_layout):
    """解析原始布局，得到层到tile的映射"""
    layer_to_tiles = {}
    
    for tile_idx, layer_id in enumerate(original_layout):
        if layer_id not in layer_to_tiles:
            layer_to_tiles[layer_id] = []
        layer_to_tiles[layer_id].append(tile_idx)
    
    return layer_to_tiles

def update_noc_with_new_mapping(noc_records, chiplet_id, original_layer_to_tiles, chromosome):
    """根据新的染色体更新NoC记录"""
    if chiplet_id not in noc_records:
        return []
    
    # 创建新的层映射
    optimizer = LayerMappingOptimizer()
    new_mapping = optimizer.create_layer_mapping_from_chromosome(chromosome, original_layer_to_tiles)
    
    updated_records = []
    
    for record in noc_records[chiplet_id]:
        src_tiles, dst_tiles, data_volume = record
        
        # 找出源层和目的层
        src_layer = None
        dst_layer = None
        
        for layer_id, tiles in original_layer_to_tiles.items():
            if set(src_tiles) == set(tiles):
                src_layer = layer_id
            if set(dst_tiles) == set(tiles):
                dst_layer = layer_id
        
        # 获取新的tile分配
        if src_layer and dst_layer:
            new_src_tiles = new_mapping[src_layer]
            new_dst_tiles = new_mapping[dst_layer]
        else:
            new_src_tiles = src_tiles
            new_dst_tiles = dst_tiles
        
        updated_records.append([new_src_tiles, new_dst_tiles, data_volume])
    
    return updated_records

def optimize_noc_layout_complete(noc_records, nop_records, original_layouts, 
                                 grid_size=4, pop_size=15, generations=30):
    """完整的NoC布局优化"""
    
    print("=" * 80)
    print("NoC布局优化开始")
    print("=" * 80)
    
    # 创建优化器
    optimizer = LayerMappingOptimizer(grid_size)
    
    # 获取chiplet数量
    chiplet_ids = list(noc_records.keys())
    
    # 优化每个chiplet
    optimized_noc = {}
    optimized_nop = deepcopy(nop_records)
    optimized_chromosomes = {}
    optimization_results = {}
    
    for chiplet_id in chiplet_ids:
        print(f"\n优化 Chiplet {chiplet_id}:")
        
        if chiplet_id not in original_layouts:
            print(f"  警告: Chiplet {chiplet_id} 没有原始布局信息")
            continue
        
        original_layout = original_layouts[chiplet_id]
        original_layer_to_tiles = parse_original_layout(original_layout)
        
        # 验证原始布局
        all_tiles = []
        for tiles in original_layer_to_tiles.values():
            all_tiles.extend(tiles)
        
        if len(set(all_tiles)) != len(original_layout):
            print(f"  错误: Chiplet {chiplet_id} 的原始布局有重复tile")
            continue
        
        # 创建原始染色体
        original_chromosome = []
        for layer in sorted(original_layer_to_tiles.keys()):
            original_chromosome.extend(original_layer_to_tiles[layer])
        
        # 计算原始代价
        original_cost = optimizer.calculate_cost(
            original_chromosome, noc_records, chiplet_id, original_layer_to_tiles
        )
        print(f"  原始布局代价: {original_cost:.2f}")
        
        # 遗传算法优化
        best_chromosome, best_cost, history = optimizer.genetic_algorithm_optimize(
            noc_records, chiplet_id, original_layer_to_tiles, 
            pop_size=pop_size, generations=generations
        )
        
        # 保存优化结果
        optimized_chromosomes[chiplet_id] = best_chromosome
        optimization_results[chiplet_id] = {
            'original_cost': original_cost,
            'optimized_cost': best_cost,
            'improvement': ((original_cost - best_cost) / original_cost * 100) if original_cost > 0 else 0,
            'history': history
        }
        
        print(f"  优化布局代价: {best_cost:.2f}")
        print(f"  优化提升: {optimization_results[chiplet_id]['improvement']:.2f}%")
        
        # 更新NoC记录
        updated_noc_records = update_noc_with_new_mapping(
            noc_records, chiplet_id, original_layer_to_tiles, best_chromosome
        )
        optimized_noc[chiplet_id] = updated_noc_records
        
        # 显示优化前后的映射
        print(f"\n  Chiplet {chiplet_id} 原始映射:")
        for layer_id in sorted(original_layer_to_tiles.keys()):
            print(f"    层{layer_id}: tiles {sorted(original_layer_to_tiles[layer_id])}")
        
        # 创建新的映射
        new_mapping = optimizer.create_layer_mapping_from_chromosome(best_chromosome, original_layer_to_tiles)
        print(f"\n  Chiplet {chiplet_id} 优化后映射:")
        for layer_id in sorted(new_mapping.keys()):
            print(f"    层{layer_id}: tiles {sorted(new_mapping[layer_id])}")
    
    # 统计总优化效果
    print("\n" + "=" * 80)
    print("优化结果汇总:")
    print("=" * 80)
    
    total_original = 0
    total_optimized = 0
    
    for chiplet_id, result in optimization_results.items():
        total_original += result['original_cost']
        total_optimized += result['optimized_cost']
        print(f"Chiplet {chiplet_id}: {result['original_cost']:.2f} -> {result['optimized_cost']:.2f} "
              f"({result['improvement']:.1f}%)")
    
    if total_original > 0:
        total_improvement = (total_original - total_optimized) / total_original * 100
        print(f"\n总优化效果: {total_original:.2f} -> {total_optimized:.2f} ({total_improvement:.1f}%)")
    
    print("\n" + "=" * 80)
    print("优化完成!")
    print("=" * 80)
    
    return optimized_noc, optimized_nop, optimized_chromosomes

def convert_noc_to_layout(noc_records, grid_size=4):
    """
    从NoC记录转换为原始布局
    
    参数:
        noc_records: NoC记录字典 {chiplet_id: [[源tile列表], [目的tile列表], 数据量], ...}
        grid_size: 网格大小，默认4x4=16个tile
    
    返回:
        original_layouts: 原始布局字典 {chiplet_id: [层分配列表]}
    """
    original_layouts = {}
    tile_count = grid_size * grid_size
    
    for chiplet_id, records in noc_records.items():
        # 1. 收集所有唯一的tile组合（每个组合对应一个层）
        tile_groups = []
        
        for record in records:
            src_tiles, dst_tiles, _ = record
            
            # 添加源tile组合
            if src_tiles not in tile_groups:
                tile_groups.append(src_tiles)
            
            # 添加目的tile组合
            if dst_tiles not in tile_groups:
                tile_groups.append(dst_tiles)
        
        # 2. 按tile数量排序（小的在前），然后按第一个tile编号排序
        # 这样可以确保层ID的分配是确定性的
        tile_groups.sort(key=lambda x: (len(x), x[0] if x else 0))
        
        # 3. 创建布局数组
        layout = [-1] * tile_count
        
        # 4. 为每个tile组合分配层ID
        for layer_id, tile_group in enumerate(tile_groups, 1):  # 层ID从1开始
            for tile in tile_group:
                if 0 <= tile < tile_count:
                    layout[tile] = layer_id
        
        # 5. 处理未分配的tile（如果有）
        unassigned_tiles = [i for i, layer_id in enumerate(layout) if layer_id == -1]
        if unassigned_tiles:
            # 为未分配的tile分配新的层ID
            next_layer_id = max(layout) + 1 if any(x != -1 for x in layout) else 1
            
            # 将未分配的tile分组，每组大小与最小tile组相同
            if tile_groups:
                min_group_size = min(len(group) for group in tile_groups)
            else:
                min_group_size = 1
            
            for i in range(0, len(unassigned_tiles), min_group_size):
                group = unassigned_tiles[i:i+min_group_size]
                for tile in group:
                    layout[tile] = next_layer_id
                next_layer_id += 1
        
        original_layouts[chiplet_id] = layout
    
    return original_layouts

# 快速测试函数
def quick_test():
    """快速测试"""
    # 示例数据
    noc_records_example = {
        0: [[[0, 1], [2, 3], 262144], 
            [[2, 3], [4, 5], 262144], 
            [[4, 5], [6, 7], 262144], 
            [[6, 7], [8, 9], 262144], 
            [[8, 9], [10, 11], 262144], 
            [[10, 11], [12, 13], 262144], 
            [[12, 13], [14, 15], 262144]],
        
        1: [[[0, 1], [2, 3], 131072], 
            [[2, 3], [4, 5], 131072], 
            [[4, 5], [6, 7], 131072], 
            [[6, 7], [8, 9], 131072], 
            [[8, 9], [10, 11], 131072], 
            [[10, 11], [12, 13, 14, 15], 65536]],
        
        2: [[[0, 1, 2, 3], [4, 5, 6, 7], 65536], 
            [[4, 5, 6, 7], [8, 9, 10, 11], 65536], 
            [[8, 9, 10, 11], [12, 13, 14, 15], 65536]]
    }
    
    nop_records_example = {
        0: [[[0], [1], 131072], 
            [[1], [2], 65536], 
            [[2], [3], 1024]]
    }
    
    # 原始布局
    # original_layouts = {
    #     0: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    #     1: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7],
    #     2: [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
    # }
    original_layouts = convert_noc_to_layout(noc_records_example)
    print(original_layouts)
    print("开始优化...")
    
    # 运行优化（使用较小的参数以加快速度）
    optimized_noc, optimized_nop, optimized_chromosomes = optimize_noc_layout_complete(
        noc_records_example, 
        nop_records_example,
        original_layouts,
        grid_size=4,
        pop_size=20,      # 较小的种群
        generations=30    # 较少的代数
    )

    print("优化完成！")
    print("优化后的NoC布局：")
    print(optimized_noc)
    print("优化后的NOP布局：")
    print(optimized_nop)
    
    return optimized_noc, optimized_nop, optimized_chromosomes

if __name__ == "__main__":
    import time
    
    print("开始测试NoC布局优化...")
    start_time = time.time()
    
    optimized_noc, optimized_nop, optimized_chromosomes = quick_test()
    print(optimized_noc)
    print(optimized_nop)
    
    end_time = time.time()
    print(f"\n总运行时间: {end_time - start_time:.2f} 秒")