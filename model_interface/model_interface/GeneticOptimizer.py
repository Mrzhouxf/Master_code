import random
from copy import deepcopy

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
    