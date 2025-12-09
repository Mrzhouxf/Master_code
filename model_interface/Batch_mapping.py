from __future__ import annotations
from model_interface.function import *
from Hardware_Model.Buffer import buffer
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.PE_latency import PE_latency_analysis

from typing import List
import math

# SimConfig = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")

def select_optimal_intra_schemes(multi_layer_schemes, total_resource_limit, weights):
    """
    多层方案选择函数（支持主次优先级排序）。
    
    输入:
        multi_layer_schemes: 三维列表 [Layer][Option][Data, Cycle, Resource, Design]
        total_resource_limit: 资源限制总和
        weights: [a, b] 权重列表
        
    返回:
        最优方案列表
    """
    if not multi_layer_schemes:
        return []
        
    alpha, beta = weights
    
    # ----------------------------------------------------
    # 1. 全局数据收集与 Min-Max 归一化参数计算
    # ----------------------------------------------------
    all_data = []
    all_cycles = []
    
    for layer in multi_layer_schemes:
        for opt in layer:
            all_data.append(opt[0])
            all_cycles.append(opt[1])
            
    # 获取最大最小值，防止分母为0
    d_min, d_max = min(all_data), max(all_data)
    c_min, c_max = min(all_cycles), max(all_cycles)
    d_range = d_max - d_min if d_max != d_min else 1.0
    c_range = c_max - c_min if c_max != c_min else 1.0

    # ----------------------------------------------------
    # 2. 预处理每一层每一个方案的 "代价元组"
    # ----------------------------------------------------
    # Python元组比较规则：(A, B) < (C, D) 
    # 首先比较A和C，如果A < C则结果为True；
    # 如果A == C，则继续比较B和D。
    
    normalized_layers = []
    for layer in multi_layer_schemes:
        layer_opts = []
        for idx, option in enumerate(layer):
            d_val, c_val, res_val = option[0], option[1], option[2]
            
            # 归一化到 [0, 1]
            norm_d = (d_val - d_min) / d_range
            norm_c = (c_val - c_min) / c_range
            
            # --- 核心逻辑：构建代价元组 ---
            weighted_main_cost = alpha * norm_d + beta * norm_c
            
            # 定义副代价 (Tie-breaker)
            # 如果 alpha > beta (看重数据)，副指标就是 Cycles
            # 如果 beta > alpha (看重计算)，副指标就是 Data
            # 如果相等，这里默认用 Cycles 做副指标
            if alpha >= beta:
                tie_breaker = norm_c 
            else:
                tie_breaker = norm_d
                
            # 存储格式：(主代价, 副代价, 资源消耗, 原始索引)
            # 注意：这里的 cost 是一个 tuple
            cost_tuple = (weighted_main_cost, tie_breaker)
            
            layer_opts.append({
                'cost_tuple': cost_tuple, 
                'res': res_val, 
                'orig_idx': idx
            })
        normalized_layers.append(layer_opts)

    # ----------------------------------------------------
    # 3. 动态规划 (DP)
    # ----------------------------------------------------
    # dp[resource] = (accumulated_cost_tuple, last_layer_idx)
    # 初始状态：资源消耗0，代价为(0.0, 0.0)
    dp = {0: (0.0, 0.0)}
    
    # 记录路径用于回溯: parent[layer][res] = (prev_res, option_idx)
    num_layers = len(normalized_layers)
    parent = [{} for _ in range(num_layers)]

    for layer_idx in range(num_layers):
        new_dp = {}
        current_layer_opts = normalized_layers[layer_idx]
        
        for prev_res, prev_cost_tuple in dp.items():
            for opt in current_layer_opts:
                new_res = prev_res + opt['res']
                
                # 资源检查
                if new_res <= total_resource_limit:
                    # 代价累加：元组的对应位置相加
                    # new_cost = (主代价和, 副代价和)
                    new_cost_tuple = (
                        prev_cost_tuple[0] + opt['cost_tuple'][0],
                        prev_cost_tuple[1] + opt['cost_tuple'][1]
                    )
                    
                    # 状态转移：
                    # 如果该资源状态未记录，或找到了更小的代价元组（自动处理主次排序）
                    if new_res not in new_dp or new_cost_tuple < new_dp[new_res]:
                        new_dp[new_res] = new_cost_tuple
                        parent[layer_idx][new_res] = (prev_res, opt['orig_idx'])
        
        dp = new_dp
        if not dp:
            print(f"Error: Layer {layer_idx+1} 无法找到满足资源限制的方案。")
            return None

    # ----------------------------------------------------
    # 4. 选择最优结果并回溯
    # ----------------------------------------------------
    # 在最后一层的所有可行资源状态中，找代价元组最小的
    best_res = -1
    min_final_cost_tuple = (float('inf'), float('inf'))
    
    for res, cost_tuple in dp.items():
        if cost_tuple < min_final_cost_tuple:
            min_final_cost_tuple = cost_tuple
            best_res = res
            
    if best_res == -1:
        return None

    # 回溯
    selected_schemes = []
    curr_res = best_res
    for i in range(num_layers - 1, -1, -1):
        prev_res, opt_idx = parent[i][curr_res]
        selected_schemes.append(multi_layer_schemes[i][opt_idx])
        curr_res = prev_res
        
    selected_schemes.reverse()
    return selected_schemes



class Batch_mapping:
    def __init__(self, batch_map:List[dict],net:List[List],mapping_data:List[List],resource_limit:int,array_row:int,array_col:int,inprecision:int):
        # batch_map:List[dict] = {'batch1':mapping_data,'batch2':mapping_data}
        
        
        self.num_batch = len(batch_map)
        self.batch_map = batch_map
        self.nn = net
        self.single_layer_map = mapping_data
        self.resource = resource_limit
        self.is_split = {}
        self.ar = array_row
        self.ac = array_col
        self.total_latency = 0
        self.inprecision = inprecision
        self.SimConfig = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
        # transfer_latency tile_buf_wlatency PE_latency jointmodule_latency
        self.layer_latency = []
        self.layers_load_data = []
    def intra_layer_optimize(self):
        pareto = []
        optimal_map = []
        for i in range(self.num_batch):
            used_resource = 0
            pareto = []
            for k in self.batch_map[i]['layers']:

                used_resource = used_resource + self.single_layer_map[k][2]
                dynamic_resource = self.resource - used_resource
                if dynamic_resource < 0:
                    if k in self.is_split.keys():
                        self.is_split[k] = self.is_split[k] + 1
                    else:
                        self.is_split[k] = 1
                        optimal_map.append(self.single_layer_map[k])
                    # if self.single_layer_map[k] in optimal_map:
                    #     break
                    # else:
                    #     optimal_map.append(self.single_layer_map[k])
                    # self.is_split[k] = 1

                else:
                    schemes = auto_mapping(self.nn[k][0],self.nn[k][1],self.nn[k][3],self.nn[k][2],int(self.nn[k][5]),self.ar,self.ac,int(self.single_layer_map[k][2]+dynamic_resource))
                    
                    pareto.append(schemes)
                    self.is_split[k] = 1 
                    
            result = select_optimal_intra_schemes(pareto,self.resource,[1,0])
            # print(pareto)
            if result:
                optimal_map.extend(result)
            # if k==1:
            #     print("pareto:",pareto)
            #     print("optimal_map:",optimal_map)
            # print(len(optimal_map))
        for s in optimal_map:
            print(s)
            if len(s) == 4:
                read_row = self.nn[k][3]*self.nn[k][3]*s[3][2]
                read_col = (s[3][0]-self.nn[k][3])*(s[3][1]-self.nn[k][3])*s[3][3]*s[2]
                indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
                rdata = s[0]
                precision = self.inprecision
                PE_num = s[2]
                # print(SimConfig)
                tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,indata,rdata,precision,PE_num)
                self.total_latency = tile_analysis.tile_latency +self.total_latency
                # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
                self.layer_latency.append(tile_analysis.tile_latency)
            else:
                read_row = self.nn[k][3]*self.nn[k][3]*s[5]
                read_col = (s[3]-self.nn[k][3])*(s[4]-self.nn[k][3])*s[6]*s[2]
                indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
                rdata = s[0]
                precision = self.inprecision
                PE_num = s[2]
                tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,indata,rdata,precision,PE_num)
                self.total_latency = tile_analysis.tile_latency+self.total_latency
                self.layer_latency.append(tile_analysis.tile_latency)
                # print(len(self.layer_latency))
                    # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
        return optimal_map, self.is_split, self.total_latency, self.layer_latency
    
    def off_chip_latency(self):
        for i in range(len(self.nn)):
            # if self.nn[i][-1]>max_layer_resource:
            #     max_layer_resource = self.nn[i][-1]
            self.layers_load_data.append([self.nn[i][0]*self.nn[i][1]*self.nn[i][2],self.single_layer_map[i][2]])

        optimal_data,optimal_plan = find_optimal_network_mapping(self.layers_load_data,self.resource)
        print("optimal_data:",optimal_data)
        buffer3 = buffer(self.SimConfig,3)
        buffer3.calculate_buf_write_latency(math.ceil(optimal_data*self.inprecision/8))
        buffer3.calculate_buf_write_energy(math.ceil(optimal_data*self.inprecision/8))

        return buffer3.buf_wlatency,buffer3.buf_wenergy
