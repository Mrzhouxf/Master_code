from __future__ import annotations
from model_interface.function import *
from Hardware_Model.Buffer import buffer
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.PE_latency import PE_latency_analysis

from copy import deepcopy
from typing import List
import math
import configparser as cp
SimConfig = os.path.join(os.getcwd(), "SimConfig.ini")
# print(SimConfig)
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
        self.layer_sequence_latency = []
        self.total_layer_sq_latency = 0
        self.heterogeneous = 0
        self.pe_num = 0
        self.tile_num = 0
    def intra_layer_optimize(self):
        hetero = cp.ConfigParser()
        hetero.read(SimConfig,encoding='UTF-8')
        self.heterogeneous = int(hetero.get('Package level','heterogeneous'))
        tile_num_list = list(map(int,hetero.get('Architecture level','Tile_Num').split(',')))
        self.tile_num = tile_num_list[0]*tile_num_list[1]
        pe_num_list = list(map(int,hetero.get('Tile level','PE_Num').split(',')))
        self.pe_num = pe_num_list[0]*pe_num_list[1]
        print('pe_num:',self.pe_num)
        pareto = []
        optimal_map = []
        if self.heterogeneous == 1:
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
                # print(s)
                if len(s) == 4:
                    read_row = self.nn[k][3]*self.nn[k][3]*s[3][2]
                    read_col = (s[3][0]-self.nn[k][3])*(s[3][1]-self.nn[k][3])*s[3][3]*s[2]
                    indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
                    rdata = s[0]
                    precision = self.inprecision
                    PE_num = s[2]
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
        else:
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
                
            for s in optimal_map:
                # print(s)
                if len(s) == 4:
                    read_row = self.nn[k][3]*self.nn[k][3]*s[3][2]
                    read_col = (s[3][0]-self.nn[k][3])*(s[3][1]-self.nn[k][3])*s[3][3]*s[2]
                    indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
                    rdata = s[0]
                    precision = self.inprecision
                    PE_num = s[2]
                    need_tile_num = math.ceil(PE_num/self.pe_num)
                    tile_analysis = tile_latency_analysis(self.SimConfig,math.ceil(read_row/need_tile_num),math.ceil(read_col/need_tile_num),math.ceil(indata/need_tile_num),math.ceil(rdata/need_tile_num),precision,self.pe_num)
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
                    need_tile_num = math.ceil(PE_num/self.pe_num)
                    tile_analysis = tile_latency_analysis(self.SimConfig,math.ceil(read_row/need_tile_num),math.ceil(read_col/need_tile_num),math.ceil(indata/need_tile_num),math.ceil(rdata/need_tile_num),precision,self.pe_num)
                    self.total_latency = tile_analysis.tile_latency +self.total_latency
                    # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
                    self.layer_latency.append(tile_analysis.tile_latency)
        return optimal_map, self.is_split, self.total_latency, self.layer_latency
    
    # 片外读写延迟
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
    
    # def layer_pipline():
    #     return
    

    def layer_sequence(self):
        layers_info = []                                        # 最终返回列表
        read_data = [ly[0]*ly[1]*ly[2] for ly in self.nn]      # 每层读数据量
        last_out  = (self.nn[-1][0] - self.nn[-1][3])**2 * self.nn[-1][5]

        layer_sq_buffer3 = buffer(self.SimConfig,3)
        for idx, res in enumerate(self.single_layer_map):
            read  = read_data[idx]
            write = last_out if idx == len(self.single_layer_map)-1 else read_data[idx+1]

            if res[2] > self.resource:                             # ===== 需要拆分 =====
                batch_num = math.ceil(res[2] / self.resource)
                for b in range(1, batch_num + 1):
                    layers_info.append({
                        'layer': [idx],
                        'off_chip_read_data':  read,
                        'off_chip_write_data': math.ceil(write / batch_num),
                        'is_split_part': True,
                        'split_part_info': (b, batch_num),      # (第b批, 总批数)
                        'note': idx
                    })
                    layer_sq_buffer3.calculate_buf_write_latency(math.ceil(write / batch_num))
                    layer_sq_buffer3.calculate_buf_write_energy(math.ceil(write / batch_num))
                    layer_sq_buffer3.calculate_buf_read_latency(read)
                    layer_sq_buffer3.calculate_buf_read_energy(read)
                    temp_w_latency = layer_sq_buffer3.buf_wlatency
                    temp_w_energy = layer_sq_buffer3.buf_wenergy
                    temp_r_latency = layer_sq_buffer3.buf_rlatency
                    temp_r_energy = layer_sq_buffer3.buf_renergy
                    read_row = self.nn[idx][3]*self.nn[idx][3]*self.single_layer_map[idx][5]
                    read_col = (self.single_layer_map[idx][3]-self.nn[idx][3])*(self.single_layer_map[idx][4]-self.nn[idx][3])*self.single_layer_map[idx][6]
                    tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,read,self.single_layer_map[idx][0],self.inprecision,self.resource)
                
                self.layer_sequence_latency.append(temp_r_latency*batch_num+temp_w_latency*batch_num+tile_analysis.tile_latency)
            else:                                               # ===== 不拆分 =====
                layers_info.append({
                    'layer': [idx],
                    'off_chip_read_data': read,
                    'off_chip_write_data': write,
                    'is_split_part': False,
                    'split_part_info': None,
                    'note': idx
                })
                layer_sq_buffer3.calculate_buf_write_latency(write)
                layer_sq_buffer3.calculate_buf_write_energy(write)
                layer_sq_buffer3.calculate_buf_read_latency(read)
                layer_sq_buffer3.calculate_buf_read_energy(read)
                temp_w_latency = layer_sq_buffer3.buf_wlatency
                temp_w_energy = layer_sq_buffer3.buf_wenergy
                temp_r_latency = layer_sq_buffer3.buf_rlatency
                temp_r_energy = layer_sq_buffer3.buf_renergy
                read_row = self.nn[idx][3]*self.nn[idx][3]*self.single_layer_map[idx][5]
                read_col = (self.single_layer_map[idx][3]-self.nn[idx][3])*(self.single_layer_map[idx][4]-self.nn[idx][3])*self.single_layer_map[idx][6]
                tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,read,self.single_layer_map[idx][0],self.inprecision,self.resource)
                accelerate_ratio = math.ceil(self.resource/res[2])
                self.layer_sequence_latency.append(temp_r_latency+temp_w_latency+tile_analysis.tile_latency/accelerate_ratio)
        
        self.total_layer_sq_latency = sum(self.layer_sequence_latency)
        return layers_info,self.layer_sequence_latency,self.total_layer_sq_latency
    
    def layer_by_pipeline(self):



        return
    
    def inter_tile(self):
        return
    
    def intra_tile(self):
        return