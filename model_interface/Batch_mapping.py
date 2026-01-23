# from __future__ import annotations
# from model_interface.function import *
# from Hardware_Model.Buffer import buffer
# from Latency_Model.Tile_latency import tile_latency_analysis
# from Latency_Model.PE_latency import PE_latency_analysis

# from copy import deepcopy
# from typing import List
# import math
# import configparser as cp
# SimConfig = os.path.join(os.getcwd(), "SimConfig.ini")
# # print(SimConfig)
# def select_optimal_intra_schemes(multi_layer_schemes, total_resource_limit, weights):
#     """
#     多层方案选择函数（支持主次优先级排序）。
    
#     输入:
#         multi_layer_schemes: 三维列表 [Layer][Option][Data, Cycle, Resource, Design]
#         total_resource_limit: 资源限制总和
#         weights: [a, b] 权重列表
        
#     返回:
#         最优方案列表
#     """
#     if not multi_layer_schemes:
#         return []
        
#     alpha, beta = weights
    
#     # ----------------------------------------------------
#     # 1. 全局数据收集与 Min-Max 归一化参数计算
#     # ----------------------------------------------------
#     all_data = []
#     all_cycles = []
    
#     for layer in multi_layer_schemes:
#         for opt in layer:
#             all_data.append(opt[0])
#             all_cycles.append(opt[1])
            
#     # 获取最大最小值，防止分母为0
#     d_min, d_max = min(all_data), max(all_data)
#     c_min, c_max = min(all_cycles), max(all_cycles)
#     d_range = d_max - d_min if d_max != d_min else 1.0
#     c_range = c_max - c_min if c_max != c_min else 1.0

#     # ----------------------------------------------------
#     # 2. 预处理每一层每一个方案的 "代价元组"
#     # ----------------------------------------------------
#     # Python元组比较规则：(A, B) < (C, D) 
#     # 首先比较A和C，如果A < C则结果为True；
#     # 如果A == C，则继续比较B和D。
    
#     normalized_layers = []
#     for layer in multi_layer_schemes:
#         layer_opts = []
#         for idx, option in enumerate(layer):
#             d_val, c_val, res_val = option[0], option[1], option[2]
            
#             # 归一化到 [0, 1]
#             norm_d = (d_val - d_min) / d_range
#             norm_c = (c_val - c_min) / c_range
            
#             # --- 核心逻辑：构建代价元组 ---
#             weighted_main_cost = alpha * norm_d + beta * norm_c
            
#             # 定义副代价 (Tie-breaker)
#             # 如果 alpha > beta (看重数据)，副指标就是 Cycles
#             # 如果 beta > alpha (看重计算)，副指标就是 Data
#             # 如果相等，这里默认用 Cycles 做副指标
#             if alpha >= beta:
#                 tie_breaker = norm_c 
#             else:
#                 tie_breaker = norm_d
                
#             # 存储格式：(主代价, 副代价, 资源消耗, 原始索引)
#             # 注意：这里的 cost 是一个 tuple
#             cost_tuple = (weighted_main_cost, tie_breaker)
            
#             layer_opts.append({
#                 'cost_tuple': cost_tuple, 
#                 'res': res_val, 
#                 'orig_idx': idx
#             })
#         normalized_layers.append(layer_opts)

#     # ----------------------------------------------------
#     # 3. 动态规划 (DP)
#     # ----------------------------------------------------
#     # dp[resource] = (accumulated_cost_tuple, last_layer_idx)
#     # 初始状态：资源消耗0，代价为(0.0, 0.0)
#     dp = {0: (0.0, 0.0)}
    
#     # 记录路径用于回溯: parent[layer][res] = (prev_res, option_idx)
#     num_layers = len(normalized_layers)
#     parent = [{} for _ in range(num_layers)]

#     for layer_idx in range(num_layers):
#         new_dp = {}
#         current_layer_opts = normalized_layers[layer_idx]
        
#         for prev_res, prev_cost_tuple in dp.items():
#             for opt in current_layer_opts:
#                 new_res = prev_res + opt['res']
                
#                 # 资源检查
#                 if new_res <= total_resource_limit:
#                     # 代价累加：元组的对应位置相加
#                     # new_cost = (主代价和, 副代价和)
#                     new_cost_tuple = (
#                         prev_cost_tuple[0] + opt['cost_tuple'][0],
#                         prev_cost_tuple[1] + opt['cost_tuple'][1]
#                     )
                    
#                     # 状态转移：
#                     # 如果该资源状态未记录，或找到了更小的代价元组（自动处理主次排序）
#                     if new_res not in new_dp or new_cost_tuple < new_dp[new_res]:
#                         new_dp[new_res] = new_cost_tuple
#                         parent[layer_idx][new_res] = (prev_res, opt['orig_idx'])
        
#         dp = new_dp
#         if not dp:
#             print(f"Error: Layer {layer_idx+1} 无法找到满足资源限制的方案。")
#             return None

#     # ----------------------------------------------------
#     # 4. 选择最优结果并回溯
#     # ----------------------------------------------------
#     # 在最后一层的所有可行资源状态中，找代价元组最小的
#     best_res = -1
#     min_final_cost_tuple = (float('inf'), float('inf'))
    
#     for res, cost_tuple in dp.items():
#         if cost_tuple < min_final_cost_tuple:
#             min_final_cost_tuple = cost_tuple
#             best_res = res
            
#     if best_res == -1:
#         return None

#     # 回溯
#     selected_schemes = []
#     curr_res = best_res
#     for i in range(num_layers - 1, -1, -1):
#         prev_res, opt_idx = parent[i][curr_res]
#         selected_schemes.append(multi_layer_schemes[i][opt_idx])
#         curr_res = prev_res
        
#     selected_schemes.reverse()
#     return selected_schemes



# class Batch_mapping:
#     def __init__(self, batch_map:List[dict],net:List[List],mapping_data:List[List],resource_limit:int,array_row:int,array_col:int,inprecision:int):
#         # batch_map:List[dict] = {'batch1':mapping_data,'batch2':mapping_data}
        
        
#         self.num_batch = len(batch_map)
#         self.batch_map = batch_map
#         self.nn = net
#         self.single_layer_map = mapping_data
#         self.resource = resource_limit
#         self.is_split = {}
#         self.ar = array_row
#         self.ac = array_col
#         self.total_latency = 0
#         self.inprecision = inprecision
#         self.SimConfig = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
#         # transfer_latency tile_buf_wlatency PE_latency jointmodule_latency
#         self.layer_latency = []
#         self.layers_load_data = []
#         self.layer_sequence_latency = []
#         self.total_layer_sq_latency = 0
#         self.heterogeneous = 0
#         self.pe_num = 0
#         self.tile_num = 0
#     def intra_layer_optimize(self):
#         hetero = cp.ConfigParser()
#         hetero.read(SimConfig,encoding='UTF-8')
#         self.heterogeneous = int(hetero.get('Package level','heterogeneous'))
#         tile_num_list = list(map(int,hetero.get('Architecture level','Tile_Num').split(',')))
#         self.tile_num = tile_num_list[0]*tile_num_list[1]
#         pe_num_list = list(map(int,hetero.get('Tile level','PE_Num').split(',')))
#         self.pe_num = pe_num_list[0]*pe_num_list[1]
#         print('pe_num:',self.pe_num)
#         pareto = []
#         optimal_map = []
#         if self.heterogeneous == 1:
#             for i in range(self.num_batch):
#                 used_resource = 0
#                 pareto = []
#                 for k in self.batch_map[i]['layers']:

#                     used_resource = used_resource + self.single_layer_map[k][2]
#                     dynamic_resource = self.resource - used_resource
#                     if dynamic_resource < 0:
#                         if k in self.is_split.keys():
#                             self.is_split[k] = self.is_split[k] + 1
#                         else:
#                             self.is_split[k] = 1
#                             optimal_map.append(self.single_layer_map[k])
#                         # if self.single_layer_map[k] in optimal_map:
#                         #     break
#                         # else:
#                         #     optimal_map.append(self.single_layer_map[k])
#                         # self.is_split[k] = 1

#                     else:
#                         schemes = auto_mapping(self.nn[k][0],self.nn[k][1],self.nn[k][3],self.nn[k][2],int(self.nn[k][5]),self.ar,self.ac,int(self.single_layer_map[k][2]+dynamic_resource))
                        
#                         pareto.append(schemes)
#                         self.is_split[k] = 1 
                        
#                 result = select_optimal_intra_schemes(pareto,self.resource,[1,0])
#                 # print(pareto)
#                 if result:
#                     optimal_map.extend(result)
#                 # if k==1:
#                 #     print("pareto:",pareto)
#                 #     print("optimal_map:",optimal_map)
#                 # print(len(optimal_map))
#             for s in optimal_map:
#                 # print(s)
#                 if len(s) == 4:
#                     read_row = self.nn[k][3]*self.nn[k][3]*s[3][2]
#                     read_col = (s[3][0]-self.nn[k][3])*(s[3][1]-self.nn[k][3])*s[3][3]*s[2]
#                     indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
#                     rdata = s[0]
#                     precision = self.inprecision
#                     PE_num = s[2]
#                     tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,indata,rdata,precision,PE_num)
#                     self.total_latency = tile_analysis.tile_latency +self.total_latency
#                     # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
#                     self.layer_latency.append(tile_analysis.tile_latency)
#                 else:
#                     read_row = self.nn[k][3]*self.nn[k][3]*s[5]
#                     read_col = (s[3]-self.nn[k][3])*(s[4]-self.nn[k][3])*s[6]*s[2]
#                     indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
#                     rdata = s[0]
#                     precision = self.inprecision
#                     PE_num = s[2]
#                     tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,indata,rdata,precision,PE_num)
#                     self.total_latency = tile_analysis.tile_latency+self.total_latency
#                     self.layer_latency.append(tile_analysis.tile_latency)
#                 # print(len(self.layer_latency))
#                     # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
#         else:
#             for i in range(self.num_batch):
#                 used_resource = 0
#                 pareto = []
#                 for k in self.batch_map[i]['layers']:

#                     used_resource = used_resource + self.single_layer_map[k][2]
#                     dynamic_resource = self.resource - used_resource
#                     if dynamic_resource < 0:
#                         if k in self.is_split.keys():
#                             self.is_split[k] = self.is_split[k] + 1
#                         else:
#                             self.is_split[k] = 1
#                             optimal_map.append(self.single_layer_map[k])
#                         # if self.single_layer_map[k] in optimal_map:
#                         #     break
#                         # else:
#                         #     optimal_map.append(self.single_layer_map[k])
#                         # self.is_split[k] = 1

#                     else:
#                         schemes = auto_mapping(self.nn[k][0],self.nn[k][1],self.nn[k][3],self.nn[k][2],int(self.nn[k][5]),self.ar,self.ac,int(self.single_layer_map[k][2]+dynamic_resource))
                        
#                         pareto.append(schemes)
#                         self.is_split[k] = 1 
                        
#                 result = select_optimal_intra_schemes(pareto,self.resource,[1,0])
#                 # print(pareto)
#                 if result:
#                     optimal_map.extend(result)
                
#             for s in optimal_map:
#                 # print(s)
#                 if len(s) == 4:
#                     read_row = self.nn[k][3]*self.nn[k][3]*s[3][2]
#                     read_col = (s[3][0]-self.nn[k][3])*(s[3][1]-self.nn[k][3])*s[3][3]*s[2]
#                     indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
#                     rdata = s[0]
#                     precision = self.inprecision
#                     PE_num = s[2]
#                     need_tile_num = math.ceil(PE_num/self.pe_num)
#                     tile_analysis = tile_latency_analysis(self.SimConfig,math.ceil(read_row/need_tile_num),math.ceil(read_col/need_tile_num),math.ceil(indata/need_tile_num),math.ceil(rdata/need_tile_num),precision,self.pe_num)
#                     self.total_latency = tile_analysis.tile_latency +self.total_latency
#                     # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
#                     self.layer_latency.append(tile_analysis.tile_latency)
#                 else:
#                     read_row = self.nn[k][3]*self.nn[k][3]*s[5]
#                     read_col = (s[3]-self.nn[k][3])*(s[4]-self.nn[k][3])*s[6]*s[2]
#                     indata = self.nn[k][0]*self.nn[k][1]*self.nn[k][2]*self.inprecision
#                     rdata = s[0]
#                     precision = self.inprecision
#                     PE_num = s[2]
#                     need_tile_num = math.ceil(PE_num/self.pe_num)
#                     tile_analysis = tile_latency_analysis(self.SimConfig,math.ceil(read_row/need_tile_num),math.ceil(read_col/need_tile_num),math.ceil(indata/need_tile_num),math.ceil(rdata/need_tile_num),precision,self.pe_num)
#                     self.total_latency = tile_analysis.tile_latency +self.total_latency
#                     # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
#                     self.layer_latency.append(tile_analysis.tile_latency)
#         return optimal_map, self.is_split, self.total_latency, self.layer_latency
    
#     # 片外读写延迟
#     def off_chip_latency(self):
#         for i in range(len(self.nn)):
#             # if self.nn[i][-1]>max_layer_resource:
#             #     max_layer_resource = self.nn[i][-1]
#             self.layers_load_data.append([self.nn[i][0]*self.nn[i][1]*self.nn[i][2],self.single_layer_map[i][2]])

#         optimal_data,optimal_plan = find_optimal_network_mapping(self.layers_load_data,self.resource)
#         print("optimal_data:",optimal_data)
#         buffer3 = buffer(self.SimConfig,3)
#         buffer3.calculate_buf_write_latency(math.ceil(optimal_data*self.inprecision/8))
#         buffer3.calculate_buf_write_energy(math.ceil(optimal_data*self.inprecision/8))

#         return buffer3.buf_wlatency,buffer3.buf_wenergy
    
#     # def layer_pipline():
#     #     return
    

#     def layer_sequence(self):
#         layers_info = []                                        # 最终返回列表
#         read_data = [ly[0]*ly[1]*ly[2] for ly in self.nn]      # 每层读数据量
#         last_out  = (self.nn[-1][0] - self.nn[-1][3])**2 * self.nn[-1][5]

#         layer_sq_buffer3 = buffer(self.SimConfig,3)
#         for idx, res in enumerate(self.single_layer_map):
#             read  = read_data[idx]
#             write = last_out if idx == len(self.single_layer_map)-1 else read_data[idx+1]

#             if res[2] > self.resource:                             # ===== 需要拆分 =====
#                 batch_num = math.ceil(res[2] / self.resource)
#                 for b in range(1, batch_num + 1):
#                     layers_info.append({
#                         'layer': [idx],
#                         'off_chip_read_data':  read,
#                         'off_chip_write_data': math.ceil(write / batch_num),
#                         'is_split_part': True,
#                         'split_part_info': (b, batch_num),      # (第b批, 总批数)
#                         'note': idx
#                     })
#                     layer_sq_buffer3.calculate_buf_write_latency(math.ceil(write / batch_num))
#                     layer_sq_buffer3.calculate_buf_write_energy(math.ceil(write / batch_num))
#                     layer_sq_buffer3.calculate_buf_read_latency(read)
#                     layer_sq_buffer3.calculate_buf_read_energy(read)
#                     temp_w_latency = layer_sq_buffer3.buf_wlatency
#                     temp_w_energy = layer_sq_buffer3.buf_wenergy
#                     temp_r_latency = layer_sq_buffer3.buf_rlatency
#                     temp_r_energy = layer_sq_buffer3.buf_renergy
#                     read_row = self.nn[idx][3]*self.nn[idx][3]*self.single_layer_map[idx][5]
#                     read_col = (self.single_layer_map[idx][3]-self.nn[idx][3])*(self.single_layer_map[idx][4]-self.nn[idx][3])*self.single_layer_map[idx][6]
#                     tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,read,self.single_layer_map[idx][0],self.inprecision,self.resource)
                
#                 self.layer_sequence_latency.append(temp_r_latency*batch_num+temp_w_latency*batch_num+tile_analysis.tile_latency)
#             else:                                               # ===== 不拆分 =====
#                 layers_info.append({
#                     'layer': [idx],
#                     'off_chip_read_data': read,
#                     'off_chip_write_data': write,
#                     'is_split_part': False,
#                     'split_part_info': None,
#                     'note': idx
#                 })
#                 layer_sq_buffer3.calculate_buf_write_latency(write)
#                 layer_sq_buffer3.calculate_buf_write_energy(write)
#                 layer_sq_buffer3.calculate_buf_read_latency(read)
#                 layer_sq_buffer3.calculate_buf_read_energy(read)
#                 temp_w_latency = layer_sq_buffer3.buf_wlatency
#                 temp_w_energy = layer_sq_buffer3.buf_wenergy
#                 temp_r_latency = layer_sq_buffer3.buf_rlatency
#                 temp_r_energy = layer_sq_buffer3.buf_renergy
#                 read_row = self.nn[idx][3]*self.nn[idx][3]*self.single_layer_map[idx][5]
#                 read_col = (self.single_layer_map[idx][3]-self.nn[idx][3])*(self.single_layer_map[idx][4]-self.nn[idx][3])*self.single_layer_map[idx][6]
#                 tile_analysis = tile_latency_analysis(self.SimConfig,read_row,read_col,read,self.single_layer_map[idx][0],self.inprecision,self.resource)
#                 accelerate_ratio = math.ceil(self.resource/res[2])
#                 self.layer_sequence_latency.append(temp_r_latency+temp_w_latency+tile_analysis.tile_latency/accelerate_ratio)
        
#         self.total_layer_sq_latency = sum(self.layer_sequence_latency)
#         return layers_info,self.layer_sequence_latency,self.total_layer_sq_latency
    
#     def layer_by_pipeline(self):



#         return
    
#     def inter_tile(self):
#         return
    
#     def intra_tile(self):
#         return
    


import configparser as cp
import numpy as np
from typing import List, Dict, Tuple
from Hardware_Model.Buffer import buffer
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.PE_latency import PE_latency_analysis
from model_interface.function import *
from model_interface.noc import *
from Interconnect.noc_estimation import *

class Batch_mapping:
    def __init__(self, SimConfig_path, NN, mapping_data, inprecision: int):
        """
        初始化分批映射类
        
        参数:
            SimConfig_path: 硬件配置文件路径
            NN: 神经网络结构参数（二维数组），每一行[输入高, 输入宽, 输入通道, 卷积核宽, 卷积核高, 输出通道, ...]
            mapping_data: 每一层的映射数据（二维数组），每一行[数据传输量, 计算周期, 使用tile数, ...]
            inprecision: 输入精度（bit）
        """
        self.simconfig_path = SimConfig_path
        self.NN = NN  # 神经网络结构
        self.mapping_data = mapping_data  # 层映射数据
        self.inprecision = inprecision  # 输入精度
        
        # 读取硬件架构配置
        accelerator_arch = cp.ConfigParser()
        accelerator_arch.read(SimConfig_path, encoding='UTF-8')
        
        # 解析硬件参数
        self.chiplet_num = list(map(int, accelerator_arch.get('Package level', 'Chiplet_Num').split(',')))
        self.tile_num = list(map(int, accelerator_arch.get('Architecture level', 'Tile_Num').split(',')))
        self.pe_num = list(map(int, accelerator_arch.get('Tile level', 'PE_Num').split(',')))
        self.array = list(map(int, accelerator_arch.get('Crossbar level', 'Xbar_Size').split(',')))
        self.array_num = int(accelerator_arch.get('Process element level', 'Group_Num'))
        
        # 计算硬件资源总量
        self.total_chiplets = self.chiplet_num[0] * self.chiplet_num[1]
        self.tiles_per_chiplet = self.tile_num[0] * self.tile_num[1]
        self.total_tiles = self.total_chiplets * self.tiles_per_chiplet  # 总可用tile数
        self.pes_per_tile = self.pe_num[0] * self.pe_num[1]
        self.total_pes = self.total_tiles * self.pes_per_tile
        self.arrays_per_pe = self.array_num
        self.arrays_per_tile = self.pes_per_tile * self.arrays_per_pe
        self.arrays_per_chiplet = self.tiles_per_chiplet * self.arrays_per_tile
        
        # 初始化硬件资源状态
        self.hardware_status = {
            'chiplets': [{'tiles': [{'available': True, 'layer_id': -1, 'used_arrays': 0, 
                                    'parallel_id': -1}
                                   for _ in range(self.tiles_per_chiplet)]} 
                        for _ in range(self.total_chiplets)],
            'layer_mappings': []
        }
        
        # 预处理每层数据
        self.layer_info = self._preprocess_layer_info()
        # 最小分批数
        self.min_batch_num = self._calculate_min_batch_num()
        # 最优分批结果
        self.optimal_batches = None
        # 映射结果字典
        self.batch_mapping_result = {}
        
        # 延迟相关
        self.squetial_total_latency = 0
        self.squetial_layer_latency = []
        self.uniform_total_latency = 0
        self.uniform_layer_latency = []

        self.optimize_mapping = []
        self.origin_layer_latency = []
        self.origin_total_latency = 0

        self.optimize_layer_latency = []
        self.optimize_total_latency = 0
    def _preprocess_layer_info(self) -> List[Dict]:
        """预处理每层信息：计算tile需求、传输量、计算周期、输出大小"""
        layer_info = []
        for layer_id, (nn_layer, map_layer) in enumerate(zip(self.NN, self.mapping_data)):
            # 解析NN层参数
            in_h, in_w, in_chan, _, _, out_chan = nn_layer[:6]
            # 解析映射数据
            data_transfer, compute_cycles, tile_need = map_layer[:3]
            
            # 计算层输入数据量（bit）
            input_data_size = in_h * in_w * in_chan * self.inprecision
            # 计算层输出数据量（假设输出精度同输入）
            output_data_size = in_h * in_w * out_chan * self.inprecision  # 卷积后尺寸暂未考虑stride/padding
            
            layer_info.append({
                'layer_id': layer_id,
                'tile_need': int(tile_need),  # 该层需要的tile数
                'data_transfer': data_transfer,  # 原始传输量
                'compute_cycles': compute_cycles,  # 原始计算周期
                'input_data_size': input_data_size,  # 输入数据量
                'output_data_size': output_data_size  # 输出数据量
            })
        return layer_info

    def _calculate_min_batch_num(self) -> int:
        """计算最小分批数：基于总tile需求和总可用tile数"""
        total_tile_need = sum([layer['tile_need'] for layer in self.layer_info])
        # 向上取整得到最小分批数
        min_batch = int(np.ceil(total_tile_need / self.total_tiles))
        return max(min_batch, 1)  # 至少1批

    def _calculate_batch_cost(self, batch_segments: List[int]) -> float:
        """
        计算分批方案的总成本：批次间计算量差值和 + 分批点传输量和
        
        参数:
            batch_segments: 分批点索引，如[5, 10]表示分为[0-4], [5-9], [10-end]
        
        返回:
            总成本（越小越优）
        """
        # 生成批次区间
        batches = []
        start = 0
        for seg in batch_segments:
            batches.append((start, seg-1))
            start = seg
        batches.append((start, len(self.layer_info)-1))
        
        # 计算每批次的总计算周期
        batch_cycles = []
        for (s, e) in batches:
            total_cycles = sum([self.layer_info[i]['compute_cycles'] for i in range(s, e+1)])
            batch_cycles.append(total_cycles)
        
        # 计算批次间计算量差值总和
        cycle_diff_sum = 0
        for i in range(1, len(batch_cycles)):
            cycle_diff_sum += abs(batch_cycles[i] - batch_cycles[i-1])
        
        # 计算分批点传输量总和（每个分批点的输出数据量）
        seg_transfer_sum = sum([self.layer_info[seg-1]['output_data_size'] for seg in batch_segments])
        
        # 总成本 = 计算量差值和 + 分批点传输量和（可加权重调整优先级）
        total_cost = cycle_diff_sum + seg_transfer_sum
        return total_cost

    def find_optimal_batches(self) -> List[List[int]]:
        """
        寻找最优分批方案：遍历可能的分段点，选择总成本最小的方案
        
        返回:
            最优批次划分，如[[0,1,2,3], [4,5,6], [7,8,9]]
        """
        min_batch = self.min_batch_num
        num_layers = len(self.layer_info)
        
        # 生成所有可能的分段点组合（简化版：均匀遍历）
        best_cost = float('inf')
        best_batches = None
        
        # 简单遍历：按最小分批数均匀划分附近的分段点
        # 更优方案可使用动态规划/遗传算法，此处简化实现
        base_seg_step = num_layers // min_batch
        for offset in range(-2, 3):  # 偏移量，扩大搜索范围
            seg_step = max(1, base_seg_step + offset)
            batch_segments = [seg_step * i for i in range(1, min_batch)]
            # 确保分段点有效
            batch_segments = [s for s in batch_segments if s < num_layers-1]
            
            # 计算该方案成本
            cost = self._calculate_batch_cost(batch_segments)
            if cost < best_cost:
                best_cost = cost
                best_batches = batch_segments
        
        # 生成最终批次划分
        final_batches = []
        start = 0
        for seg in best_batches:
            final_batches.append(list(range(start, seg)))
            start = seg
        final_batches.append(list(range(start, num_layers)))
        
        self.optimal_batches = final_batches
        return final_batches

    def map_batches_to_hardware(self) -> Dict:
        """
        将最优批次映射到硬件资源，生成最终的映射字典
        
        返回:
            包含批次信息、层映射、资源利用率的字典
        """
        if self.optimal_batches is None:
            self.find_optimal_batches()
        
        batch_mapping = {
            'batch_num': len(self.optimal_batches),
            'batches': [],
            'total_utilization': {
                'total_tiles_available': self.total_tiles,
                'total_tiles_used': sum([layer['tile_need'] for layer in self.layer_info]),
                'utilization_rate': 0.0
            }
        }
        
        # 计算总体利用率
        total_used = batch_mapping['total_utilization']['total_tiles_used']
        total_available = batch_mapping['total_utilization']['total_tiles_available']
        batch_mapping['total_utilization']['utilization_rate'] = (total_used / (total_available * len(self.optimal_batches))) * 100
        
        # 逐批次映射
        for batch_id, layers_in_batch in enumerate(self.optimal_batches):
            # 初始化该批次的tile分配
            current_chiplet = 0
            current_tile = 0
            batch_layer_mappings = []
            batch_total_cycles = 0
            batch_total_transfer = 0
            
            for layer_id in layers_in_batch:
                layer = self.layer_info[layer_id]
                tile_need = layer['tile_need']
                tile_indices = []
                remaining_tiles = tile_need
                
                # 分配tile
                while remaining_tiles > 0:
                    # 当前chiplet剩余tile数
                    tile_remaining_in_chiplet = self.tiles_per_chiplet - current_tile
                    assign_num = min(remaining_tiles, tile_remaining_in_chiplet)
                    
                    # 生成tile索引
                    tile_indices.extend([current_tile + i for i in range(assign_num)])
                    
                    # 更新状态
                    remaining_tiles -= assign_num
                    current_tile += assign_num
                    
                    # 切换chiplet
                    if current_tile >= self.tiles_per_chiplet:
                        current_chiplet += 1
                        current_tile = 0
            
                # 构建层映射信息
                layer_map = {
                    'layer_id': layer_id,
                    'batch_id': batch_id,
                    'chiplet_id': current_chiplet if remaining_tiles == 0 else current_chiplet - 1,
                    'segment_id': batch_id,
                    'tile_indices': tile_indices,
                    'tile_count': tile_need,
                    'arrays_needed': tile_need * self.arrays_per_tile,
                    'actual_arrays': tile_need * self.arrays_per_tile,
                    'efficiency': 100.0,
                    'data_transfer': layer['data_transfer'],
                    'compute_cycles': layer['compute_cycles'],
                    'input_data_size': layer['input_data_size'],
                    'output_data_size': layer['output_data_size']
                }
                batch_layer_mappings.append(layer_map)
                
                # 累加批次统计
                batch_total_cycles += layer['compute_cycles']
                batch_total_transfer += layer['data_transfer']
            
            # 构建批次信息
            batch_info = {
                'batch_id': batch_id,
                'layers': layers_in_batch,
                'layer_mappings': batch_layer_mappings,
                'total_compute_cycles': batch_total_cycles,
                'total_data_transfer': batch_total_transfer,
                'total_tile_used': sum([self.layer_info[l]['tile_need'] for l in layers_in_batch]),
                'tile_utilization': (sum([self.layer_info[l]['tile_need'] for l in layers_in_batch]) / self.total_tiles) * 100
            }
            batch_mapping['batches'].append(batch_info)
        
        self.batch_mapping_result = batch_mapping
        return batch_mapping

    def run_batch_mapping(self) -> Dict:
        """执行完整的分批映射流程，返回最终结果"""
        # 1. 找到最优分批方案
        self.find_optimal_batches()
        # 2. 映射到硬件并生成结果
        result = self.map_batches_to_hardware()
        return result
    
    def select_best_mapping_scheme(
        self,
        all_layer_schemes,
        segment_info,
        segment_resource_limits,
        weight=[1, 0]
    ):
        """
        强化版：严格满足分段资源上限 + 归一化消除量纲，选择每层最优映射方案
        参数：
            all_layer_schemes: 嵌套列表，长度=层数，每个元素是该层的多个映射方案
                            方案格式：[数据传输量, 计算周期, 使用资源, [映射参数]]
            segment_info: 嵌套列表，每段包含该段的层索引（从0开始），如[[0,1,2,3],[4,5,6],...]
            segment_resource_limits: 列表，长度=分段数，每个元素是对应分段的总资源上限
            weight: 列表，[传输量权重, 计算周期权重]，总和=1，得分越低越优
        返回：
            final_selected: 列表，长度=层数，每个元素是该层选中的唯一方案
        """
        # 严格输入校验
        assert len(weight) == 2 and abs(sum(weight) - 1) < 1e-6, "权重必须长度为2，总和为1"
        assert len(segment_info) == len(segment_resource_limits), "分段数与资源限制数必须一致"
        assert len(all_layer_schemes) == sum(len(seg) for seg in segment_info), "层数与分段层索引不匹配"

        final_selected = [None] * len(all_layer_schemes)

        # 遍历每个分段，独立处理
        for seg_idx, (seg_layers, seg_res_limit) in enumerate(zip(segment_info, segment_resource_limits)):
            print(f"\n===== 处理第 {seg_idx+1} 分段 =====")
            print(f"分段包含层索引：{seg_layers} | 分段资源上限：{seg_res_limit}")

            # ========== 步骤1：收集该分段每层的所有方案，并预处理 ==========
            layer_scheme_dict = {}  # key: 层索引, value: 该层的方案列表（带归一化得分）
            for layer_idx in seg_layers:
                schemes = all_layer_schemes[layer_idx]
                if not schemes:
                    raise ValueError(f"层 {layer_idx} 无可用映射方案")
                
                # 提取该层所有方案的传输量、计算周期、资源
                traffics = np.array([s[0] for s in schemes])
                cycles = np.array([s[1] for s in schemes])
                resources = np.array([s[2] for s in schemes])

                # ---------- 核心：min-max 归一化，消除量纲差异 ----------
                # 防止除零：当所有值相同时，归一化结果为0
                norm_traffic = (traffics - traffics.min()) / (traffics.max() - traffics.min() + 1e-9)
                norm_cycle = (cycles - cycles.min()) / (cycles.max() - cycles.min() + 1e-9)

                # 计算综合得分：得分越低，方案越优
                scores = weight[0] * norm_traffic + weight[1] * norm_cycle

                # 为每个方案添加元信息
                layer_scheme_dict[layer_idx] = [
                    {
                        "raw_scheme": s,
                        "traffic": s[0],
                        "cycle": s[1],
                        "resource": s[2],
                        "score": scores[i],
                        "norm_traffic": norm_traffic[i],
                        "norm_cycle": norm_cycle[i]
                    }
                    for i, s in enumerate(schemes)
                ]

            # ========== 步骤2：枚举合法方案组合（总资源 ≤ 分段上限） ==========
            # 生成每层的方案索引列表，用于笛卡尔积枚举
            layer_idx_list = list(layer_scheme_dict.keys())
            scheme_idx_options = [range(len(layer_scheme_dict[lid])) for lid in layer_idx_list]

            best_combination = None
            min_total_score = float('inf')

            # 遍历所有可能的方案组合（适用于分段层数少的场景）
            from itertools import product
            for idx_tuple in product(*scheme_idx_options):
                # 计算该组合的总资源和总得分
                total_res = 0
                total_score = 0
                combination = []
                for lid, idx in zip(layer_idx_list, idx_tuple):
                    scheme = layer_scheme_dict[lid][idx]
                    total_res += scheme["resource"]
                    total_score += scheme["score"]
                    combination.append((lid, scheme))
                
                # 核心约束：总资源不超过分段上限
                if total_res > seg_res_limit:
                    continue
                
                # 更新最优组合（总得分越低越好）
                if total_score < min_total_score:
                    min_total_score = total_score
                    best_combination = combination

            # 校验是否存在合法组合
            if best_combination is None:
                raise ValueError(
                    f"分段 {seg_idx+1} 无合法方案组合！总资源上限 {seg_res_limit} 过低"
                )

            # ========== 步骤3：保存该分段的最优方案 ==========
            for lid, scheme in best_combination:
                final_selected[lid] = scheme["raw_scheme"]
            
            total_res_selected = sum(s["resource"] for _, s in best_combination)
            print(f"该分段最优组合总资源：{total_res_selected}（≤ 上限 {seg_res_limit}）")
            print(f"该分段最优组合总得分：{min_total_score:.4f}")

        # 最终校验：所有层都有选中的方案
        assert None not in final_selected, "存在未选中方案的层"
        return final_selected
    
    def update_mapping_dict(self, new_mapping_scheme):
        """
        根据新的映射方案更新原始的分批映射字典
        
        参数:
            original_dict: 原始的分批映射字典
            new_mapping_scheme: 新的映射方案列表，每一行[数据传输量, 计算周期, 使用tile数, 映射方案]
            tiles_per_chiplet: 每个chiplet的tile数量，默认16
        
        返回:
            dict: 更新后的映射字典
        """
        # 深拷贝原始字典，避免修改原数据
        import copy
        updated_dict = copy.deepcopy(self.run_batch_mapping())
        
        # 1. 为每个批次重新分配tile并更新层信息
        for batch in updated_dict['batches']:
            batch_id = batch['batch_id']
            layer_mappings = batch['layer_mappings']
            
            # 初始化当前批次的tile分配状态
            current_chiplet_id = 0
            current_tile_idx = 0
            
            # 遍历批次内的每一层
            for layer_map in layer_mappings:
                layer_id = layer_map['layer_id']
                
                # 从新映射方案中获取该层的新参数
                new_transfer, new_cycles, new_tile_count, _ = new_mapping_scheme[layer_id]
                new_tile_count = int(new_tile_count)
                
                # 重新分配tile索引
                tile_indices = []
                remaining_tiles = new_tile_count
                
                # 跨chiplet分配tile
                while remaining_tiles > 0:
                    tiles_remaining_in_chiplet = self.tiles_per_chiplet - current_tile_idx
                    tiles_to_assign = min(remaining_tiles, tiles_remaining_in_chiplet)
                    
                    tile_indices.extend([current_tile_idx + i for i in range(tiles_to_assign)])
                    
                    remaining_tiles -= tiles_to_assign
                    current_tile_idx += tiles_to_assign
                    
                    if current_tile_idx >= self.tiles_per_chiplet:
                        current_chiplet_id += 1
                        current_tile_idx = 0
                
                # 更新层映射信息
                layer_map['tile_count'] = new_tile_count
                layer_map['tile_indices'] = tile_indices
                layer_map['chiplet_id'] = current_chiplet_id if remaining_tiles == 0 else current_chiplet_id - 1
                layer_map['data_transfer'] = new_transfer
                layer_map['compute_cycles'] = new_cycles
                layer_map['arrays_needed'] = new_tile_count  # 保持arrays数与tile数一致
                layer_map['actual_arrays'] = new_tile_count
            
            # 2. 更新批次级别的统计信息
            # 重新计算总tile使用数
            batch['total_tile_used'] = sum([lm['tile_count'] for lm in layer_mappings])
            # 重新计算总计算周期
            batch['total_compute_cycles'] = sum([lm['compute_cycles'] for lm in layer_mappings])
            # 重新计算总数据传输量
            batch['total_data_transfer'] = sum([lm['data_transfer'] for lm in layer_mappings])
            # 重新计算tile利用率
            batch['tile_utilization'] = (batch['total_tile_used'] / self.tiles_per_chiplet) * 100
        
        # 3. 更新全局利用率统计
        total_tiles_used = sum([batch['total_tile_used'] for batch in updated_dict['batches']])
        total_tiles_available = updated_dict['total_utilization']['total_tiles_available']
        total_batches = updated_dict['batch_num']
        
        updated_dict['total_utilization']['total_tiles_used'] = total_tiles_used
        updated_dict['total_utilization']['utilization_rate'] = round(
            (total_tiles_used / (total_tiles_available * total_batches)) * 100, 3
        )
        
        return updated_dict
    
    def evaluate_perfermance(self,new_mapping):
        original_map = self.run_batch_mapping()
        off_chip_latency = 0
        off_chip_energy = 0
        off_chip_read_data = 0
        cut = 0
        
        break_point = []
        off_read_write = []
        buffer3 = buffer(self.simconfig_path,3)
        for batch in original_map['batches']:
            cut = len(batch['layers']) + cut
            break_point.append(len(batch['layers']))
            off_chip_read_data = self.NN[len(batch['layers'])-1][0]*self.NN[len(batch['layers'])-1][1]*self.NN[len(batch['layers'])-1][2]*self.inprecision
            off_read_write.append(off_chip_read_data)
        buffer3.calculate_buf_write_latency(math.ceil(sum(off_read_write)*self.inprecision/8))
        buffer3.calculate_buf_write_energy(math.ceil(sum(off_read_write)*self.inprecision/8))
        dram = []
        dram.append(buffer3.buf_wlatency)
        dram.append(buffer3.buf_wenergy)
        
        for i in range(len(new_mapping)):
            
                origin_read_row = self.NN[i][3]*self.NN[i][3]*self.mapping_data[i][5]
                origin_read_col = (self.mapping_data[i][3]-self.NN[i][3]+1)*(self.mapping_data[i][4]-self.NN[i][3]+1)*self.mapping_data[i][6]
                origin_indata = self.NN[i][0]*self.NN[i][1]*self.NN[i][2]*self.inprecision
                origin_rdata = self.mapping_data[i][0]

                optimize_read_row = self.NN[i][3]*self.NN[i][3]*new_mapping[i][3][2]
                optimize_read_col = (new_mapping[i][3][0]-self.NN[i][3]+1)*(new_mapping[i][3][1]-self.NN[i][3]+1)*new_mapping[i][3][3]
                optimize_indata = self.NN[i][0]*self.NN[i][1]*self.NN[i][2]*self.inprecision
                optimize_rdata = new_mapping[i][0]
                # precision = self.inprecision
                # PE_num = s[2]
                origin_tile_analysis = tile_latency_analysis(self.simconfig_path,origin_read_row,origin_read_col,origin_indata,origin_rdata,self.inprecision,self.pes_per_tile)
                self.origin_total_latency = origin_tile_analysis.tile_latency +self.origin_total_latency
                # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
                self.origin_layer_latency.append(origin_tile_analysis.tile_latency)

                optimize_tile_analysis = tile_latency_analysis(self.simconfig_path,optimize_read_row,optimize_read_col,optimize_indata,optimize_rdata,self.inprecision,self.pes_per_tile)
                self.optimize_total_latency = optimize_tile_analysis.tile_latency +self.optimize_total_latency
                # self.layer_latency_pareto.append([tile_analysis.tile_latency,PE_num])
                self.optimize_layer_latency.append(optimize_tile_analysis.tile_latency)
              
        return self.origin_total_latency,self.origin_layer_latency,self.optimize_total_latency,self.optimize_layer_latency,dram


    
def generate_noc_nop_records(mapping_report, NN, inprecision, package_id=0):
    """
    根据映射结果生成NoC和NoP传输记录
    
    参数:
        mapping_report: 映射返回的结果字典
        NN: 神经网络参数，二维数组格式 [层数][参数]
        inprecision: 输入精度（位宽）
        package_id: Package编号，默认为0
    
    返回:
        noc_records: NoC传输记录字典 {chiplet_id: [[源tile列表], [目的tile列表], 数据量], ...}
        nop_records: NoP传输记录字典 {package_id: [[源chiplet编号], [目的chiplet编号], 数据传输量], ...}
    """
    
    # 从映射报告中获取层映射信息
    layer_mappings = mapping_report['layer_mappings']
    
    # 按层ID排序以确保顺序
    sorted_mappings = sorted(layer_mappings, key=lambda x: x['layer_id'])
    
    # 初始化记录字典
    noc_records = {}
    nop_records = {package_id: []}
    
    # 遍历层映射，为每对相邻层生成传输记录
    for i in range(len(sorted_mappings) - 1):
        current_layer = sorted_mappings[i]
        next_layer = sorted_mappings[i + 1]
        
        # 获取当前层和下一层的映射信息
        src_chiplet_id = current_layer['chiplet_id']
        src_tile_list = current_layer['tile_indices']
        
        dst_chiplet_id = next_layer['chiplet_id']
        dst_tile_list = next_layer['tile_indices']
        
        # 计算数据传输量：下一层的输入特征图大小 × 精度
        next_layer_id = next_layer['layer_id']
        if 0 <= next_layer_id < len(NN):
            next_layer_nn = NN[next_layer_id]
            if len(next_layer_nn) >= 3:
                # 输入特征图大小 = 高 × 宽 × 输入通道数
                input_height = next_layer_nn[0]
                input_width = next_layer_nn[1]
                input_channels = next_layer_nn[2]
                
                # 数据传输量（位）= 输入特征图大小 × 精度
                data_volume = input_height * input_width * input_channels * inprecision
            else:
                # 如果NN格式不符合预期，使用默认值
                data_volume = 0
                print(f"警告: 第{next_layer_id}层NN参数不足3个")
        else:
            # 如果层ID超出范围，使用默认值
            data_volume = 0
            print(f"警告: 第{next_layer_id}层超出NN参数范围")
        
        # 生成传输记录
        if src_chiplet_id == dst_chiplet_id:
            # 同一chiplet内传输 -> NoC
            if src_chiplet_id not in noc_records:
                noc_records[src_chiplet_id] = []
            record = [src_tile_list, dst_tile_list, data_volume]
            noc_records[src_chiplet_id].append(record)
        else:
            # 不同chiplet间传输 -> NoP (Package级别)
            # 注意: NoP记录中不包含tile信息，只包含chiplet级别信息
            nop_record = [[src_chiplet_id], [dst_chiplet_id], data_volume]
            nop_records[package_id].append(nop_record)
    
    return noc_records, nop_records
def generate_layer_mappings(nn_structure, layer_mappings_data, tiles_per_chiplet=16):
    """
    修正版：tile 0-15归属到对应chiplet，避免提前切换导致的归属错误
    
    参数:
        nn_structure: 神经网络结构参数列表
        layer_mappings_data: 每一层的映射数据列表
        tiles_per_chiplet: 每个chiplet的tile数量，默认16
    
    返回:
        dict: 修正后的映射记录字典
    """
    current_chiplet_id = 0  
    current_tile_idx = 0    
    layer_mappings = []     
    total_tiles_used = 0    
    
    for layer_id, (nn_layer, mapping_layer) in enumerate(zip(nn_structure, layer_mappings_data)):
        data_transfer_raw, compute_cycles_raw, resource_num, _ = mapping_layer
        normalize_base = mapping_layer[1]  
        
        tile_count = int(resource_num)
        tile_indices = []
        remaining_tiles_needed = tile_count
        
        # 记录分配前的chiplet_id（核心修正：先记录当前chiplet，再分配）
        assign_chiplet_id = current_chiplet_id
        
        while remaining_tiles_needed > 0:
            tiles_remaining_in_chiplet = tiles_per_chiplet - current_tile_idx
            tiles_to_assign = min(remaining_tiles_needed, tiles_remaining_in_chiplet)
            
            tile_indices.extend([current_tile_idx + i for i in range(tiles_to_assign)])
            
            remaining_tiles_needed -= tiles_to_assign
            current_tile_idx += tiles_to_assign
            
            if current_tile_idx >= tiles_per_chiplet:
                current_chiplet_id += 1
                current_tile_idx = 0
        
        # 计算归一化值
        data_transfer = round(data_transfer_raw / normalize_base, 0)
        compute_cycles = round(compute_cycles_raw / normalize_base, 0)
        
        # 核心修正：使用分配前的chiplet_id，而非分配后的
        layer_info = {
            'layer_id': layer_id,
            'chiplet_id': assign_chiplet_id,  # 改用分配前的chiplet_id
            'segment_id': assign_chiplet_id,
            'tile_indices': tile_indices,
            'tile_count': tile_count,
            'arrays_needed': tile_count,
            'actual_arrays': tile_count,
            'efficiency': 100.0,
            'data_transfer': int(data_transfer),
            'compute_cycles': int(compute_cycles)
        }
        layer_mappings.append(layer_info)
        total_tiles_used += tile_count
    
    # 重新计算利用率
    total_chiplets_used = current_chiplet_id + (1 if current_tile_idx > 0 else 0)
    total_tiles_available = total_chiplets_used * tiles_per_chiplet
    utilization_rate = (total_tiles_used / total_tiles_available) * 100.0 if total_tiles_available > 0 else 0.0
    
    utilization = {
        'tiles_used': total_tiles_used,
        'tiles_available': total_tiles_available - total_tiles_used,
        'utilization_rate': round(utilization_rate, 1)
    }
    
    return {'layer_mappings': layer_mappings, 'utilization': utilization}