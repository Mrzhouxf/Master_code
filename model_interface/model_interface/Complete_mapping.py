from __future__ import annotations
from model_interface.function import *
from Hardware_Model.Buffer import buffer
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.PE_latency import PE_latency_analysis
from model_interface.Layout_function import *
from model_interface.noc import *
from model_interface.nop import *
from Interconnect.nop_estimation import *
from Interconnect.noc_estimation import *


from copy import deepcopy
from typing import List, Dict, Any, Tuple
import math
import configparser as cp

class Complete_mapping:
    def __init__(self, SimConfig_path, NN, inprecision: int):
        self.simconfig_path = SimConfig_path
        accelerator_arch = cp.ConfigParser()
        accelerator_arch.read(SimConfig_path, encoding='UTF-8')
        
        self.NN = NN  # 二维数组，每一行表示一层的参数
        self.inprecision = inprecision  # 输入精度
        
        # 读取硬件架构配置
        self.chiplet_num = list(map(int, accelerator_arch.get('Package level', 'Chiplet_Num').split(',')))
        self.tile_num = list(map(int, accelerator_arch.get('Architecture level', 'Tile_Num').split(',')))
        self.pe_num = list(map(int, accelerator_arch.get('Tile level', 'PE_Num').split(',')))
        self.array = list(map(int, accelerator_arch.get('Crossbar level', 'Xbar_Size').split(',')))
        self.array_num = int(accelerator_arch.get('Process element level', 'Group_Num'))
        
        # 计算硬件资源总量
        self.total_chiplets = self.chiplet_num[0] * self.chiplet_num[1]
        self.tiles_per_chiplet = self.tile_num[0] * self.tile_num[1]
        self.total_tiles = self.total_chiplets * self.tiles_per_chiplet
        self.pes_per_tile = self.pe_num[0] * self.pe_num[1]
        self.total_pes = self.total_tiles * self.pes_per_tile
        self.arrays_per_pe = self.array_num
        self.arrays_per_tile = self.pes_per_tile * self.arrays_per_pe
        self.arrays_per_chiplet = self.tiles_per_chiplet * self.arrays_per_tile
        
        # 初始化硬件资源状态
        self.hardware_status = {
            'chiplets': [{'tiles': [{'available': True, 'layer_id': -1, 'used_arrays': 0, 
                                    'parallel_id': -1}  # 新增：并行ID
                                   for _ in range(self.tiles_per_chiplet)]} 
                        for _ in range(self.total_chiplets)],
            'layer_mappings': []  # 存储每层的映射结果
        }
        self.squetial_total_latency = 0
        self.squetial_layer_latency = []
        self.uniform_total_latency = 0
        self.uniform_layer_latency = []
        
    def parse_mapping_list(self, mapping_data: List[List[float]]) -> List[Dict]:
        """解析映射列表数据（新格式：二维数组）"""
        mappings = []
        for i, row in enumerate(mapping_data):
            if len(row) >= 7:
                mapping = {
                    'layer_id': i,
                    'data_transfer': float(row[0]),
                    'compute_cycles': float(row[1]),
                    'array_count': int(row[2]),  # 使用阵列个数
                    'mapping_scheme': list(row[3:7]),  # 映射方案
                    'chiplet_id': -1,
                    'tile_ids': [],
                    'parallel_count': 1,  # 默认并行度为1
                    'parallel_tiles': []   # 并行分配的所有tile
                }
                mappings.append(mapping)
        return mappings
    
    def sequential_parallel_mapping(self, mapping_list: List[List[float]]) -> Dict[str, Any]:
        """
        支持并行操作的顺序映射算法
        
        规则:
        1. 不能多个层映射在一个tile里面（一个tile只能映射一层）
        2. 一层映射尽量不要跨chiplet，除非这一层使用的阵列个数就已经超过了一个chiplet的阵列个数
        3. 按照顺序填充chiplet
        4. 当资源空闲时，执行并行操作：
           a. 当总资源是所需资源的整数倍时，直接将每一层的使用资源翻倍
           b. 当有剩余资源时，按照计算周期大小依次分配剩余资源块
        5. 一层的映射不要跨chiplet（除非这一层所使用的资源大于整个chiplet的tile个数）
        """
        # 解析映射列表
        layer_mappings = self.parse_mapping_list(mapping_list)
        
        # 清空之前的映射
        for chiplet in self.hardware_status['chiplets']:
            for tile in chiplet['tiles']:
                tile['available'] = True
                tile['layer_id'] = -1
                tile['used_arrays'] = 0
                tile['parallel_id'] = -1
        self.hardware_status['layer_mappings'] = []
        
        # 1. 计算每层基本需要的tile数量
        for layer in layer_mappings:
            layer['basic_tiles_needed'] = math.ceil(layer['array_count'] / self.arrays_per_tile)
        
        # 2. 计算总共需要的tile数量（基本需求）
        total_basic_tiles_needed = sum(layer['basic_tiles_needed'] for layer in layer_mappings)
        
        print(f"总tile数: {self.total_tiles}")
        print(f"基本需求tile数: {total_basic_tiles_needed}")
        
        # 3. 计算倍数关系和剩余资源
        if total_basic_tiles_needed > 0:
            multiple = self.total_tiles // total_basic_tiles_needed
            remaining_tiles = self.total_tiles % total_basic_tiles_needed
        else:
            multiple = 0
            remaining_tiles = 0
        
        print(f"资源倍数: {multiple}")
        print(f"剩余tile数: {remaining_tiles}")
        
        # 4. 确定每层的并行度
        if multiple >= 2:
            # 情况a: 资源是需求的整数倍，每层资源翻倍
            print("资源是需求的整数倍，执行资源翻倍分配")
            for layer in layer_mappings:
                layer['parallel_count'] = multiple
                layer['total_tiles_needed'] = layer['basic_tiles_needed'] * multiple
        else:
            # 情况b: 基本分配，然后分配剩余资源
            print("执行基本分配+剩余资源分配")
            for layer in layer_mappings:
                layer['parallel_count'] = 1
                layer['total_tiles_needed'] = layer['basic_tiles_needed']
            
            # 按照计算周期大小排序，分配剩余资源
            if remaining_tiles > 0:
                print(f"分配{remaining_tiles}个剩余tile资源")
                # 按计算周期降序排序
                sorted_layers = sorted(layer_mappings, 
                                      key=lambda x: x['compute_cycles'], 
                                      reverse=True)
                
                # 为计算周期最大的层分配剩余资源
                for layer in sorted_layers:
                    if remaining_tiles <= 0:
                        break
                    
                    # 检查该层是否能接受更多资源（不跨chiplet限制）
                    max_tiles_in_chiplet = min(self.tiles_per_chiplet, 
                                             layer['total_tiles_needed'] + remaining_tiles)
                    
                    if layer['total_tiles_needed'] < max_tiles_in_chiplet:
                        # 分配一个额外tile
                        layer['total_tiles_needed'] += 1
                        layer['parallel_count'] += 1
                        remaining_tiles -= 1
                        print(f"为第{layer['layer_id']}层分配一个额外tile，现在共需{layer['total_tiles_needed']}个tile")
        
        # 5. 执行顺序映射（基于新的tile需求）
        current_chiplet = 0
        current_tile = 0
        
        for layer in layer_mappings:
            arrays_needed = layer['array_count'] * layer['parallel_count']
            tiles_needed = layer['total_tiles_needed']
            
            print(f"\n映射第{layer['layer_id']}层:")
            print(f"  需要阵列: {layer['array_count']} × {layer['parallel_count']} = {arrays_needed}")
            print(f"  需要tile: {tiles_needed}")
            
            if tiles_needed <= self.tiles_per_chiplet:
                # 尝试在同一个chiplet内分配
                chiplet_id, tile_ids = self._allocate_with_parallelism(
                    tiles_needed, arrays_needed, current_chiplet, current_tile, 
                    layer['layer_id'], layer['parallel_count']
                )
                
                if chiplet_id >= 0:
                    # 分配成功
                    layer['chiplet_id'] = chiplet_id
                    layer['tile_ids'] = tile_ids
                    layer['parallel_tiles'] = self._organize_parallel_tiles(
                        tile_ids, layer['parallel_count'], layer['basic_tiles_needed']
                    )
                    
                    # 更新当前位置
                    if chiplet_id == current_chiplet:
                        current_tile = max(tile_ids) + 1
                        if current_tile >= self.tiles_per_chiplet:
                            current_chiplet += 1
                            current_tile = 0
                    else:
                        current_chiplet = chiplet_id
                        current_tile = max(tile_ids) + 1
                else:
                    # 当前chiplet空间不足，查找其他chiplet
                    chiplet_id, tile_ids = self._find_chiplet_with_parallelism(
                        tiles_needed, arrays_needed, layer['layer_id'], layer['parallel_count']
                    )
                    
                    if chiplet_id >= 0:
                        layer['chiplet_id'] = chiplet_id
                        layer['tile_ids'] = tile_ids
                        layer['parallel_tiles'] = self._organize_parallel_tiles(
                            tile_ids, layer['parallel_count'], layer['basic_tiles_needed']
                        )
                        
                        current_chiplet = chiplet_id
                        current_tile = max(tile_ids) + 1
                    else:
                        print(f"警告: 第{layer['layer_id']}层无法映射，需要的tile数{tiles_needed}超过硬件容量")
                        continue
            else:
                # 需要的tile数超过一个chiplet容量，必须跨chiplet
                print(f"第{layer['layer_id']}层需要跨chiplet分配")
                chiplet_ids, tile_ids = self._allocate_across_chiplets_with_parallelism(
                    tiles_needed, arrays_needed, layer['layer_id'], layer['parallel_count']
                )
                
                if chiplet_ids:
                    layer['chiplet_id'] = chiplet_ids[0]
                    layer['tile_ids'] = tile_ids
                    layer['parallel_tiles'] = self._organize_parallel_tiles(
                        tile_ids, layer['parallel_count'], layer['basic_tiles_needed']
                    )
                else:
                    print(f"警告: 第{layer['layer_id']}层无法映射，需要的tile数{tiles_needed}超过总硬件容量")
                    continue
            
            # 保存映射结果
            mapping_result = {
                'layer_id': layer['layer_id'],
                'chiplet_id': layer['chiplet_id'],
                'tile_ids': layer['tile_ids'],
                'parallel_tiles': layer['parallel_tiles'],
                'parallel_count': layer['parallel_count'],
                'arrays_needed': arrays_needed,
                'actual_arrays': len(layer['tile_ids']) * self.arrays_per_tile,
                'data_transfer': layer['data_transfer'],
                'compute_cycles': layer['compute_cycles'],
                'mapping_scheme': layer['mapping_scheme']
            }
            self.hardware_status['layer_mappings'].append(mapping_result)
        
        return self._generate_parallel_mapping_report(layer_mappings)
    
    def _allocate_with_parallelism(self, tiles_needed: int, arrays_needed: int, 
                                  chiplet_id: int, start_tile: int, 
                                  layer_id: int, parallel_count: int) -> Tuple[int, List[int]]:
        """考虑并行度的分配"""
        if chiplet_id >= self.total_chiplets:
            return -1, []
        
        chiplet = self.hardware_status['chiplets'][chiplet_id]
        
        # 检查是否有足够的连续tile
        available_tiles = []
        
        for i in range(start_tile, self.tiles_per_chiplet):
            if chiplet['tiles'][i]['available']:
                available_tiles.append(i)
                if len(available_tiles) >= tiles_needed:
                    # 找到了足够的tile
                    for tile_idx, tile_id in enumerate(available_tiles):
                        chiplet['tiles'][tile_id]['available'] = False
                        chiplet['tiles'][tile_id]['layer_id'] = layer_id
                        chiplet['tiles'][tile_id]['parallel_id'] = tile_idx // (tiles_needed // parallel_count)
                        chiplet['tiles'][tile_id]['used_arrays'] = min(
                            self.arrays_per_tile, 
                            arrays_needed - (tile_idx * self.arrays_per_tile)
                        )
                    return chiplet_id, available_tiles
        
        return -1, []
    
    def _find_chiplet_with_parallelism(self, tiles_needed: int, arrays_needed: int,
                                      layer_id: int, parallel_count: int) -> Tuple[int, List[int]]:
        """查找可用的chiplet分配资源（考虑并行度）"""
        for chiplet_id in range(self.total_chiplets):
            chiplet = self.hardware_status['chiplets'][chiplet_id]
            
            # 检查chiplet是否有足够的空闲tile
            available_tiles = []
            for i in range(self.tiles_per_chiplet):
                if chiplet['tiles'][i]['available']:
                    available_tiles.append(i)
                    if len(available_tiles) >= tiles_needed:
                        # 找到了足够的tile
                        for tile_idx, tile_id in enumerate(available_tiles):
                            chiplet['tiles'][tile_id]['available'] = False
                            chiplet['tiles'][tile_id]['layer_id'] = layer_id
                            chiplet['tiles'][tile_id]['parallel_id'] = tile_idx // (tiles_needed // parallel_count)
                            chiplet['tiles'][tile_id]['used_arrays'] = min(
                                self.arrays_per_tile, 
                                arrays_needed - (tile_idx * self.arrays_per_tile)
                            )
                        return chiplet_id, available_tiles
        
        return -1, []
    
    def _allocate_across_chiplets_with_parallelism(self, tiles_needed: int, arrays_needed: int,
                                                  layer_id: int, parallel_count: int) -> Tuple[List[int], List[int]]:
        """跨多个chiplet分配资源（考虑并行度）"""
        chiplet_ids = []
        tile_ids = []
        arrays_allocated = 0
        
        for chiplet_id in range(self.total_chiplets):
            chiplet = self.hardware_status['chiplets'][chiplet_id]
            
            # 使用当前chiplet的所有可用tile
            for tile_id in range(self.tiles_per_chiplet):
                if chiplet['tiles'][tile_id]['available']:
                    chiplet_ids.append(chiplet_id)
                    tile_ids.append(tile_id)
                    
                    # 标记tile为已使用
                    chiplet['tiles'][tile_id]['available'] = False
                    chiplet['tiles'][tile_id]['layer_id'] = layer_id
                    chiplet['tiles'][tile_id]['parallel_id'] = len(tile_ids) // (tiles_needed // parallel_count)
                    chiplet['tiles'][tile_id]['used_arrays'] = min(
                        self.arrays_per_tile, 
                        arrays_needed - arrays_allocated
                    )
                    
                    arrays_allocated += self.arrays_per_tile
                    if len(tile_ids) >= tiles_needed:
                        return chiplet_ids, tile_ids
        
        return chiplet_ids, tile_ids
    
    def _organize_parallel_tiles(self, tile_ids: List[int], parallel_count: int, 
                                basic_tiles_per_copy: int) -> List[List[int]]:
        """组织并行分配的tile，按并行副本分组"""
        parallel_tiles = []
        
        if parallel_count == 1:
            # 没有并行，所有tile属于一个副本
            parallel_tiles.append(tile_ids)
        else:
            # 将tile按并行副本分组
            tiles_per_copy = basic_tiles_per_copy
            for i in range(parallel_count):
                start_idx = i * tiles_per_copy
                end_idx = start_idx + tiles_per_copy
                if end_idx <= len(tile_ids):
                    parallel_tiles.append(tile_ids[start_idx:end_idx])
                else:
                    # 如果tile数量不足，用剩余tile填充最后一个副本
                    parallel_tiles.append(tile_ids[start_idx:])
        
        return parallel_tiles
    
    def _generate_parallel_mapping_report(self, layer_mappings: List[Dict]) -> Dict[str, Any]:
        """生成并行映射报告"""
        report = {
            'hardware_config': {
                'total_chiplets': self.total_chiplets,
                'tiles_per_chiplet': self.tiles_per_chiplet,
                'total_tiles': self.total_tiles,
                'arrays_per_tile': self.arrays_per_tile,
                'arrays_per_chiplet': self.arrays_per_chiplet
            },
            'parallel_analysis': {
                'total_basic_tiles_needed': sum(layer.get('basic_tiles_needed', 0) for layer in layer_mappings),
                'total_allocated_tiles': sum(layer.get('total_tiles_needed', 0) for layer in layer_mappings),
                'average_parallelism': sum(layer.get('parallel_count', 1) for layer in layer_mappings) / len(layer_mappings) if layer_mappings else 0
            },
            'layer_mappings': [],
            'utilization': {
                'tiles_used': 0,
                'tiles_available': self.total_tiles,
                'utilization_rate': 0.0
            }
        }
        
        # 统计tile使用情况
        used_tiles = 0
        for chiplet in self.hardware_status['chiplets']:
            for tile in chiplet['tiles']:
                if not tile['available']:
                    used_tiles += 1
        
        # 生成每层的映射信息
        for mapping in self.hardware_status['layer_mappings']:
            tile_indices = mapping['tile_ids']
            parallel_groups = []
            
            # 将并行分组的tile转换为索引
            for group in mapping.get('parallel_tiles', []):
                parallel_groups.append(group)
            
            layer_mapping = {
                'layer_id': mapping['layer_id'],
                'chiplet_id': mapping['chiplet_id'],
                'tile_indices': tile_indices,
                'tile_count': len(tile_indices),
                'parallel_count': mapping['parallel_count'],
                'parallel_groups': parallel_groups,
                'arrays_needed': mapping['arrays_needed'],
                'actual_arrays': mapping['actual_arrays'],
                'efficiency': min(100.0, mapping['arrays_needed'] / mapping['actual_arrays'] * 100) if mapping['actual_arrays'] > 0 else 0,
                'data_transfer': mapping['data_transfer'],
                'compute_cycles': mapping['compute_cycles'],
                'speedup_ratio': mapping['parallel_count']  # 加速比等于并行度
            }
            report['layer_mappings'].append(layer_mapping)
        
        # 计算利用率
        report['utilization']['tiles_used'] = used_tiles
        report['utilization']['tiles_available'] = self.total_tiles - used_tiles
        report['utilization']['utilization_rate'] = used_tiles / self.total_tiles * 100 if self.total_tiles > 0 else 0
        
        return report
    
    def print_parallel_mapping_summary(self, report: Dict[str, Any] = None):
        """打印并行映射摘要"""
        if report is None:
            print("请先调用sequential_parallel_mapping方法获取报告")
            return
        
        print("=" * 80)
        print("硬件架构配置:")
        print("=" * 80)
        hw = report['hardware_config']
        print(f"Chiplet总数: {hw['total_chiplets']}")
        print(f"每个Chiplet的Tile数: {hw['tiles_per_chiplet']}")
        print(f"Tile总数: {hw['total_tiles']}")
        print(f"每个Tile的阵列数: {hw['arrays_per_tile']}")
        print(f"每个Chiplet的阵列数: {hw['arrays_per_chiplet']}")
        
        print("\n" + "=" * 80)
        print("并行分析:")
        print("=" * 80)
        parallel = report['parallel_analysis']
        print(f"基本需求Tile数: {parallel['total_basic_tiles_needed']}")
        print(f"实际分配Tile数: {parallel['total_allocated_tiles']}")
        print(f"平均并行度: {parallel['average_parallelism']:.2f}")
        
        print("\n" + "=" * 80)
        print("并行映射结果:")
        print("=" * 80)
        for mapping in report['layer_mappings']:
            if mapping['chiplet_id'] >= 0:
                tiles_str = ', '.join(str(t) for t in mapping['tile_indices'])
                
                print(f"\n第{mapping['layer_id']}层:")
                print(f"  Chiplet: {mapping['chiplet_id']}, 并行度: {mapping['parallel_count']}")
                print(f"  总Tiles: [{tiles_str}]")
                print(f"  计算周期: {mapping['compute_cycles']}, 数据传输: {mapping['data_transfer']}")
                
                # 显示并行分组
                for i, group in enumerate(mapping['parallel_groups']):
                    group_str = ', '.join(str(t) for t in group)
                    print(f"  并行副本{i+1}: Tiles [{group_str}]")
                
                print(f"  需要的阵列: {mapping['arrays_needed']}, 实际分配: {mapping['actual_arrays']}")
                print(f"  映射效率: {mapping['efficiency']:.1f}%, 加速比: {mapping['speedup_ratio']:.1f}x")
            else:
                print(f"第{mapping['layer_id']}层: 未分配")
        
        print("\n" + "=" * 80)
        print("资源利用率:")
        print("=" * 80)
        util = report['utilization']
        print(f"已使用Tile数: {util['tiles_used']}")
        print(f"可用Tile数: {util['tiles_available']}")
        print(f"利用率: {util['utilization_rate']:.2f}%")
    
    def visualize_parallel_mapping(self, report: Dict[str, Any] = None):
        """可视化并行映射结果"""
        if report is None:
            print("请先调用sequential_parallel_mapping方法获取报告")
            return
        
        print("\n" + "=" * 80)
        print("并行映射可视化:")
        print("=" * 80)
        
        # 为每个chiplet创建一个可视化网格
        for chiplet_id in range(self.total_chiplets):
            print(f"\nChiplet {chiplet_id}:")
            print("-" * 40)
            
            # 创建tile网格
            rows = self.tile_num[0]
            cols = self.tile_num[1]
            
            for r in range(rows):
                row_str = ""
                for c in range(cols):
                    tile_id = r * cols + c
                    tile = self.hardware_status['chiplets'][chiplet_id]['tiles'][tile_id]
                    
                    if not tile['available']:
                        layer_id = tile['layer_id']
                        parallel_id = tile['parallel_id']
                        # 使用简写显示层ID和并行ID
                        if parallel_id >= 0:
                            row_str += f" L{layer_id:02d}P{parallel_id} "
                        else:
                            row_str += f" L{layer_id:02d} "
                    else:
                        row_str += " [ ] "
                print(row_str)
        
        # 显示图例
        print("\n图例:")
        print("  L## : 第##层占用的Tile")
        print("  L##P#: 第##层的第#个并行副本")
        print("  [ ] : 空闲Tile")
    # 均匀映射函数
    def uniform_segmentation_mapping(self, mapping_list: List[List[float]]) -> Dict[str, Any]:
        """
        均匀分段映射函数
        
        规则:
        1. 将神经网络分成total_chiplets段
        2. 每一段映射到一个chiplet上
        3. 每一段的数据传输量总和尽量相近
        4. 每一段使用的资源（tile数）不超过一个chiplet的总资源
        5. 每一段内的资源分配尽量均衡
        """
        # 解析映射列表
        layer_mappings = self.parse_mapping_list(mapping_list)
        
        # 清空之前的映射
        for chiplet in self.hardware_status['chiplets']:
            for tile in chiplet['tiles']:
                tile['available'] = True
                tile['layer_id'] = -1
                tile['used_arrays'] = 0
        self.hardware_status['layer_mappings'] = []
        
        # 1. 计算每层基本需要的tile数量
        for layer in layer_mappings:
            layer['tiles_needed'] = math.ceil(layer['array_count'] / self.arrays_per_tile)
        
        # 2. 使用优化算法进行分段
        segments = self._optimal_segmentation(layer_mappings)
        
        print("=" * 60)
        print("均匀分段映射分析:")
        print("=" * 60)
        print(f"神经网络总层数: {len(layer_mappings)}")
        print(f"Chiplet数量: {self.total_chiplets}")
        print(f"分段数量: {len(segments)}")
        print(f"每个Chiplet的Tile数: {self.tiles_per_chiplet}")
        
        # 3. 为每段分配chiplet并进行映射
        for segment_id, segment in enumerate(segments):
            chiplet_id = segment_id  # 每段对应一个chiplet
            
            # 计算该段需要的总tile数
            segment_tiles_needed = sum(layer['tiles_needed'] for layer in segment)
            
            if segment_tiles_needed > self.tiles_per_chiplet:
                print(f"警告: 第{segment_id}段需要的tile数({segment_tiles_needed})超过chiplet容量({self.tiles_per_chiplet})")
                # 尝试在该chiplet内优化分配
                success = self._allocate_segment_with_optimization(segment, chiplet_id, segment_id)
                if not success:
                    print(f"错误: 第{segment_id}段无法在chiplet {chiplet_id}上完全映射")
                    continue
            else:
                # 正常分配
                success = self._allocate_segment(segment, chiplet_id, segment_id)
                if not success:
                    print(f"警告: 第{segment_id}段在chiplet {chiplet_id}上分配失败")
                    continue
            
            # 标记段的层信息
            for layer in segment:
                layer['segment_id'] = segment_id
                layer['chiplet_id'] = chiplet_id
        
        # 4. 生成报告
        return self._generate_uniform_segmentation_report(layer_mappings, segments)
    
    def _optimal_segmentation(self, layers: List[Dict]) -> List[List[Dict]]:
        """
        使用动态规划寻找最优分段方案
        
        目标: 
        1. 每段的数据传输量总和尽量相近
        2. 每段的tile总数不超过tiles_per_chiplet
        3. 每段的资源使用尽量均衡
        """
        n_layers = len(layers)
        n_segments = self.total_chiplets
        
        print(f"\n进行最优分段 (层数={n_layers}, 段数={n_segments}):")
        
        # 如果层数少于段数，每段一层
        if n_layers <= n_segments:
            segments = [[layer] for layer in layers]
            # 补充空段
            for _ in range(n_segments - n_layers):
                segments.append([])
            return segments
        
        # 计算前缀和，用于快速计算区间和
        data_prefix = [0] * (n_layers + 1)
        tile_prefix = [0] * (n_layers + 1)
        
        for i, layer in enumerate(layers):
            data_prefix[i+1] = data_prefix[i] + layer['data_transfer']
            tile_prefix[i+1] = tile_prefix[i] + layer['tiles_needed']
        
        total_data = data_prefix[n_layers]
        target_data_per_segment = total_data / n_segments
        
        print(f"总数据传输量: {total_data:.2f}")
        print(f"目标每段数据传输量: {target_data_per_segment:.2f}")
        
        # 方法1: 贪心算法 + 回溯调整
        segments = self._greedy_segmentation(layers, n_segments, target_data_per_segment)
        
        # 检查分段是否有效
        valid = True
        for seg_id, segment in enumerate(segments):
            seg_tiles = sum(layer['tiles_needed'] for layer in segment)
            if seg_tiles > self.tiles_per_chiplet:
                print(f"警告: 段{seg_id}的tile数({seg_tiles})超过chiplet容量")
                valid = False
        
        if not valid:
            print("贪心算法结果无效，尝试优化调整...")
            segments = self._adjust_segmentation(segments, layers, n_segments)
        
        return segments
    
    def _greedy_segmentation(self, layers: List[Dict], n_segments: int, 
                            target_data_per_segment: float) -> List[List[Dict]]:
        """贪心算法进行初步分段"""
        segments = []
        current_segment = []
        current_data = 0
        current_tiles = 0
        segment_id = 0
        
        for layer in layers:
            layer_tiles = layer['tiles_needed']
            layer_data = layer['data_transfer']
            
            # 如果当前段为空，直接加入
            if not current_segment:
                current_segment.append(layer)
                current_data += layer_data
                current_tiles += layer_tiles
            else:
                # 检查加入该层是否会使段的数据传输量接近目标值
                new_data = current_data + layer_data
                data_ratio = new_data / target_data_per_segment
                
                # 检查资源限制
                new_tiles = current_tiles + layer_tiles
                
                if (data_ratio <= 1.3 and new_tiles <= self.tiles_per_chiplet and 
                    segment_id < n_segments - 1):
                    # 可以加入当前段
                    current_segment.append(layer)
                    current_data = new_data
                    current_tiles = new_tiles
                else:
                    # 开始新的一段
                    segments.append(current_segment)
                    segment_id += 1
                    current_segment = [layer]
                    current_data = layer_data
                    current_tiles = layer_tiles
        
        # 添加最后一段
        if current_segment:
            segments.append(current_segment)
        
        # 确保分段数量正确
        if len(segments) > n_segments:
            # 合并最后几段
            while len(segments) > n_segments:
                last_segment = segments.pop()
                segments[-1].extend(last_segment)
        elif len(segments) < n_segments:
            # 拆分最大的一段
            while len(segments) < n_segments:
                # 找到最大的段
                max_idx = max(range(len(segments)), 
                             key=lambda i: sum(l['tiles_needed'] for l in segments[i]))
                max_segment = segments[max_idx]
                
                # 找到最佳分割点
                best_split = self._find_best_split_point(max_segment, target_data_per_segment)
                
                if best_split > 0:
                    # 分割为两段
                    segments[max_idx] = max_segment[:best_split]
                    segments.insert(max_idx + 1, max_segment[best_split:])
                else:
                    # 无法分割，补充空段
                    segments.append([])
        
        # 打印分段结果
        print("\n贪心算法分段结果:")
        for seg_id, segment in enumerate(segments):
            seg_data = sum(layer['data_transfer'] for layer in segment)
            seg_tiles = sum(layer['tiles_needed'] for layer in segment)
            seg_layers = [layer['layer_id'] for layer in segment]
            print(f"  段{seg_id}: {len(segment)}层, 数据传输量={seg_data:.2f}, tile数={seg_tiles}, 层ID={seg_layers}")
        
        return segments
    
    def _find_best_split_point(self, segment: List[Dict], target_data: float) -> int:
        """在段中找到最佳分割点"""
        if len(segment) <= 1:
            return 0
        
        best_split = 0
        min_diff = float('inf')
        
        for split_point in range(1, len(segment)):
            first_half = segment[:split_point]
            second_half = segment[split_point:]
            
            first_data = sum(layer['data_transfer'] for layer in first_half)
            second_data = sum(layer['data_transfer'] for layer in second_half)
            
            # 计算两段数据量与目标值的差异
            diff = abs(first_data - target_data) + abs(second_data - target_data)
            
            if diff < min_diff:
                min_diff = diff
                best_split = split_point
        
        return best_split
    
    def _adjust_segmentation(self, segments: List[List[Dict]], all_layers: List[Dict],
                           n_segments: int) -> List[List[Dict]]:
        """调整分段以满足资源约束"""
        print("开始调整分段以满足资源约束...")
        
        # 展平所有段
        all_segment_layers = []
        for segment in segments:
            all_segment_layers.extend(segment)
        
        # 重新构建层列表，保持原始顺序
        layer_dict = {layer['layer_id']: layer for layer in all_layers}
        ordered_layers = [layer_dict[layer['layer_id']] for layer in all_segment_layers 
                         if layer['layer_id'] in layer_dict]
        
        # 使用更保守的贪心算法重新分段
        adjusted_segments = []
        current_segment = []
        current_tiles = 0
        
        for layer in ordered_layers:
            layer_tiles = layer['tiles_needed']
            
            # 检查能否加入当前段
            if current_tiles + layer_tiles <= self.tiles_per_chiplet:
                current_segment.append(layer)
                current_tiles += layer_tiles
            else:
                # 开始新的一段
                adjusted_segments.append(current_segment)
                current_segment = [layer]
                current_tiles = layer_tiles
        
        # 添加最后一段
        if current_segment:
            adjusted_segments.append(current_segment)
        
        # 确保分段数量正确
        if len(adjusted_segments) > n_segments:
            # 合并小的段
            while len(adjusted_segments) > n_segments:
                # 找到最小的段
                min_idx = min(range(len(adjusted_segments)), 
                             key=lambda i: sum(l['tiles_needed'] for l in adjusted_segments[i]))
                min_segment = adjusted_segments.pop(min_idx)
                
                # 找到最合适的段合并
                best_target = -1
                best_combined_tiles = float('inf')
                
                for target_idx in range(len(adjusted_segments)):
                    target_segment = adjusted_segments[target_idx]
                    combined_tiles = (sum(l['tiles_needed'] for l in min_segment) + 
                                     sum(l['tiles_needed'] for l in target_segment))
                    
                    if combined_tiles <= self.tiles_per_chiplet and combined_tiles < best_combined_tiles:
                        best_target = target_idx
                        best_combined_tiles = combined_tiles
                
                if best_target >= 0:
                    adjusted_segments[best_target].extend(min_segment)
                else:
                    # 无法合并，放弃最小的段
                    print(f"警告: 无法合并段，放弃{len(min_segment)}层")
        
        # 确保有足够的分段
        while len(adjusted_segments) < n_segments:
            adjusted_segments.append([])
        
        return adjusted_segments
    
    def _allocate_segment(self, segment: List[Dict], chiplet_id: int, 
                         segment_id: int) -> bool:
        """在指定chiplet上分配一个段"""
        if chiplet_id >= self.total_chiplets:
            return False
        
        chiplet = self.hardware_status['chiplets'][chiplet_id]
        current_tile = 0
        
        for layer in segment:
            # 计算该层需要的tile数
            arrays_needed = layer['array_count']
            tiles_needed = math.ceil(arrays_needed / self.arrays_per_tile)
            
            # 查找连续可用的tile
            tile_ids = []
            consecutive_count = 0
            start_tile = -1
            
            for i in range(current_tile, self.tiles_per_chiplet):
                if chiplet['tiles'][i]['available']:
                    if consecutive_count == 0:
                        start_tile = i
                    consecutive_count += 1
                    
                    if consecutive_count >= tiles_needed:
                        # 找到足够的连续tile
                        tile_ids = list(range(start_tile, start_tile + tiles_needed))
                        break
                else:
                    consecutive_count = 0
                    start_tile = -1
            
            # 如果没找到连续tile，从头开始找
            if not tile_ids:
                for i in range(0, self.tiles_per_chiplet):
                    if chiplet['tiles'][i]['available']:
                        if consecutive_count == 0:
                            start_tile = i
                        consecutive_count += 1
                        
                        if consecutive_count >= tiles_needed:
                            tile_ids = list(range(start_tile, start_tile + tiles_needed))
                            break
                    else:
                        consecutive_count = 0
                        start_tile = -1
            
            if not tile_ids:
                print(f"  段{segment_id}中第{layer['layer_id']}层无法在chiplet {chiplet_id}上分配{tiles_needed}个连续tile")
                return False
            
            # 标记tile为已使用
            for tile_id in tile_ids:
                chiplet['tiles'][tile_id]['available'] = False
                chiplet['tiles'][tile_id]['layer_id'] = layer['layer_id']
                chiplet['tiles'][tile_id]['used_arrays'] = min(
                    self.arrays_per_tile,
                    arrays_needed - (tile_id - start_tile) * self.arrays_per_tile
                )
            
            layer['chiplet_id'] = chiplet_id
            layer['tile_ids'] = tile_ids
            
            # 更新当前位置
            current_tile = tile_ids[-1] + 1
            if current_tile >= self.tiles_per_chiplet:
                current_tile = 0
            
            # 保存映射结果
            mapping_result = {
                'layer_id': layer['layer_id'],
                'chiplet_id': chiplet_id,
                'segment_id': segment_id,
                'tile_ids': tile_ids,
                'arrays_needed': arrays_needed,
                'actual_arrays': len(tile_ids) * self.arrays_per_tile,
                'data_transfer': layer['data_transfer'],
                'compute_cycles': layer['compute_cycles'],
                'mapping_scheme': layer['mapping_scheme']
            }
            self.hardware_status['layer_mappings'].append(mapping_result)
        
        return True
    
    def _allocate_segment_with_optimization(self, segment: List[Dict], chiplet_id: int,
                                          segment_id: int) -> bool:
        """优化分配段，处理资源紧张的情况"""
        if chiplet_id >= self.total_chiplets:
            return False
        
        chiplet = self.hardware_status['chiplets'][chiplet_id]
        
        # 先尝试正常分配
        if self._allocate_segment(segment, chiplet_id, segment_id):
            return True
        
        # 如果正常分配失败，尝试更灵活的分配策略
        print(f"对段{segment_id}使用优化分配策略...")
        
        # 重新尝试，允许不连续的tile分配
        chiplet = self.hardware_status['chiplets'][chiplet_id]
        available_tiles = [i for i in range(self.tiles_per_chiplet) 
                          if chiplet['tiles'][i]['available']]
        
        if len(available_tiles) < sum(math.ceil(layer['array_count'] / self.arrays_per_tile) 
                                      for layer in segment):
            print(f"  chiplet {chiplet_id}上的可用tile不足")
            return False
        
        # 为每层分配tile（可能不连续）
        tile_allocations = []
        for layer in segment:
            arrays_needed = layer['array_count']
            tiles_needed = math.ceil(arrays_needed / self.arrays_per_tile)
            
            if len(available_tiles) < tiles_needed:
                print(f"  无法为层{layer['layer_id']}分配{tiles_needed}个tile")
                return False
            
            # 分配tile
            tile_ids = available_tiles[:tiles_needed]
            available_tiles = available_tiles[tiles_needed:]
            
            # 标记tile为已使用
            for tile_id in tile_ids:
                chiplet['tiles'][tile_id]['available'] = False
                chiplet['tiles'][tile_id]['layer_id'] = layer['layer_id']
                chiplet['tiles'][tile_id]['used_arrays'] = min(
                    self.arrays_per_tile,
                    arrays_needed
                )
            
            layer['chiplet_id'] = chiplet_id
            layer['tile_ids'] = tile_ids
            
            # 保存映射结果
            mapping_result = {
                'layer_id': layer['layer_id'],
                'chiplet_id': chiplet_id,
                'segment_id': segment_id,
                'tile_ids': tile_ids,
                'arrays_needed': arrays_needed,
                'actual_arrays': len(tile_ids) * self.arrays_per_tile,
                'data_transfer': layer['data_transfer'],
                'compute_cycles': layer['compute_cycles'],
                'mapping_scheme': layer['mapping_scheme']
            }
            self.hardware_status['layer_mappings'].append(mapping_result)
        
        return True
    
    def _generate_uniform_segmentation_report(self, layer_mappings: List[Dict],
                                            segments: List[List[Dict]]) -> Dict[str, Any]:
        """生成均匀分段映射报告"""
        # 统计资源使用情况
        used_tiles = 0
        for chiplet in self.hardware_status['chiplets']:
            for tile in chiplet['tiles']:
                if not tile['available']:
                    used_tiles += 1
        
        # 计算段统计信息
        segment_stats = []
        total_data = 0
        total_tiles = 0
        
        for seg_id, segment in enumerate(segments):
            seg_data = sum(layer['data_transfer'] for layer in segment)
            seg_tiles = sum(layer['tiles_needed'] for layer in segment)
            seg_layers = len(segment)
            
            segment_stats.append({
                'segment_id': seg_id,
                'chiplet_id': seg_id,
                'layer_count': seg_layers,
                'layers': [layer['layer_id'] for layer in segment],
                'total_data_transfer': seg_data,
                'total_tiles_needed': seg_tiles,
                'utilization_rate': seg_tiles / self.tiles_per_chiplet * 100
            })
            
            total_data += seg_data
            total_tiles += seg_tiles
        
        avg_data_per_segment = total_data / len(segments) if segments else 0
        
        # 计算数据均衡度
        data_balance_score = 0
        if avg_data_per_segment > 0:
            variance = sum((stat['total_data_transfer'] - avg_data_per_segment) ** 2 
                          for stat in segment_stats) / len(segment_stats)
            std_dev = math.sqrt(variance)
            data_balance_score = 100 * (1 - std_dev / avg_data_per_segment)
        
        # 计算资源均衡度
        avg_tiles_per_segment = total_tiles / len(segments) if segments else 0
        resource_balance_score = 0
        if avg_tiles_per_segment > 0:
            variance = sum((stat['total_tiles_needed'] - avg_tiles_per_segment) ** 2 
                          for stat in segment_stats) / len(segment_stats)
            std_dev = math.sqrt(variance)
            resource_balance_score = 100 * (1 - std_dev / avg_tiles_per_segment)
        
        report = {
            'hardware_config': {
                'total_chiplets': self.total_chiplets,
                'tiles_per_chiplet': self.tiles_per_chiplet,
                'total_tiles': self.total_tiles,
                'arrays_per_tile': self.arrays_per_tile,
                'arrays_per_chiplet': self.arrays_per_chiplet
            },
            'segmentation_info': {
                'total_segments': len(segments),
                'total_layers': len(layer_mappings),
                'avg_layers_per_segment': len(layer_mappings) / len(segments) if segments else 0,
                'total_data_transfer': total_data,
                'avg_data_per_segment': avg_data_per_segment,
                'data_balance_score': data_balance_score,
                'resource_balance_score': resource_balance_score
            },
            'segment_statistics': segment_stats,
            'layer_mappings': [],
            'utilization': {
                'tiles_used': used_tiles,
                'tiles_available': self.total_tiles - used_tiles,
                'utilization_rate': used_tiles / self.total_tiles * 100 if self.total_tiles > 0 else 0
            }
        }
        
        # 生成每层的映射信息
        for mapping in self.hardware_status['layer_mappings']:
            layer_mapping = {
                'layer_id': mapping['layer_id'],
                'chiplet_id': mapping['chiplet_id'],
                'segment_id': mapping['segment_id'],
                'tile_indices': mapping['tile_ids'],
                'tile_count': len(mapping['tile_ids']),
                'arrays_needed': mapping['arrays_needed'],
                'actual_arrays': mapping['actual_arrays'],
                'efficiency': min(100.0, mapping['arrays_needed'] / mapping['actual_arrays'] * 100) if mapping['actual_arrays'] > 0 else 0,
                'data_transfer': mapping['data_transfer'],
                'compute_cycles': mapping['compute_cycles']
            }
            report['layer_mappings'].append(layer_mapping)
        
        return report
    
    def print_uniform_segmentation_summary(self, report: Dict[str, Any] = None):
        """打印均匀分段映射摘要"""
        if report is None:
            print("请先调用uniform_segmentation_mapping方法获取报告")
            return
        
        print("\n" + "=" * 80)
        print("均匀分段映射结果:")
        print("=" * 80)
        
        # 硬件配置
        hw = report['hardware_config']
        print(f"硬件配置: {hw['total_chiplets']}个Chiplet, 每个{hw['tiles_per_chiplet']}个Tile")
        
        # 分段信息
        seg_info = report['segmentation_info']
        print(f"\n分段信息:")
        print(f"  总段数: {seg_info['total_segments']}")
        print(f"  总层数: {seg_info['total_layers']}")
        print(f"  每段平均层数: {seg_info['avg_layers_per_segment']:.1f}")
        print(f"  总数据传输量: {seg_info['total_data_transfer']:.2f}")
        print(f"  每段平均数据传输量: {seg_info['avg_data_per_segment']:.2f}")
        print(f"  数据均衡度: {seg_info['data_balance_score']:.1f}%")
        print(f"  资源均衡度: {seg_info['resource_balance_score']:.1f}%")
        
        # 段统计
        print(f"\n各段详细统计:")
        print("-" * 60)
        for stat in report['segment_statistics']:
            print(f"\n段{stat['segment_id']} (Chiplet {stat['chiplet_id']}):")
            print(f"  层数: {stat['layer_count']}")
            print(f"  层ID: {stat['layers']}")
            print(f"  数据传输量: {stat['total_data_transfer']:.2f}")
            print(f"  需要Tile数: {stat['total_tiles_needed']}")
            print(f"  资源利用率: {stat['utilization_rate']:.1f}%")
        
        # 层映射详情
        print(f"\n各层映射详情:")
        print("-" * 60)
        for mapping in report['layer_mappings']:
            tiles_str = ', '.join(str(t) for t in mapping['tile_indices'])
            print(f"第{mapping['layer_id']}层: 段{mapping['segment_id']}, Chiplet {mapping['chiplet_id']}, Tiles [{tiles_str}]")
            print(f"  数据传输: {mapping['data_transfer']:.1f}, 计算周期: {mapping['compute_cycles']:.1f}")
            print(f"  阵列需求: {mapping['arrays_needed']}, 实际分配: {mapping['actual_arrays']}")
            print(f"  映射效率: {mapping['efficiency']:.1f}%")
        
        # 资源利用率
        util = report['utilization']
        print(f"\n资源利用率:")
        print(f"  已使用Tile数: {util['tiles_used']}")
        print(f"  可用Tile数: {util['tiles_available']}")
        print(f"  总利用率: {util['utilization_rate']:.2f}%")
    
    def visualize_uniform_segmentation(self, report: Dict[str, Any] = None):
        """可视化均匀分段映射结果"""
        if report is None:
            print("请先调用uniform_segmentation_mapping方法获取报告")
            return
        
        print("\n" + "=" * 80)
        print("均匀分段映射可视化:")
        print("=" * 80)
        
        # 为每个chiplet创建一个可视化网格
        for chiplet_id in range(self.total_chiplets):
            print(f"\nChiplet {chiplet_id}:")
            print("-" * 40)
            
            # 查找该chiplet对应的段
            segment_id = chiplet_id
            segment_layers = []
            for stat in report['segment_statistics']:
                if stat['chiplet_id'] == chiplet_id:
                    segment_layers = stat['layers']
                    break
            
            # 创建tile网格
            rows = self.tile_num[0]
            cols = self.tile_num[1]
            
            for r in range(rows):
                row_str = ""
                for c in range(cols):
                    tile_id = r * cols + c
                    tile = self.hardware_status['chiplets'][chiplet_id]['tiles'][tile_id]
                    
                    if not tile['available']:
                        layer_id = tile['layer_id']
                        # 显示层ID和段ID
                        if layer_id in segment_layers:
                            row_str += f" L{layer_id:02d}S{segment_id} "
                        else:
                            row_str += f" L{layer_id:02d} "
                    else:
                        row_str += " [ ] "
                print(row_str)
        
        # 显示图例
        print("\n图例:")
        print("  L##S#: 第##层，属于第#段")
        print("  [ ] : 空闲Tile")

    def select_best_mapping_scheme_v2(
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
    
    def performance_sim_sequential(self,mappingdata,mode):
        if mode == "sequential":
            squeue_map = self.sequential_parallel_mapping(mappingdata)
            for i in range(len(squeue_map['layer_mappings'])):
                read_row = mappingdata[i][3]*mappingdata[i][4]*mappingdata[i][5]
                print(read_row)
                read_col = mappingdata[i][6]*(mappingdata[i][3]-self.NN[i][3]+1)**2
                print(read_col)
                indata = self.NN[i][0]*self.NN[i][1]*self.NN[i][2]
                read_data = mappingdata[i][0]
                tile_analysis = tile_latency_analysis(self.simconfig_path,read_row,read_col,indata,read_data,self.inprecision,self.pe_num[0]*self.pe_num[1])
                parallel_count = squeue_map['layer_mappings'][i]['parallel_count']
                self.squetial_total_latency = tile_analysis.tile_latency + self.squetial_total_latency
                self.squetial_layer_latency.append(tile_analysis.tile_latency)
        
        return self.squetial_total_latency
    def performance_sim_uniform(self,mappingdata,mode):
        print("uniform start")
        uniform_map = self.uniform_segmentation_mapping(mappingdata)
        print(uniform_map)
        print(len(uniform_map['layer_mappings']))
        for i in range(len(self.NN)):
            print(i)
            print(mappingdata[i][3])
            read_row = mappingdata[i][3][0]*mappingdata[i][3][1]*mappingdata[i][3][2]
            print(read_row)
            read_col = mappingdata[i][3][3]*(mappingdata[i][3][0]-self.NN[i][3]+1)**2
            print(read_col)
            indata = self.NN[i][0]*self.NN[i][1]*self.NN[i][2]
            read_data = mappingdata[i][0]
            tile_analysis = tile_latency_analysis(self.simconfig_path,read_row,read_col,indata,read_data,self.inprecision,self.pe_num[0]*self.pe_num[1])
            self.uniform_total_latency = tile_analysis.tile_latency + self.uniform_total_latency
            self.uniform_layer_latency.append(tile_analysis.tile_latency)
        return self.uniform_total_latency


# 生成noc和nop记录
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


def update_resource_allocation(original_data, new_mapping_schemes):
    """
    根据新的每层映射方案，更新原始资源分配字典
    
    参数：
        original_data: 原始资源分配字典（第一份数据）
        new_mapping_schemes: 新的每层映射方案列表（第二份数据），长度=层数，每个元素格式：
                            [data_transfer, compute_cycles, tile_count, mapping_params]
    
    返回：
        updated_data: 更新后的资源分配字典
    """
    # 深拷贝原始数据，避免修改原字典
    import copy
    updated_data = copy.deepcopy(original_data)
    
    # ========== 1. 提取基础配置和分段映射关系 ==========
    # 硬件配置：每个chiplet的tile数量
    tiles_per_chiplet = updated_data['hardware_config']['tiles_per_chiplet']
    # 分段-芯片-层 映射：{chiplet_id: {'layers': 层列表, 'next_tile_idx': 下一个可用tile编号}}
    chiplet_tile_alloc = {}
    for seg in updated_data['segment_statistics']:
        chiplet_id = seg['chiplet_id']
        layers = seg['layers']
        # 初始化每个chiplet的tile分配器（从0开始编号）
        if chiplet_id not in chiplet_tile_alloc:
            chiplet_tile_alloc[chiplet_id] = {
                'layers': layers,
                'next_tile_idx': 0  # 下一个可分配的tile编号
            }
    
    # ========== 2. 遍历每层，更新layer_mappings ==========
    total_tiles_used = 0  # 统计总tile使用量
    # 按层ID遍历新映射方案
    for layer_id, new_scheme in enumerate(new_mapping_schemes):
        # 解析新方案：数据传输量、计算周期、tile数量、映射参数
        data_transfer, compute_cycles, tile_count, _ = new_scheme
        
        # 找到该层所属的chiplet（从segment_statistics中匹配）
        chiplet_id = None
        for seg in updated_data['segment_statistics']:
            if layer_id in seg['layers']:
                chiplet_id = seg['chiplet_id']
                break
        if chiplet_id is None:
            raise ValueError(f"层{layer_id}未找到对应的chiplet")
        
        # 检查chiplet的tile容量是否足够
        chiplet_alloc = chiplet_tile_alloc[chiplet_id]
        if chiplet_alloc['next_tile_idx'] + tile_count > tiles_per_chiplet:
            raise ValueError(
                f"chiplet {chiplet_id} 剩余tile不足！已用{chiplet_alloc['next_tile_idx']}, "
                f"需分配{tile_count}, 总容量{tiles_per_chiplet}"
            )
        
        # 分配tile_indices（顺序编号）
        start_idx = chiplet_alloc['next_tile_idx']
        tile_indices = list(range(start_idx, start_idx + tile_count))
        # 更新chiplet的下一个可用tile编号
        chiplet_alloc['next_tile_idx'] += tile_count
        
        # 更新layer_mappings中的字段
        for layer_map in updated_data['layer_mappings']:
            if layer_map['layer_id'] == layer_id:
                layer_map['tile_count'] = tile_count
                layer_map['tile_indices'] = tile_indices
                layer_map['data_transfer'] = data_transfer
                layer_map['compute_cycles'] = compute_cycles
                # arrays_needed/actual_arrays与tile_count一致（按原始逻辑）
                layer_map['arrays_needed'] = tile_count
                layer_map['actual_arrays'] = tile_count
                # 效率：arrays_needed/actual_arrays * 100（始终100%）
                layer_map['efficiency'] = 100.0 if tile_count > 0 else 0.0
                break
        
        # 累加总tile使用量
        total_tiles_used += tile_count
    
    # ========== 3. 更新segment_statistics ==========
    for seg in updated_data['segment_statistics']:
        seg_layers = seg['layers']
        # 计算该段的总tile数和总数据传输量
        seg_total_tiles = 0
        seg_total_data = 0.0
        for layer_id in seg_layers:
            # 找到该层的映射信息
            for layer_map in updated_data['layer_mappings']:
                if layer_map['layer_id'] == layer_id:
                    seg_total_tiles += layer_map['tile_count']
                    seg_total_data += layer_map['data_transfer']
                    break
        # 更新分段统计
        seg['total_tiles_needed'] = seg_total_tiles
        seg['total_data_transfer'] = seg_total_data
        # 更新分段tile使用率（tile数 / 该chiplet总tile数 * 100）
        seg['utilization_rate'] = (seg_total_tiles / tiles_per_chiplet) * 100
    
    # ========== 4. 更新整体utilization统计 ==========
    total_tiles_available = updated_data['hardware_config']['total_tiles']
    updated_data['utilization'] = {
        'tiles_used': total_tiles_used,
        'tiles_available': total_tiles_available - total_tiles_used,
        'utilization_rate': (total_tiles_used / total_tiles_available) * 100
    }
    
    # ========== 5. 更新全局统计（可选） ==========
    # 重新计算总数据传输量
    total_data_transfer = sum(
        layer_map['data_transfer'] for layer_map in updated_data['layer_mappings']
    )
    updated_data['segmentation_info']['total_data_transfer'] = total_data_transfer
    updated_data['segmentation_info']['avg_data_per_segment'] = (
        total_data_transfer / updated_data['segmentation_info']['total_segments']
    )
    
    return updated_data


# 使用示例
if __name__ == "__main__":
    # 映射列表数据（新格式）
    mapping_data = [
        [24300.0, 900.0, 1.0, 3.0, 3.0, 3.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 16.0],
        [129600.0, 900.0, 1.0, 3.0, 3.0, 16.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 32.0],
        [56448.0, 196.0, 1.0, 3.0, 3.0, 32.0, 64.0],
        [20736.0, 72.0, 2.0, 3.0, 3.0, 64.0, 64.0],
        [20736.0, 72.0, 2.0, 3.0, 3.0, 64.0, 64.0],
        [20736.0, 72.0, 2.0, 3.0, 3.0, 64.0, 64.0],
        [20736.0, 72.0, 2.0, 3.0, 3.0, 64.0, 64.0],
        [20736.0, 72.0, 2.0, 3.0, 3.0, 64.0, 64.0],
        [64.0, 1.0, 1.0, 1.0, 1.0, 64.0, 10.0]
    ]
    NN_example = [
        [32, 32, 3, 3, 3, 16, 0, 0],
        [32, 32, 16, 3, 3, 16, 0, 0.125],
        [32, 32, 16, 3, 3, 16, 0, 0.1875],
        [32, 32, 16, 3, 3, 16, 0, 0.28125],
        [32, 32, 16, 3, 3, 16, 0, 0.21875],
        [32, 32, 16, 3, 3, 16, 0, 0.21875],
        [32, 32, 16, 3, 3, 16, 0, 0.28125],
        [32, 32, 16, 3, 3, 32, 1, 0.125],
        [16, 16, 32, 3, 3, 32, 0, 0.132812],
        [16, 16, 32, 3, 3, 32, 0, 0.421875],
        [16, 16, 32, 3, 3, 32, 0, 0.148438],
        [16, 16, 32, 3, 3, 32, 0, 0.21875],
        [16, 16, 32, 3, 3, 32, 0, 0.210938],
        [16, 16, 32, 3, 3, 64, 1, 0.164062],
        [8, 8, 64, 3, 3, 64, 0, 0.191406],
        [8, 8, 64, 3, 3, 64, 0, 0.287109],
        [8, 8, 64, 3, 3, 64, 0, 0.376953],
        [8, 8, 64, 3, 3, 64, 0, 0.269531],
        [8, 8, 64, 3, 3, 64, 1, 0.515625],
        [1, 1, 64, 1, 1, 10, 0, 0]
    ]
    # 假设配置文件路径
    simconfig_path = "SimConfig.ini"
    
    # 创建映射器实例（假设精度为16位）
    mapper = Complete_mapping(simconfig_path, NN_example, 16)
    
    print("=" * 80)
    print("测试支持并行操作的顺序映射算法:")
    print("=" * 80)
    
    # 执行并行顺序映射
    parallel_result = mapper.sequential_parallel_mapping(mapping_data)
    print("并行顺序映射结果:")
    print(parallel_result)
    # 打印映射摘要
    # mapper.print_parallel_mapping_summary(parallel_result)
    
    # # 可视化映射结果
    # mapper.visualize_parallel_mapping(parallel_result)
    # 执行均匀分段映射
    uniform_result = mapper.uniform_segmentation_mapping(mapping_data)
    print("uniform_result:")
    print(uniform_result['segment_statistics'])
    print(len(uniform_result['layer_mappings']))
    segment_info = []
    for segment in uniform_result['segment_statistics']:
        segment_info.append(segment['layers'])
    print(segment_info)
    # 打印映射摘要
    # mapper.print_uniform_segmentation_summary(uniform_result)

    # 可视化映射结果
    # mapper.visualize_uniform_segmentation(uniform_result)

    # 生成noc
   
    print(uniform_result)
    
    
    sim1 = mapper.performance_sim_sequential(mapping_data,'sequential')
    print("monijiegou:")
    print(sim1)
    # print(mapper.performance_sim(mapping_data,'sequential'))
    # 打印映射摘要
    # mapper.print_uniform_segmentation_summary(uniform_result)
    
    # 可视化映射结果
    # mapper.visualize_uniform_segmentation(uniform_result)

    # 生成noc

    all_designs = []

    for i in uniform_result['segment_statistics']:
        for j in i['layers']:
            print(i['layers'])
            design = auto_mapping(NN_example[j][0], NN_example[j][1], NN_example[j][3], NN_example[j][2], NN_example[j][5], 512,512, 16-len(i['layers']))
            all_designs.append(design)

    final_result = mapper.select_best_mapping_scheme_v2(
        all_designs,
        segment_info,
        [16,16,16,16],
        [1,0]
    )
    print("final_result")
    print(final_result)
    sim2= mapper.performance_sim_uniform(final_result,'new')
    print("monijiegou111:")
    print(sim2)
    
    # print(mapper.performance_sim(final_result,'new'))
    

    update_mapping_data = update_resource_allocation(uniform_result, final_result)
    print("更新后的映射数据:")
    print(update_mapping_data)
    # 示例映射报告（使用您提供的格式）
    example_mapping_report = {
        'hardware_config': {'total_chiplets': 4, 'tiles_per_chiplet': 16, 'total_tiles': 64},
        'parallel_analysis': {'total_basic_tiles_needed': 25, 'total_allocated_tiles': 50},
        'layer_mappings': [
            {'layer_id': 0, 'chiplet_id': 0, 'tile_indices': [0, 1]},
            {'layer_id': 1, 'chiplet_id': 0, 'tile_indices': [2, 3]},
            {'layer_id': 2, 'chiplet_id': 0, 'tile_indices': [4, 5]},
            {'layer_id': 3, 'chiplet_id': 0, 'tile_indices': [6, 7]},
            {'layer_id': 4, 'chiplet_id': 0, 'tile_indices': [8, 9]},
            {'layer_id': 5, 'chiplet_id': 0, 'tile_indices': [10, 11]},
            {'layer_id': 6, 'chiplet_id': 0, 'tile_indices': [12, 13]},
            {'layer_id': 7, 'chiplet_id': 0, 'tile_indices': [14, 15]},
            {'layer_id': 8, 'chiplet_id': 1, 'tile_indices': [0, 1]},
            {'layer_id': 9, 'chiplet_id': 1, 'tile_indices': [2, 3]},
            {'layer_id': 10, 'chiplet_id': 1, 'tile_indices': [4, 5]},
            {'layer_id': 11, 'chiplet_id': 1, 'tile_indices': [6, 7]},
            {'layer_id': 12, 'chiplet_id': 1, 'tile_indices': [8, 9]},
            {'layer_id': 13, 'chiplet_id': 1, 'tile_indices': [10, 11]},
            {'layer_id': 14, 'chiplet_id': 1, 'tile_indices': [12, 13, 14, 15]},
            {'layer_id': 15, 'chiplet_id': 2, 'tile_indices': [0, 1, 2, 3]},
            {'layer_id': 16, 'chiplet_id': 2, 'tile_indices': [4, 5, 6, 7]},
            {'layer_id': 17, 'chiplet_id': 2, 'tile_indices': [8, 9, 10, 11]},
            {'layer_id': 18, 'chiplet_id': 2, 'tile_indices': [12, 13, 14, 15]},
            {'layer_id': 19, 'chiplet_id': 3, 'tile_indices': [0, 1]},
        ],
        'utilization': {'tiles_used': 50, 'tiles_available': 14}
    }
    
    # 生成NoC和NoP记录
    noc_records, nop_records = generate_noc_nop_records(
        example_mapping_report,#update_mapping_data,
        NN_example, 
        inprecision=16  # 假设16位精度
    )
    print(noc_records)
    print(nop_records)
    

    # 优化NoC布局
    original_layouts = convert_noc_to_layout(noc_records)
    print(original_layouts)
    print("开始优化...")
    
    # 运行优化（使用较小的参数以加快速度）
    optimized_noc, optimized_nop, optimized_chromosomes = optimize_noc_layout_complete(
        noc_records, 
        nop_records,
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



    # 生成noc和nop记录 基础映射情况

    generate_traces_noc(64,'Resnet201',noc_records,10)
    generate_traces_nop(64,'Resnet201',nop_records,10)
    run_booksim_noc("/home/zxf1/master_code/Interconnect/","Resnet201",4)

    run_booksim_nop("/home/zxf1/master_code/Interconnect/",'Resnet201',4)

    # 生成noc和nop记录 优化映射情况

    generate_traces_noc_GA(64,'Resnet201',optimized_noc,10)
    generate_traces_nop_GA(64,'Resnet201',optimized_nop,10)
    run_booksim_noc("/home/zxf1/master_code/Genetic_A/","Resnet201",4)

