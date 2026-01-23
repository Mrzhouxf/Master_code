import os
import csv
import math
import numpy as np

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
        filename = f"NetWork_{network_name}_{param1}_{param2}_cof.csv"
        file_path = os.path.join(".", network_name, filename)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误:文件不存在 - {file_path}")
            return None
        with open('NetWork_'+network_name+'.csv', newline='', encoding='utf-8') as f:
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
                
                result.append(processed_parts)
        
        print(f"成功读取并处理文件:{file_path},共{len(result)}行")
        return result,net
    
    except Exception as e:
        print(f"读取文件时发生错误:{str(e)}")
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
        nop_record = [[src_chip], [dest_chip], [total_data_volume]]
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
    Interconnect_path = './Interconnect'                                    # 根目录
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
    create_folder('Genetic_A')
    Interconnect_path = './Genetic_A'                                    # 根目录
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


def generate_traces_noc_batch(bus_width, netname, noc_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    create_folder('Batch_map')
    Interconnect_path = './Batch_map'                                    # 根目录
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



