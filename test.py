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
    生成神经网络层的映射记录字典
    
    参数:
        nn_structure: 神经网络结构参数列表，每一行对应一层
        layer_mappings_data: 每一层的映射数据列表，格式[数据传输量, 计算周期, 使用资源数, 映射方案]
        tiles_per_chiplet: 每个chiplet包含的tile数量，默认16
    
    返回:
        dict: 包含layer_mappings和utilization的映射记录字典
    """
    # 初始化变量
    current_chiplet_id = 0  # 当前使用的chiplet编号
    current_tile_idx = 0    # 当前chiplet中可用的tile起始索引
    layer_mappings = []     # 存储每一层的映射信息
    total_tiles_used = 0    # 累计使用的tile数量
    
    # 遍历每一层
    for layer_id, (nn_layer, mapping_layer) in enumerate(zip(nn_structure, layer_mappings_data)):
        # 解析映射数据
        data_transfer_raw, compute_cycles_raw, resource_num, _ = mapping_layer
        normalize_base = mapping_layer[1]  # 归一化基数（900/196/72等）
        
        # 计算当前层需要的tile数量（使用资源数作为tile_count）
        tile_count = int(resource_num)
        
        # 计算当前层可分配的tile索引列表
        tile_indices = []
        remaining_tiles_needed = tile_count
        
        # 分配tile（跨chiplet处理）
        while remaining_tiles_needed > 0:
            # 当前chiplet剩余可用tile数
            tiles_remaining_in_chiplet = tiles_per_chiplet - current_tile_idx
            
            # 本次能分配的tile数
            tiles_to_assign = min(remaining_tiles_needed, tiles_remaining_in_chiplet)
            
            # 生成tile索引
            tile_indices.extend(
                [current_tile_idx + i for i in range(tiles_to_assign)]
            )
            
            # 更新剩余需要的tile数和当前tile索引
            remaining_tiles_needed -= tiles_to_assign
            current_tile_idx += tiles_to_assign
            
            # 如果当前chiplet的tile用完，切换到下一个chiplet
            if current_tile_idx >= tiles_per_chiplet:
                current_chiplet_id += 1
                current_tile_idx = 0
        
        # 计算归一化后的传输量和计算周期
        data_transfer = round(data_transfer_raw / normalize_base, 0)
        compute_cycles = round(compute_cycles_raw / normalize_base, 0)
        
        # 构建当前层的映射信息
        layer_info = {
            'layer_id': layer_id,
            'chiplet_id': current_chiplet_id if remaining_tiles_needed == 0 else current_chiplet_id - 1,
            'segment_id': current_chiplet_id if remaining_tiles_needed == 0 else current_chiplet_id - 1,
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
    
    # 计算资源利用率
    # 总可用tile数 = 已使用的chiplet数 × 每chiplet的tile数
    total_chiplets_used = current_chiplet_id + (1 if current_tile_idx > 0 else 0)
    total_tiles_available = total_chiplets_used * tiles_per_chiplet
    utilization_rate = (total_tiles_used / total_tiles_available) * 100.0 if total_tiles_available > 0 else 0.0
    
    utilization = {
        'tiles_used': total_tiles_used,
        'tiles_available': total_tiles_available - total_tiles_used,
        'utilization_rate': round(utilization_rate, 1)
    }
    
    # 构建最终结果字典
    result = {
        'layer_mappings': layer_mappings,
        'utilization': utilization
    }
    
    return result


# 测试数据
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

mapping_example = [
    [24300.0, 900.0, 1.0, [3.0, 3.0, 3.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 16.0]],
    [129600.0, 900.0, 1.0, [3.0, 3.0, 16.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 32.0]],
    [56448.0, 196.0, 1.0, [3.0, 3.0, 32.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [20736.0, 72.0, 2.0, [3.0, 3.0, 64.0, 64.0]],
    [64.0, 1.0, 1.0, [1.0, 1.0, 64.0, 10.0]]
]

# 调用函数生成映射记录
if __name__ == "__main__":
    result = generate_layer_mappings(NN_example, mapping_example, tiles_per_chiplet=16)
    
    # 打印结果（格式化输出）
    print(result)
    noc,nop = generate_noc_nop_records(result,NN_example,8)
    print(noc)
    print(nop)