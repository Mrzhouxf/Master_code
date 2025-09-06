import os
import csv
import math

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


# Start splitting noc and nop
# Separate noc data and nop data based on network topology and chip size
# all_transfer_path represent data transfer path
# topology is net shape
# k is number of chips in a single dimension
# n is dimensionality degree
def split_transfer_data(all_transfer_path,topology,k,n):
    total_num_arrays = all_transfer_path[0][1] + all_transfer_path[0][2]
    for i in range(1,len(all_transfer_path)):
        total_num_arrays = total_num_arrays + all_transfer_path[i][2]
    if topology == 'mesh':
        num_tile = k**n
        num_chiplet = math.ceil(total_num_arrays/num_tile)





