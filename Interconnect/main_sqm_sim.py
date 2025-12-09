import os
import sys
import argparse
from typing import List, Dict, Tuple, Any
import glob
import re
import subprocess
from squeue_optimized import *
import time

bandwidth = 32
scale = 100
# nop_bandwidth = 10
nop_scale = 10




def run_full_simulation(
    network_name: str, 
    array_row: int, 
    array_col: int, 
    num_chiplet: int, 
    mesh_size: int, 
    pop_size: int, 
    generations: int, 
    mutation_rate: float
):
    """
    执行完整的通信模拟流程，包括映射、优化和仿真。
    """
    
    print("===================================================")
    print(f"--- start {network_name} simulation (Array: {array_row}x{array_col}, Chiplet: {num_chiplet}, Mesh: {mesh_size}x{mesh_size}) ---")
    print("===================================================")

    try:
        # 1. 读取网络结构并划分映射
        print("========= Read network structure  ===============")
        mapping, net, resource = read_net_mapping_strategy(network_name, array_row, array_col)
        num_chiplet = math.ceil(resource/mesh_size/mesh_size)
        # 2. 计算传输路径并分配芯片
        print("========= Calculate transmission path and chip allocation ===============")
        transfer_path = calculate_transferpath(mapping, net, 8) # 假设quantify为 8
        chip_map = allocate_chips_new(mapping, mesh_size * mesh_size, num_chiplet)
        split_data = split_transmissions(chip_map, transfer_path)
        
        # 3. 顺序映射与NoC/NoP计算 (如果需要，但 GA 流程可能不需要)
        chip_layer, noc, nop = Sequential_mapping(split_data)
        
        # 4. 块映射 (用于初始布局)
        print("========= Generate initial block basic mapping layout ===============")
        input_layout, all_chip = block_mapping(mapping, mesh_size)
        
        
        # 5. 生成跟踪文件
        print("========= Generate Booksim tracking file(NOC and NOP) ===============")
        # 假设这里是生成 NoC and NoP跟踪，用于 Booksim
        
        generate_traces_noc(bandwidth, network_name, noc, scale)
        generate_traces_nop(bandwidth, network_name, nop, nop_scale)
        # 6. 运行 Booksim 模拟并处理结果

        print("========= Run Booksim simulation and analyze the results ===============")
        process_network_traces(
            network_name=network_name,
            mapping_mode="sqm",
            measurement_method="NoC",
            mesh=mesh_size
        )
        process_network_traces(
            network_name=network_name,
            mapping_mode="sqm",
            measurement_method="NoP",
            mesh=mesh_size
        )

        print("\n--- The simulation process has been fully completed ---")
        
    except NameError as e:
        print(f"Fatal error: Unable to execute simulation. Missing key functions/modules: {e}")
        print("Please ensure that the GeneticOptimizer.py file contains all referenced functions.")
    except Exception as e:
        print(f"An error occurred during the simulation process: {e}")
        
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='运行芯片网络映射优化和通信模拟流程。',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 核心参数
    parser.add_argument('--network', type=str, default='Resnet20', 
                        help='网络名称 (例如: Resnet20, Resnet110)')
    parser.add_argument('--ar', type=int, default=512, 
                        help='PIM 阵列的行数 (array_row)')
    parser.add_argument('--ac', type=int, default=512, 
                        help='PIM 阵列的列数 (array_col)')
    
    # 芯片/NoC 参数
    parser.add_argument('--chiplet', type=int, default=3, 
                        help='芯片 (Chiplet) 的数量')
    parser.add_argument('--mesh', type=int, default=4, 
                        help='单个芯片上 Mesh NoC 的 k 值 (例如 4x4 mesh 的 k=4)')
    parser.add_argument('--bandwidth', type=int, default=100,
                        help='数据传输带宽 (例如 8)')
                        
    # 遗传算法参数
    parser.add_argument('--pop', type=int, default=50, 
                        help='遗传算法种群大小 (pop_size)')
    parser.add_argument('--gen', type=int, default=100, 
                        help='遗传算法迭代代数 (generations)')
    parser.add_argument('--mut', type=float, default=0.1, 
                        help='遗传算法变异率 (mutation_rate)')
    
    args = parser.parse_args()
    start_time = time.perf_counter()
    run_full_simulation(
        network_name=args.network,
        array_row=args.ar,
        array_col=args.ac,
        num_chiplet=args.chiplet,
        mesh_size=args.mesh,
        pop_size=args.pop,
        generations=args.gen,
        mutation_rate=args.mut
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{elapsed_time:.2f} 秒")  