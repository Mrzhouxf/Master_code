import os
import sys
import argparse
from typing import List, Dict, Tuple, Any
import glob
import re
import subprocess
from GeneticOptimizer import *
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
    print(f"--- 启动 {network_name} 模拟 (阵列: {array_row}x{array_col}, 芯片: {num_chiplet}, Mesh: {mesh_size}x{mesh_size}) ---")
    print("===================================================")

    try:
        # 1. 读取网络结构并划分映射
        print("1. 读取网络结构...")
        mapping, net, resource = read_net_mapping_strategy(network_name, array_row, array_col)
        num_chiplet = math.ceil(resource/mesh_size/mesh_size)
        # 2. 计算传输路径并分配芯片
        print("2. 计算传输路径与芯片分配...")
        transfer_path = calculate_transferpath(mapping, net, 8) # 假设带宽为 8
        chip_map = allocate_chips_new(mapping, mesh_size * mesh_size, num_chiplet)
        split_data = split_transmissions(chip_map, transfer_path)
        
        # 3. 顺序映射与NoC/NoP计算 (如果需要，但 GA 流程可能不需要)
        # chip_layer, noc, nop = Sequential_mapping(split_data)
        
        # 4. 块映射 (用于 GA 初始布局)
        print("3. 生成初始块映射布局...")
        input_layout, all_chip = block_mapping(mapping, mesh_size)
        
        # 5. 遗传算法优化
        print(f"4. 启动遗传算法优化 (Pop: {pop_size}, Gen: {generations}, Mut: {mutation_rate})...")
        optimized_layouts, hops_summary = optimize_all_chips(
            input_layout=input_layout,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate
        )
        
        # 6. GA 映射并生成最终结果
        print("5. 整合 GA 优化结果...")
        full_result = GA_mapping(optimized_layouts, split_data)
        
        # 7. 生成跟踪文件
        print("6. 生成 Booksim 跟踪文件...")
        # 假设这里是生成 NoC 跟踪，用于 Booksim
        
        generate_traces_noc_GA(bandwidth, network_name, full_result['intra_records'], scale)
        generate_traces_nop_GA(bandwidth, network_name, full_result['inter_records'], nop_scale)
        # 8. 运行 Booksim 模拟并处理结果
        print("7. 运行 Booksim 模拟NOC并分析结果...")
        process_network_traces(
            network_name=network_name,
            mapping_mode="GA",
            measurement_method="NoC",
            mesh=mesh_size
        )

        print("8. 运行 Booksim 模拟NOP并分析结果...")

        process_network_traces(
            network_name=network_name,
            mapping_mode="GA",
            measurement_method="NoP",
            mesh=mesh_size
        )

        print("\n--- 模拟流程全部完成 ---")
        
    except NameError as e:
        print(f"致命错误: 无法执行模拟。缺失关键函数/模块: {e}")
        print("请确保 GeneticOptimizer.py 文件中包含所有被引用的函数。")
    except Exception as e:
        print(f"模拟过程中发生错误: {e}")
        
    


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
    parser.add_argument('--bandwidth', type=int, default=10,
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
