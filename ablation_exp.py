from model_interface.function import * # 假设这里包含了所有的映射和搜索函数
import argparse
import time
import os
import sys
from model_interface.parallel_optimized_search import *

# --- 0. 网络限制配置 ---
# 定义 Resnet20, Resnet110, sqtf, vgg16 的周期、传输和资源限制。


# --- 1. 参数解析 ---
parser = argparse.ArgumentParser(description='Set the parameters for ablation study.')
parser.add_argument('--ar', default=512, type=int, help='N of rows of the PIM array')
parser.add_argument('--ac', default=512, type=int, help='N of columns of the PIM array')
parser.add_argument('--network', default='vgg16', type=str, help='Network name (e.g., vgg16)')
parser.add_argument('--resource', default=60, type=int, help='Resource limit (default: 60)')
parser.add_argument('--mode', default='C', type=str, help='Final selection mode: H/L/V/C/U (default: C)')
parser.add_argument('--ratio', default=[0.5, 0.5], nargs=2, metavar=('R0', 'R1'), type=float,
                    help='Two floats that sum to 1.0, only used when mode=U (default: 0.5 0.5)')

args = parser.parse_args()

# --- 2. 核心参数设置 ---
net = args.network
array_row = args.ar
array_col = args.ac
resource_limit = args.resource
mode = args.mode
ratio = args.ratio


if array_row == 128 and array_col == 128:
# cof(ResNet20):cycles:5085 transmission:644096 resource:69
# cof(vgg16):cycles:686846 transmission:56798280 resource:579
# cof(sqtf):cycles:63005 transmission:3428712 resource:108
    NETWORK_LIMITS = {
        'Resnet20': {'cycle': 5500, 'transmission': 650000, 'resource': 100},
        'Resnet110': {'cycle': 4800, 'transmission': 2000000, 'resource': 220},
        'sqtf': {'cycle': 64000, 'transmission': 3600000, 'resource': 120},
        'vgg13': {'cycle': 750000, 'transmission': 60000000, 'resource': 700},
        'vgg16': {'cycle': 750000, 'transmission': 60000000, 'resource': 1000},
    }
if array_row == 128 and array_col == 256:
# cof(Resnet20):cycles:3847 transmission:490972 resource:92
# cof(vgg16):cycles:392265 transmission:43224624 resource:306
# cof(sqtf):cycles:40518 transmission:2846352 resource:70
    NETWORK_LIMITS = {
        'Resnet20': {'cycle': 4000, 'transmission': 550000, 'resource': 110},
        'Resnet110': {'cycle': 4800, 'transmission': 2000000, 'resource': 220},
        'sqtf': {'cycle': 42000, 'transmission': 2900000, 'resource': 90},
        'vgg13': {'cycle': 420000, 'transmission': 45000000, 'resource': 360},
        'vgg16': {'cycle': 420000, 'transmission': 45000000, 'resource': 500},
    }
if array_row == 256 and array_col == 256:
# cof(Resnet20):cycles:1954 transmission:490972 resource:47
# cof(vgg16):cycles:203751 transmission:43224624 resource:154
# cof(sqtf):cycles:34516 transmission:2846352 resource:44
    NETWORK_LIMITS = {
        'Resnet20': {'cycle': 2000, 'transmission': 520000, 'resource': 60},
        'Resnet110': {'cycle': 4800, 'transmission': 2000000, 'resource': 220},
        'sqtf': {'cycle': 36000, 'transmission': 2900000, 'resource': 60},
        'vgg13': {'cycle': 220000, 'transmission': 45000000, 'resource': 200},
        'vgg16': {'cycle': 280000, 'transmission': 55000000, 'resource': 615},
    }


if array_row == 256 and array_col == 512:
# cof(Resnet20):cycles:1954 transmission:490972 resource:47
# cof(vgg16):cycles:203751 transmission:43224624 resource:154
# cof(sqtf):cycles:34516 transmission:2846352 resource:44
    NETWORK_LIMITS = {
        'Resnet20': {'cycle': 1000, 'transmission': 500000, 'resource': 50},
        'Resnet110': {'cycle': 2400, 'transmission': 2000000, 'resource': 220},
        'sqtf': {'cycle': 18000, 'transmission': 2900000, 'resource': 40},
        'vgg13': {'cycle': 150000, 'transmission': 40000000, 'resource': 130},
        'vgg16': {'cycle': 180000, 'transmission': 40000000, 'resource': 345},
    }

if array_row == 512 and array_col == 512:

    NETWORK_LIMITS = {
        'Resnet20': {'cycle': 850, 'transmission': 380000, 'resource': 35},
        'Resnet110': {'cycle': 4800, 'transmission': 2000000, 'resource': 220},
        'sqtf': {'cycle': 24800, 'transmission': 2350000, 'resource': 35},
        'vgg13': {'cycle': 90000, 'transmission': 40000000, 'resource': 60},
        'vgg16': {'cycle': 97000, 'transmission': 44000000, 'resource': 190},
    }
# 约束和限制
print('resource:', resource_limit)
current_limits = NETWORK_LIMITS[net]
CYCLE_LIMIT = current_limits['cycle']
TRANSMISSION_LIMIT = current_limits['transmission']
resource_limit = current_limits['resource']
MAX_ROWS = 1000000
TIMEOUT_SECONDS = 1800 
print('resource:', resource_limit)
print('cycle:', CYCLE_LIMIT)
print('transmission:', TRANSMISSION_LIMIT)
# 目录和文件设置
NET_NAME = 'NetWork_' + net + '.csv'
# 结果目录：所有文件都将存放在这里
LOG_DIR = net+'/'+net+'_'+str(array_row)+'_'+str(array_col) 
# LOG_FILE = os.path.join(LOG_DIR, net+'_'+str(resource_limit)+'_'+str(CYCLE_LIMIT)+'_'+str(TRANSMISSION_LIMIT)+'_'+'ablation_study_log.txt')

LOG_FILE = os.path.join(
    LOG_DIR,
    f"{net}_{resource_limit}_{CYCLE_LIMIT}_{TRANSMISSION_LIMIT}_{array_row}_{array_col}_ablation_study_log.txt"
)

# 确保结果目录存在
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- 3. 准备网络结构和最小资源 ---
try:
    net_structure, net_conv_minarray, net_fc_minarray = calculate_min_array(NET_NAME, array_row, array_col)
    RESOURCE_MIN = net_conv_minarray + net_fc_minarray
except Exception as e:
    print(f"Error calculating min array for {NET_NAME}: {e}")
    sys.exit(1)

if RESOURCE_MIN > resource_limit:
    print(f"Insufficient resources: Min required ({RESOURCE_MIN}) > Limit ({resource_limit}). Exiting.")
    sys.exit(0)

DYNAMIC_ARRAY = resource_limit - RESOURCE_MIN

net_name = 'NetWork_'+net+'.csv'
reproduce_exp(net_name,array_row,array_col)
move_csv_by_name(str(array_row)+"_"+str(array_col),net+"/"+net+"_"+str(array_row)+"_"+str(array_col))

# --- 4. 定义消融实验的模式和函数映射 ---

# 方案生成模式 (Scheme Generation Modes)
generation_modes = {
    'simple_no_prune': repeat_auto_mapping, # 不去重
    'repeat_pruned': auto_mapping        # 去重
}

# 搜索算法模式 (Search Algorithm Modes)
search_functions = {
    'Backtrack_RealTime': process_neural_network_design_real_time_monitoring,
    'Backtrack_Optimized': process_neural_network_design_optimized,
    'BruteForce_Basic': process_neural_network_design,
    'Multi-process-RealTime':process_parallel_dlbp,
    'Multi-process_Optimized':process_parallel_backtracking
    # 'Multi-process_Basic':process_parallel_brute_force
}


# --- 5. 核心实验函数 ---

def run_ablation_test(gen_mode_name, search_mode_name, gen_func, search_func):
    """运行一次完整的消融实验，并记录时间。"""
    
    start_time = time.perf_counter()
    
    # --- Step A: 方案生成 ---
    all_design = []
    for i in range(len(net_structure)):
        # net_structure[i][-1] 假设是该层的最小资源需求
        layer_min_resource = net_structure[i][-1]
        layer_max_resource = DYNAMIC_ARRAY + layer_min_resource 
        
        design = gen_func(
            net_structure[i][0], net_structure[i][1], net_structure[i][3],
            net_structure[i][2], net_structure[i][5], array_row, array_col, layer_max_resource
        )
        all_design.append(design)
    
    # --- Step B: 搜索与计时 ---
    # 保证文件名清晰区分
    output_filename = f"{gen_mode_name}_{search_mode_name}_results.csv"
    output_file = os.path.join(LOG_DIR, output_filename)
    
    print(f"\n--- Starting: {gen_mode_name} + {search_mode_name} ---")
    
    try:
        # 调用搜索函数
        search_func(
            all_design, resource_limit, RESOURCE_MIN, CYCLE_LIMIT, 
            TRANSMISSION_LIMIT, output_file, MAX_ROWS
        )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # --- Step C: 结果统计与记录 ---
        
        # 统计找到的方案数
        found_count = 0
        try:
            with open(output_file, 'r') as f:
                # 方案数 = 行数 - 1 (减去表头)
                found_count = sum(1 for line in f) - 1 
        except FileNotFoundError:
            pass # 找不到文件，方案数仍为 0
            
        
        log_message = (
            f"Mode: {gen_mode_name} + {search_mode_name}\n"
            f"  Time: {total_time:.2f} seconds\n"
            f"  Found Schemes: {found_count}\n"
            f"  Output File: {output_filename}\n"
        )
        
        # 写入日志文件
        with open(LOG_FILE, 'a') as f:
            f.write(log_message + "\n")
            
        print(f"Completed in {total_time:.2f}s. Found {found_count} schemes.")
        
    except Exception as e:
        error_message = f"Mode: {gen_mode_name} + {search_mode_name} FAILED. Error: {e}\n"
        print(error_message)
        with open(LOG_FILE, 'a') as f:
            f.write(error_message + "\n")


# --- 6. 执行消融实验循环 ---
print(f"--- Starting Ablation Study for {net} ---")
print(f"Results will be saved in directory: {LOG_DIR}")

# 写入初始日志头
with open(LOG_FILE, 'w') as f:
    f.write(f"Ablation Study Log - Network: {net}, R_limit: {resource_limit}\n")
    f.write(f"Ablation Study Log - Network: {net}, Cycle_limit: {CYCLE_LIMIT}\n")
    f.write(f"Ablation Study Log - Network: {net}, Tranmission_limit: {TRANSMISSION_LIMIT}\n")
    f.write(f"Ablation Study Log - Network: {net}, Arrayrow*Arraycol: {array_row,array_col}\n")
    f.write("="*50 + "\n")

# 遍历所有组合，运行 6 种实验
for gen_name, gen_func in generation_modes.items():
    for search_name, search_func in search_functions.items():
        run_ablation_test(gen_name, search_name, gen_func, search_func)

print("\n--- Ablation Study Complete ---")
print(f"Detailed logs saved to {LOG_FILE}")