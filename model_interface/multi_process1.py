import argparse
import time
import os # 恢复 os 模块导入
import sys # 恢复 sys 模块导入
# 确保在这里导入您所有的自定义模块
from function import *
from nop import *
from noc import *
from parallel_optimized_search import *
# from process import *
from multi_mode1 import *

# 必须使用 if __name__ == '__main__': 来保护所有可执行代码，
# 特别是那些启动多进程（如 TaskThread, Manager）的代码，以避免在 Windows 上出现 Runtime Error
if __name__ == '__main__':
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser(description='Set the parameters to operate main.py')
    parser.add_argument('--ar', default=512, type=int, help='N of rows of the PIM array')
    parser.add_argument('--ac', default=512, type=int, help='N of columns of the PIM array')
    parser.add_argument(
        '--network',
        default='Resnet110',
        type=str,
        help='Dataset = ['
             'NetWork_Resnet20.csv, '
             'NetWork_Resnet110.csv, '
             'NetWork_sqtf.csv, '
             'NetWork_vgg16.csv]'
    )
    parser.add_argument('--resource', default=220, type=int, help='Resource limit (default: 60)')
    parser.add_argument('--mode', default='C', type=str, help='Mode: H/L/V/C/U (default: C)')
    parser.add_argument('--ratio', default=[0.5, 0.5], nargs=2,
                        metavar=('R0', 'R1'), type=float,
                        help='Two floats that sum to 1.0, only used when mode=U (default: 0.5 0.5)')

    args = parser.parse_args()

    net = args.network
    array_row = args.ar
    array_col = args.ac
    resource_limit = args.resource
    mode = args.mode
    ratio = args.ratio
    cycle_limit = 4800            # Total compute cycles上限
    transmission_limit = 2020000   # Total data transmission上限
    output_file = "global_" + net + "_mapping_design.csv"
    max_rows = 1000000              # 最大记录数
    timeout_seconds = 1000000           # 设置超时时间为 10 分钟

    # --- 路径修复核心：动态构建绝对路径 ---
    # 获取脚本所在的绝对目录，避免依赖子进程的CWD
    try:
        script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    except:
        # 适用于特殊环境，如交互式解释器
        script_dir = os.getcwd()

    net_file_name = 'NetWork_' + net + '.csv'
    # 使用 os.path.join 确保路径分隔符在 Windows 和其他系统上都正确
    net_name = os.path.join(script_dir, net_file_name)
    # ------------------------------------

    print(os.getcwd())
    # net_name 现在是绝对路径，子进程可以安全加载
    net_structure, net_conv_minarray, net_fc_minarray = calculate_min_array(net_name, array_row, array_col)
    resource_min = net_conv_minarray + net_fc_minarray
    print("resource_min:",resource_min)
    reproduce_exp(net_name, array_row, array_col)
    # move_csv_by_name(str(array_row) + "_" + str(array_col), net + "_not_repeat")
    # check_and_enter_directory(net + "_not_repeat")
    move_csv_by_name(str(array_row) + "_" + str(array_col), net)
    check_and_enter_directory(net)
    print(os.getcwd())

    if resource_min <= resource_limit:
        dynamic_array = resource_limit - resource_min
        all_design = []
        for i in range(len(net_structure)):
            layer_max_resource = dynamic_array + net_structure[i][-1]
            design = auto_mapping(net_structure[i][0], net_structure[i][1], net_structure[i][3],
                                  net_structure[i][2], net_structure[i][5], array_row, array_col, layer_max_resource)
            all_design.append(design)

        # 由于多进程任务可能耗时很久，恢复 join with timeout
        thread = TaskThread(target=process_parallel_opt_H, args=(all_design, resource_limit, resource_min, cycle_limit, transmission_limit, output_file, max_rows))
        thread.start()
        thread.join()

        if thread.is_alive():
            print("Function execution timed out and stopped.")
        else:
            print("Function execution completed.")

        result = select_design_scheme(output_file, mode, ratio, resource_limit)
        print(result)
        return_to_home_directory()

    else:
        print("Insufficient resources, unable to fully map, please re-enter")

    end_time = time.perf_counter()
    print("Total execution time:", end_time - start_time)
