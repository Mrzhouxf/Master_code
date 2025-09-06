from function import *
import argparse

parser = argparse.ArgumentParser(description='Set the parameters to operate main.py')
parser.add_argument('--ar', default = 512, type = int, help = 'N of rows of the PIM array')
parser.add_argument('--ac', default = 512, type = int, help = 'N of columns of the PIM array')
parser.add_argument(
    '--network',
    default='vgg16',
    type=str,
    help='Dataset = ['
         'NetWork_Resnet20.csv, '
         'NetWork_Resnet110.csv, '
         'NetWork_sqtf.csv, '
         'NetWork_vgg16.csv]'
)
parser.add_argument('--resource',default = 60, type = int, help='Resource limit (default: 60)')
parser.add_argument('--mode',default = 'C', type = str, help='Mode: H/L/V/C/U (default: C)')
parser.add_argument('--ratio',default = [0.5, 0.5],nargs=2, 
                    metavar=('R0', 'R1'), type = float, 
                    help='Two floats that sum to 1.0, only used when mode=U (default: 0.5 0.5)')

args = parser.parse_args()

net = args.network
array_row = args.ar
array_col = args.ac
resource_limit = args.resource
mode = args.mode
ratio = args.ratio
cycle_limit =   90000      # Total compute cycles上限
transmission_limit = 45000000  # Total data transmission上限
output_file = "global_"+net+"_mapping_design.csv"
max_rows = 1000    # 最大记录数
timeout_seconds = 600 #设置超时时间为 30 分钟（1800 秒）


net_name = 'NetWork_'+net+'.csv'
net_structure,net_conv_minarray,net_fc_minarray = calculate_min_array(net_name,array_row,array_col)
resource_min = net_conv_minarray + net_fc_minarray
reproduce_exp(net_name,array_row,array_col)
move_csv_by_name(str(array_row)+"_"+str(array_col),net+"_not_repeat")
check_and_enter_directory(net+"_not_repeat")


if resource_min <= resource_limit:
    dynamic_array = resource_limit - resource_min
    all_design = []
    for i in range(len(net_structure)):
        layer_max_resource = dynamic_array + net_structure[i][-1]
        design = repeat_auto_mapping(net_structure[i][0],net_structure[i][1],net_structure[i][3],
                                net_structure[i][2],net_structure[i][5],array_row,array_col,layer_max_resource)
        all_design.append(design)
    
    thread = TaskThread(target=process_neural_network_design, args=(all_design, resource_limit, resource_min, cycle_limit, transmission_limit, output_file, max_rows))
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        print("Function execution timed out and stopped.")
    else:
        print("Function execution completed.")
    select_design_scheme(output_file,mode,ratio,resource_limit)
    return_to_home_directory()

else:
    print("Insufficient resources, unable to fully map, please re-enter")





