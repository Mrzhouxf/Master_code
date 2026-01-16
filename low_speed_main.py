from model_interface.function import *
import argparse
import time
from model_interface.parallel_optimized_search import *
from model_interface.Batch_mapping import *
start_time = time.perf_counter()
parser = argparse.ArgumentParser(description='Set the parameters to operate main.py')
parser.add_argument('--ar', default = 512, type = int, help = 'N of rows of the PIM array')
parser.add_argument('--ac', default = 512, type = int, help = 'N of columns of the PIM array')
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
parser.add_argument('--resource',default = 220, type = int, help='Resource limit (default: 60)')
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
cycle_limit = 4800      # Total compute cycles上限
transmission_limit = 2000000  # Total data transmission上限
output_file = "global_"+net+"_mapping_design.csv"
max_rows = 1000000    # 最大记录数
timeout_seconds = 600 #设置超时时间为 30 分钟（1800 秒）


net_name = 'NetWork_'+net+'.csv'
net_structure,net_conv_minarray,net_fc_minarray = calculate_min_array(net_name,array_row,array_col)
resource_min = net_conv_minarray + net_fc_minarray



if resource_min <= resource_limit:
    reproduce_exp(net_name,array_row,array_col)
    move_csv_by_name(str(array_row)+"_"+str(array_col),net+"_not_repeat")
    check_and_enter_directory(net+"_not_repeat")
    dynamic_array = resource_limit - resource_min
    all_design = []
    for i in range(len(net_structure)):
        layer_max_resource = dynamic_array + net_structure[i][-1]
        design = auto_mapping(net_structure[i][0],net_structure[i][1],net_structure[i][3],
                                net_structure[i][2],net_structure[i][5],array_row,array_col,layer_max_resource)
        all_design.append(design)
    # target函数可以改变搜索速度，一共提供了多种搜索方法
    thread = TaskThread(target=process_neural_network_design_real_time_monitoring, args=(all_design, resource_limit, resource_min, cycle_limit, transmission_limit, output_file, max_rows))
    thread.start()
    # thread.join(timeout=timeout_seconds)
    thread.join()
    # process_neural_network_design(all_design,resource_limit,resource_min,cycle_limit,transmission_limit,output_file,1000)

    # if thread.is_alive():
    #     print("Function execution timed out and stopped.")
    # else:
    #     print("Function execution completed.")
    result = select_design_scheme(output_file,mode,ratio,resource_limit)
    print(result)
    return_to_home_directory()

else:
    print("Insufficient resources, unable to fully map, please re-enter")
    print("Available resources:",resource_limit)
    print("Array size:",array_row,array_col)
    reproduce_exp(net_name,array_row,array_col)
    move_csv_by_name(str(array_row)+"_"+str(array_col),net+"_incomplete_map/"+net+"_"+str(array_row)+"_"+str(array_col))
    check_and_enter_directory(net+"_incomplete_map/"+net+"_"+str(array_row)+"_"+str(array_col))
    im2_layers_data = []
    sdk_layers_data = []
    vwsdk_layers_data = []
    im2=read_csv_to_2d_array('NetWork_'+net+'_'+str(array_row)+'_'+str(array_col)+'_im2.csv')
    sdk=read_csv_to_2d_array('NetWork_'+net+'_'+str(array_row)+'_'+str(array_col)+'_SDK.csv')
    vwsdk=read_csv_to_2d_array('NetWork_'+net+'_'+str(array_row)+'_'+str(array_col)+'_vwsdk.csv')
    max_layer_resource = 0
    for i in range(len(net_structure)):
        if net_structure[i][-1]>max_layer_resource:
            max_layer_resource = net_structure[i][-1]
        im2_layers_data.append([net_structure[i][0]*net_structure[i][1]*net_structure[i][2],im2[i][2]])
        sdk_layers_data.append([net_structure[i][0]*net_structure[i][1]*net_structure[i][2],sdk[i][2]])
        vwsdk_layers_data.append([net_structure[i][0]*net_structure[i][1]*net_structure[i][2],vwsdk[i][2]])
    
    # original plan 
    orignal_im2_plan,original_im2_data = map_nn_layers_by_sequence(im2_layers_data,resource_limit)
    optimal_im2_data,optimal_im2_plan = find_optimal_network_mapping(im2_layers_data,resource_limit)
    orignal_sdk_plan,original_sdk_data = map_nn_layers_by_sequence(sdk_layers_data,resource_limit)
    optimal_sdk_data,optimal_sdk_plan = find_optimal_network_mapping(sdk_layers_data,resource_limit)
    orignal_vwsdk_plan,original_vwsdk_data = map_nn_layers_by_sequence(vwsdk_layers_data,resource_limit)
    optimal_vwsdk_data,optimal_vwsdk_plan = find_optimal_network_mapping(vwsdk_layers_data,resource_limit)

    # print(optimal_vwsdk_plan)
    print("==========================Im2col data transmission================================")
    print("Original Im2Col Off-chip delay issue:",original_im2_data)
    print("Optimal Im2Col Off-chip delay issue:",optimal_im2_data)
    print("==========================SDK data transmission================================")
    print("Original SDK Off-chip delay issue:",original_sdk_data)
    print("Optimal SDK Off-chip delay issue:",optimal_sdk_data)
    print("==========================VWSDK data transmission================================")
    print("Original VWSDK Off-chip delay issue:",original_vwsdk_data)
    print("Optimal VWSDK Off-chip delay issue:",optimal_vwsdk_data)

    LOG_FILE = f"{net}_{resource_limit}_{array_row}_{array_col}_ablation_study_log.txt"
    with open(LOG_FILE, 'w') as f:
        f.write(f"Ablation Study Log - Network: {net}, R_limit: {resource_limit}\n")
        f.write(f"Ablation Study Log - Network: {net}, Arrayrow*Arraycol: {array_row,array_col}\n")
        f.write("="*50 + "\n")
        f.write(f"Original Im2Col Off-chip delay issue: {original_im2_data}\n")
        f.write(f"Optimal Im2Col Off-chip delay issue: {optimal_im2_data}\n")
        f.write("="*50 + "\n")
        f.write(f"Original SDK Off-chip delay issue: {original_sdk_data}\n")
        f.write(f"Optimal SDK Off-chip delay issue: {optimal_sdk_data}\n")
        f.write("="*50 + "\n")
        f.write(f"Original VWSDK Off-chip delay issue: {original_vwsdk_data}\n")
        f.write(f"Optimal VWSDK Off-chip delay issue: {optimal_vwsdk_data}\n")

    
    vwsdk_batch = Batch_mapping(optimal_vwsdk_plan,net_structure,vwsdk, resource_limit, array_row, array_col,8)
    print("VWSDK-batch:",vwsdk_batch.intra_layer_optimize())
    print("VWSDK-batch:",vwsdk_batch.off_chip_latency())

    sdk_batch = Batch_mapping(optimal_sdk_plan,net_structure,sdk, resource_limit, array_row, array_col,8)
    print("SDK-batch:",sdk_batch.intra_layer_optimize())
    print("SDK-batch:",sdk_batch.off_chip_latency())


    im2_batch = Batch_mapping(optimal_im2_plan,net_structure,im2, resource_limit, array_row, array_col,8)
    print("im2-batch:",im2_batch.intra_layer_optimize())
    print("im2-batch:",im2_batch.off_chip_latency())

    # print(optimal_vwsdk_plan)
    return_to_home_directory()
    # print(len(orignal_vwsdk_plan))
    
    

end_time = time.perf_counter()
print("Total execution time:",end_time-start_time)


