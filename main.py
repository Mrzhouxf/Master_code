from model_interface.function import *
import argparse
import time
from model_interface.parallel_optimized_search import *
from model_interface.Batch_mapping import *
from model_interface.Complete_mapping import *
import configparser as cp
simconfig = cp.ConfigParser()

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
# array_row = args.ar
# array_col = args.ac
# resource_limit = args.resource
mode = args.mode
ratio = args.ratio
cycle_limit = 4800      # Total compute cycles上限
transmission_limit = 2000000  # Total data transmission上限
output_file = "global_"+net+"_mapping_design.csv"
max_rows = 1000000    # 最大记录数
timeout_seconds = 600 #设置超时时间为 30 分钟（1800 秒）

simconfig.read('SimConfig.ini',encoding='UTF-8')
chiplet_num_str = simconfig.get('Package level','Chiplet_num')
chiplet_num = list(map(int,chiplet_num_str.split(',')))

tile_num_str = simconfig.get('Architecture level','Tile_num')
tile_num = list(map(int,tile_num_str.split(',')))

pe_num_str = simconfig.get('Tile level','PE_Num')
pe_num = list(map(int,pe_num_str.split(',')))

array_num_str = simconfig.get('Crossbar level','Xbar_Size')
array_num = list(map(int,array_num_str.split(',')))

group_num = int(simconfig.get('Process element level','Group_num'))

hetero = int(simconfig.get('Package level','heterogeneous'))

array_row = array_num[0]
array_col = array_num[1]
resource_limit = chiplet_num[0]*tile_num[0]*chiplet_num[1]*tile_num[1]*group_num*pe_num[0]*pe_num[1]


# print(hetero)


print("resource_limit:",resource_limit)
print("array_row:",array_row,"array_col:",array_col)
print("chiplet_num:",chiplet_num,"tile_num:",tile_num)

net_name = 'NetWork_'+net+'.csv'
net_structure,net_conv_minarray,net_fc_minarray = calculate_min_array(net_name,array_row,array_col)
resource_min = net_conv_minarray + net_fc_minarray
print("Minimum resources for complete mapping:",resource_min)
mapping_data = read_csv_to_2d_array(net+"/"+net+"_"+str(array_row)+"_"+str(array_col)+"/NetWork_"+net+"_"+str(array_row)+"_"+str(array_col)+"_im2.csv")
# print("mapping_data:",mapping_data)
# print(net_structure)

if resource_min <= resource_limit:
    # 原始搜索实验
    reproduce_exp(net_name,array_row,array_col)
    move_csv_by_name(str(array_row)+"_"+str(array_col),net+"/"+net+"_"+str(array_row)+"_"+str(array_col))
    
    # 2026.1.10
    list_trans_cycle = calculate_specified_columns_sum(net+"/"+net+"_"+str(array_row)+"_"+str(array_col)+"/NetWork_"+net+"_"+str(array_row)+"_"+str(array_col)+"_cof.csv",[1,2])
    cycle_limit = list_trans_cycle[1]*1.1
    transmission_limit = list_trans_cycle[0]*1.01
    # 2026.1.10
    check_and_enter_directory(net)
    
    dynamic_array = resource_limit - resource_min
    all_design = []
    for i in range(len(net_structure)):
        layer_max_resource = dynamic_array + net_structure[i][-1]
        design = auto_mapping(net_structure[i][0],net_structure[i][1],net_structure[i][3],
                                net_structure[i][2],net_structure[i][5],array_row,array_col,layer_max_resource)
        all_design.append(design)
    # target函数可以改变搜索速度，一共提供了多种搜索方法,加速实现直接替换调target函数就行
    thread = TaskThread(target=process_neural_network_design_real_time_monitoring, args=(all_design, resource_limit, resource_min, cycle_limit, transmission_limit, output_file, max_rows))
    thread.start()
   
    thread.join()
    # 加速实验
    result = select_design_scheme(output_file,mode,ratio,resource_limit)
    print(result)
    return_to_home_directory()
    # 原始搜索实验
    # global network sim
    simconfig_path = "SimConfig.ini"
    
    # 创建映射器实例（假设精度为16位）
    mapper = Complete_mapping(simconfig_path, net_structure, 16)
    # 执行并行顺序映射
    parallel_result = mapper.sequential_parallel_mapping(mapping_data)
    # 执行均匀分段映射
    uniform_result = mapper.uniform_segmentation_mapping(mapping_data)
    segment_info = []
    for segment in uniform_result['segment_statistics']:
        segment_info.append(segment['layers'])

    # 执行顺序映射的模拟
    sim1 = mapper.performance_sim_sequential(mapping_data,'sequential')
    print("monijiegou:")
    print(sim1)
    # 执行顺序映射的模拟

    all_designs = []

    for i in uniform_result['segment_statistics']:
        for j in i['layers']:
            # print(i['layers'])
            design = auto_mapping(net_structure[j][0], net_structure[j][1], net_structure[j][3], net_structure[j][2], net_structure[j][5], 512,512, 16-len(i['layers']))
            all_designs.append(design)
    final_result = mapper.select_best_mapping_scheme_v2(
        all_designs,
        segment_info,
        [16,16,16,16],
        [1,0]
    )
    # 执行均匀映射的模拟
    sim2= mapper.performance_sim_uniform(final_result,'new')
    print("monijiegou111:")
    print(sim2)
    #优化后的数据
    update_mapping_data = update_resource_allocation(uniform_result, final_result)
    print("更新后的映射数据:")
    print(update_mapping_data)
    # 优化后的映射数据

    # 生成NoC和NoP记录
    noc_records, nop_records = generate_noc_nop_records(
        parallel_result,#update_mapping_data,
        net_structure, 
        inprecision=16  # 假设16位精度
    )

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
    # global network sim

else:
    # print("Insufficient resources, unable to fully map, please re-enter")
    print("Available resources:",resource_limit)
    print("Array size:",array_row,array_col)
    reproduce_exp(net_name,array_row,array_col)
    move_csv_by_name(str(array_row)+"_"+str(array_col),net+"_incomplete_map/"+net+"_"+str(array_row)+"_"+str(array_col))
    check_and_enter_directory(net+"_incomplete_map/"+net+"_"+str(array_row)+"_"+str(array_col))
    im2_layers_data = []
    sdk_layers_data = []
    vwsdk_layers_data = []
    im2=read_csv_to_2d_array('NetWork_'+net+'_'+str(array_row)+'_'+str(array_col)+'_im2.csv')
    print("im2:",im2)
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

    print("optimal_vwsdk_plan:",optimal_vwsdk_plan)
    print("optimal_sdk_plan:",optimal_sdk_plan)
    print("optimal_im2_plan:",optimal_im2_plan)

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

    
    # vwsdk_batch = Batch_mapping(optimal_vwsdk_plan,net_structure,vwsdk, resource_limit, array_row, array_col,8)
    # print("VWSDK-batch:",vwsdk_batch.intra_layer_optimize())
    # print("VWSDK-batch:",vwsdk_batch.off_chip_latency())

    # sdk_batch = Batch_mapping(optimal_sdk_plan,net_structure,sdk, resource_limit, array_row, array_col,8)
    # print("SDK-batch:",sdk_batch.intra_layer_optimize())
    # print("SDK-batch:",sdk_batch.off_chip_latency())


    # im2_batch = Batch_mapping(optimal_im2_plan,net_structure,im2, resource_limit, array_row, array_col,8)
    # print("im2-batch:",im2_batch.intra_layer_optimize())
    # print("im2-batch:",im2_batch.off_chip_latency())

    # print(im2_batch.layer_sequence())
    print("os.pwd():",os.getcwd())
    os.chdir("..")
    os.chdir("..")
    batch_mapper = Batch_mapping(
        SimConfig_path='/home/zxf1/master_code/SimConfig.ini',
        NN=net_structure,
        mapping_data=im2,
        inprecision=8  # 输入精度8bit
    )
    
    # 执行映射并打印结果
    result = batch_mapper.run_batch_mapping()
    all_designs = []
    segment_bathch = []
    for i in result['batches']:
        segment_bathch.append(i['layers'])
        for j in i['layers']:
            print(i['layers'])
            design = auto_mapping(net_structure[j][0], net_structure[j][1], net_structure[j][3], net_structure[j][2], net_structure[j][5], 512,512, 16-len(i['layers']))
            all_designs.append(design)

    print(result)
    final_result = batch_mapper.select_best_mapping_scheme(all_designs,segment_bathch,[16,16],[1,0])
    print("final_result:",final_result)
    print("映射完成！")
    updated_dict = batch_mapper.update_mapping_dict(final_result)
    print("update_map:",updated_dict)
    # print(result['batches'][0]['layers'])
    aaa = batch_mapper.evaluate_perfermance(final_result)
    print("evaluate_perfermance:",aaa)

    new_dict = generate_layer_mappings(net_structure,final_result,16)
    print("new_dict:",new_dict)


    noc, nop = generate_noc_nop_records(new_dict,net_structure,8)
    print("noc:",noc)
    print("nop:",nop)

    generate_traces_noc(64,'Resnet201_batch',noc,10)
    run_booksim_noc("/home/zxf1/master_code/Interconnect/","Resnet201_batch",4)
    # print(optimal_vwsdk_plan)
    return_to_home_directory()
    # print(len(orignal_vwsdk_plan))
    
    

end_time = time.perf_counter()
print("Total execution time:",end_time-start_time)


