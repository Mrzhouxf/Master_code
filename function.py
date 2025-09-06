import csv
import os
from typing import List, Union
from im2 import *
import itertools
import threading
import pandas as pd
import math
import shutil


class TaskThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)

def combine_arrays_to_csv(
    array1: List[List[Union[float, int]]],
    array2: List[List[Union[float, int]]],
    filename: str = "combined_output.csv"
) -> None:
    
    # 验证输入数组长度
    if len(array1) != len(array2):
        raise ValueError("输入数组必须具有相同的行数")
    
    # 合并数据
    combined_data = []
    for row1, row2 in zip(array1, array2):
        combined_row = row1 + row2
        combined_data.append(combined_row)
    
    # 写入CSV文件
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(combined_data)
        
    print(f"文件已保存至 {filename}")
    print(f"生成数据:{len(combined_data)} 行,每行 {len(combined_data[0])} 列")



def reproduce_exp(NetWork, array_row, array_col):
    net, conv, fc = read_csv(NetWork)
    net_im2 = []
    net_SDK = []
    net_vwsdk = []
    net_im2pw = []
    net_SDKpw = []
    net_vwsdkpw = []
    net_fm = []
    net_fmpw = []

    for i in range(len(net)):
        if net[i][3]!=1:

            T_im2col, data_im2 = im2col(net[i][0], net[i][1], net[i][3], net[i][3], net[i][2], net[i][5],array_row, array_col)
            T_SDK, w, data_SDK = SDK(net[i][0], net[i][1], net[i][3], net[i][3], net[i][2], net[i][5],array_row, array_col)
            total_cycle, overlap_col, overlap_row, row_cycle, col_cycle, ICt, OCt = vw_sdk(net[i][0], net[i][1], net[i][3], net[i][3], net[i][2], net[i][5],array_row, array_col)
            
            vwc_array = row_cycle*col_cycle
            # print(vwc_array)
            pw_row = net[i][3]+overlap_row-1
            pw_col = net[i][3]+overlap_col-1

            im2_perf, im2_pw = calculate_performance(net[i][0],net[i][1],net[i][3],net[i][2],net[i][5],net[i][3],net[i][3],net[i][2],net[i][5],array_row,array_col)
            SDK_perf, SDK_pw = calculate_performance(net[i][0],net[i][1],net[i][3],net[i][2],net[i][5],math.sqrt(w[0]),math.sqrt(w[0]),net[i][2],net[i][5],array_row,array_col)
            vw_sdk_perf, vw_sdk_pw = calculate_performance(net[i][0],net[i][1],net[i][3],net[i][2],net[i][5],pw_row,pw_col,ICt,OCt,array_row,array_col)
            auto_mm = auto_mapping(net[i][0],net[i][1],net[i][3],net[i][2],net[i][5],array_row,array_col,vwc_array)
            # print(auto_mm)
            # print(vwc_array)
            auto_mm = filter_by_index_and_value(auto_mm,2,vwc_array)

            # print(auto_mm)
            net_im2.append(im2_perf)
            net_im2pw.append(im2_pw)
            net_SDK.append(SDK_perf)
            net_SDKpw.append(SDK_pw)
            net_vwsdk.append(vw_sdk_perf)
            net_vwsdkpw.append(vw_sdk_pw)
            net_fm.append([auto_mm[0][0],auto_mm[0][1],auto_mm[0][2]])
            net_fmpw.append(auto_mm[0][3])

        else:
            fc_perf, fc_pw = calculate_performance_fc(net[i][0],net[i][1],net[i][3],net[i][2],net[i][5],array_row,array_col)
            net_im2.append(fc_perf)
            net_im2pw.append(fc_pw)
            net_SDK.append(fc_perf)
            net_SDKpw.append(fc_pw)
            net_vwsdk.append(fc_perf)
            net_vwsdkpw.append(fc_pw)
            net_fm.append(fc_perf)
            net_fmpw.append(fc_pw)
    arr_name = str(array_row)+'_'+str(array_col)
    name = os.path.splitext(NetWork)[0]
    combine_arrays_to_csv(net_im2,net_im2pw,name+"_"+arr_name+"_im2.csv")
    combine_arrays_to_csv(net_SDK,net_SDKpw,name+"_"+arr_name+"_SDK.csv")
    combine_arrays_to_csv(net_vwsdk,net_vwsdkpw,name+"_"+arr_name+"_vwsdk.csv")
    combine_arrays_to_csv(net_fm,net_fmpw,name+"_"+arr_name+"_cof.csv")
    return net_im2,net_im2pw,net_SDK,net_SDKpw,net_vwsdk,net_vwsdkpw,net_fm,net_fmpw



# Complete the performance calculation of the fully connected layer
def calculate_performance_fc(image_row, image_col, kernel, inchannel, outchannel,array_row, array_col):

    # Calculate the transmission cycle and the amount of data transmitted
    row = math.ceil(inchannel/array_row)
    col = math.ceil(outchannel/array_col)
    total_tranferdata = image_row*image_col*inchannel
    total_computecycle = row*col*image_col*image_row
    total_array = row*col

    # Computational performance
    performance = []
    performance.append(total_tranferdata)
    performance.append(total_computecycle)
    performance.append(total_array)

    # Calculation window
    flexwindow = []
    flexwindow.append(kernel)
    flexwindow.append(kernel)
    flexwindow.append(inchannel)
    flexwindow.append(outchannel)
    
    return performance, flexwindow


# conv calculate conpute cycle and transmit data
def calculate_performance(image_row, image_col, kernel, inchannel, outchannel, PW_h, PW_w, ICt, OCt, array_row, array_col):
    # reg transmit data
    flexible_window = []
    performance = []
    i = PW_h-kernel+1
    j = PW_w-kernel+1
    row_outCount = math.ceil((image_row-PW_h)/(i))+1
    col_outCount = math.ceil((image_col-PW_w)/(j))+1
    total_transfer = row_outCount*col_outCount
    num_ic = math.ceil(inchannel/ICt)
    # num_oc = math.ceil(outchannel/OCt)
    totalTransferredData = total_transfer*PW_h*PW_w*min(ICt*num_ic,inchannel)
    #reg compute cycle
    # AR_cycle = math.ceil(inchannel*PW_h*PW_w/array_row)
    AR_cycle = math.ceil(inchannel/math.floor(array_row/(PW_h*PW_w)))
    AC_cycle = math.ceil(OCt*i*j/array_col)*math.ceil(outchannel/OCt)
    # AR_cycle = math.ceil(inchannel/ICt)
    # AC_cycle = math.ceil(outchannel/OCt)
    # if inchannel == 3 :
    #     ict = math.floor(array_row /((PW_w)*(PW_h)))
    #     if ict > inchannel :
    #         ict = 3
    #     AR_cycle = math.ceil(inchannel / ict)
    # else :
    #     ict = math.floor(array_row /((PW_h)*(PW_w)))
    #     AR_cycle = math.ceil(inchannel / ict)

    col_slide = image_col - kernel + 1
    row_slide = image_row - kernel + 1
    
    col_cycle = math.ceil(outchannel/array_col)
    row_cycle = math.ceil(kernel*kernel*inchannel/array_row)
    total_cycle = col_slide * row_slide * row_cycle * col_cycle

    # print("AR:",AR_cycle)
    # print("AC:",AC_cycle)
    # print("total_transfer:",total_transfer)


    total_compute_cycle = total_transfer*AR_cycle*AC_cycle
    if total_compute_cycle>total_cycle:
        total_compute_cycle = total_cycle
    #reg array num
    flexible_window.append(PW_h)
    flexible_window.append(PW_w)
    # flexible_window.append(min(ICt,math.ceil(inchannel/num_ic)))
    flexible_window.append(ICt)
    flexible_window.append(OCt)
    total_array = AC_cycle*AR_cycle

    performance.append(totalTransferredData)
    performance.append(total_compute_cycle)
    performance.append(total_array)
    return performance, flexible_window

#
def filter_by_index_and_value(data, index, value):
    """
    Filter subarrays in a two-dimensional array based on specified indexes and values
    : param data:  Two-dimensional array
    : param index:  Index to be filtered
    : param value:  Target value (can be a single value or a list)
    : return:  List of subarrays that meet the criteria
    """
    result = []
    min_diff = float('inf')
    closest_item = None
    for item in data:
        # 检查索引是否有效
        if index < 0 or index >= len(item):
            continue  # 跳过索引无效的子数组
        if item[index] == value:
            result.append(item)
        # 计算差值
        diff = abs(item[index] - value)
        if diff < min_diff:
            min_diff = diff
            closest_item = item
    if not result and closest_item is not None:
        result.append(closest_item)
    return result
def remove_duplicates(data):
    """
    Remove duplicate subarrays from a two-dimensional array
    : param data:  Two-dimensional array
    : return:  List without duplicate subarrays
    """
    unique_items = []
    seen_features = set()
    
    for item in data:
        # Extract the first three elements
        prefix = tuple(item[:3])
        
        # Extract the combination of the first two elements from the fourth list (sorted)
        fourth_list = item[3]
        if len(fourth_list) >= 2:
            sorted_pair = tuple(sorted(fourth_list[:2]))
        else:
            sorted_pair = tuple(fourth_list)
        
        # Extract the last two elements of the fourth list
        suffix = tuple(fourth_list[2:]) if len(fourth_list) >= 2 else tuple()
        
        # Combined characteristics
        feature = (prefix, sorted_pair, suffix)
        
        if feature not in seen_features:
            seen_features.add(feature)
            unique_items.append(item)
    
    return unique_items


def select_design_by_mode(design: dict, resource_limit: float, mode: str = "compute_first", alpha: float = 0.5):

    layers = sorted(design.keys(), key=lambda x: int(x.split('layer')[-1]))
    layer_choices = [design[layer] for layer in layers]

    best_selection = None
    best_metrics = (float('inf'), float('inf'), float('inf'))

    # 预处理计算所有可行路径的最大/最小值，用于归一化
    all_compute_vals, all_cycle_vals = [], []

    def gather_extremes():
        def dfs_ext(i, total_compute, total_cycle, total_resource):
            if total_resource > resource_limit:
                return
            if i == len(layers):
                all_compute_vals.append(total_compute)
                all_cycle_vals.append(total_cycle)
                return
            for opt in layer_choices[i]:
                dfs_ext(i + 1, total_compute + opt[0], total_cycle + opt[1], total_resource + opt[2])
        dfs_ext(0, 0.0, 0, 0.0)

    gather_extremes()

    compute_min, compute_max = min(all_compute_vals), max(all_compute_vals)
    cycle_min, cycle_max = min(all_cycle_vals), max(all_cycle_vals)

    def normalize(val, min_val, max_val):
        return 0.0 if max_val == min_val else (val - min_val) / (max_val - min_val)

    def compute_stability(cycles):
        return sum(abs(cycles[i] - cycles[i+1]) for i in range(len(cycles)-1))

    def dfs(layer_index, current_selection, total_resource, total_cycle, total_compute, cycle_list):
        nonlocal best_selection, best_metrics

        if total_resource > resource_limit:
            return
        if layer_index == len(layers):
            stability = compute_stability(cycle_list)

            # Normalize values for fair weighting
            norm_compute = normalize(total_compute, compute_min, compute_max)
            norm_cycle = normalize(total_cycle, cycle_min, cycle_max)

            if mode == "compute_first":
                metrics = (total_compute, stability, total_cycle)
            elif mode == "cycle_first":
                metrics = (total_cycle, stability, total_compute)
            elif mode == "balanced":
                weighted_score = alpha * norm_compute + (1 - alpha) * norm_cycle
                metrics = (weighted_score, stability, total_cycle)
            else:
                raise ValueError("Unsupported mode")

            if metrics < best_metrics:
                best_metrics = metrics
                best_selection = list(current_selection)
            return

        for option in layer_choices[layer_index]:
            compute = option[0]
            cycle = option[1]
            resource = option[2]

            current_selection.append(option)
            dfs(layer_index + 1,
                current_selection,
                total_resource + resource,
                total_cycle + cycle,
                total_compute + compute,
                cycle_list + [cycle])
            current_selection.pop()

    dfs(0, [], 0.0, 0, 0.0, [])

    if best_selection is None:
        return {}

    return {layer: best_selection[i] for i, layer in enumerate(layers)}


# Read neural network data
# Split convolutional layers and fully connected layers
def read_csv(file_path):
    conv_layers = []
    fc_layers = []
    netstructure = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            float_row = [float(value) for value in row]
            if float_row[3]==1:
                float_row.append(0)
                fc_layers.append(float_row)   
            else:
                float_row.append(1)
                conv_layers.append(float_row)
            netstructure.append(float_row)
    return netstructure, conv_layers,fc_layers

# Computational neural networks require arrays
def calculate_min_array(file_path, array_row, array_col):
    net, conv, fc = read_csv(file_path)
    conv_array = 0
    fc_array = 0
    for i in range(len(net)):
        row = net[i][2]*net[i][3]*net[i][4]
        col = net[i][5]
        col_num = math.ceil(row/array_row)
        row_num = math.ceil(col/array_col)
        net[i].append(col_num*row_num)
    for i in range(len(conv)):
        row = conv[i][2]*conv[i][3]*conv[i][4]
        col = conv[i][5]
        conv_row_num = math.ceil(row/array_row)
        conv_col_num = math.ceil(col/array_col)
        conv_array = conv_row_num*conv_col_num + conv_array
    for j in range(len(fc)):
        fc_row_num = math.ceil(fc[j][2]/array_row)
        fc_col_num = math.ceil(fc[j][5]/array_col)
        fc_array = fc_array + fc_col_num*fc_row_num
    return net, conv_array, fc_array



#Split the data in the list
def split_listdata(data_list, index):
    data1 = []
    data2 = []
    for item in data_list:
        data1.append(item[:index])
        data2.append(item[index:])
    return data1, data2

# pareto
def is_dominated(solution1, solution2):
        """
        判断solution1是否被solution2支配
        :param solution1: 第一个解
        :param solution2: 第二个解
        :return: 如果solution1被solution2支配,返回True,否则返回False
        """
        # 检查solution2是否在所有指标上都优于solution1
        # for i in range(len(solution1)):
        if (solution2[0] <= solution1[0] and
            solution2[1] <= solution1[1] and
            solution2[2] <= solution1[2]):
            # 至少有一个指标更优
            if (solution2[0] < solution1[0] or
                solution2[1] < solution1[1] or
                solution2[2] < solution1[2]):
                return True
        return False

def update_pareto_set(new_solution, pareto_set):
    """
    更新帕累托最优解集
    :param new_solution: 新计算出的解（字典形式，包含三个指标）
    :param pareto_set: 当前的帕累托最优解集（列表形式）
    :return: 更新后的帕累托最优解集
    """
    # 检查新解是否被现有解集中的解支配
    dominated_by_existing = False
    
    if len(pareto_set)==0:
        pareto_set.append(new_solution)
        return pareto_set
    else:
        # filtered_data = [item for item in pareto_set if item[2] == resource]
        for solution in pareto_set:
            # if solution[2] == resource:
                if is_dominated(new_solution, solution):
                    dominated_by_existing = True
                    break
            # else:
            #     continue
        
        if not dominated_by_existing:
            # 移除被新解支配的解
            new_pareto_set = []
            for solution in pareto_set:
                # if solution[2] == resource:
                    if not is_dominated(solution, new_solution):
                        new_pareto_set.append(solution)
                # else:
                #     continue
            new_pareto_set.append(new_solution)
            return new_pareto_set
        else:
            return pareto_set

def auto_mapping(image_row, image_col, kernel, inchannel, outchannel, array_row, array_col, array_limit):
    PW_h = kernel
    PW_w = kernel
    # reg pareto set
    s1 = []
    s2 = []
    pareto = []
    pareto_performance = []
    pareto_design = []
    flex_window = []
    total_pareto = []
    s1_fc = []
    s2_fc = []
    # fc layer
    if kernel == 1:
        s1_fc, flex_window = calculate_performance_fc(image_row, image_row, kernel, inchannel, outchannel, array_row, array_col)
        s1_fc.append(flex_window)

        return [s1_fc]
    # conv layer
    else:
        for i in range(int(image_row - kernel)):
            for j in range(int(image_col - kernel)):
                PW_h = kernel + i
                PW_w = kernel + j
                if PW_w*PW_h>array_col:
                    continue
                ICt = max(math.floor(array_row/PW_h/PW_w),1)
                OCt = max(math.floor(array_col/(i+1)/(j+1)),1)
                if OCt>=outchannel:
                    OCt = outchannel
                if ICt>=inchannel:
                    ICt = inchannel
                s1, flex_window = calculate_performance(image_row, image_row, kernel, inchannel, outchannel,PW_h, PW_w, ICt, OCt, array_row, array_col)
                # s1.append(total_transferdata)
                # s1.append(total_computecycle)
                # s1.append(total_array)
                s1.append(flex_window)
                s2 = s1[:]
                # pareto.append(s2)
                pareto = update_pareto_set(s2,pareto)
                
                # pareto_performance, pareto_design = split_listdata(pareto, 3)
                s1.clear()
                # print("The total amount of data transmitted :",total_transferdata)
                # print("Total calculation cycle :",total_computecycle)
                # print("The total array :",total_array)
        for j in range(1,array_limit+1):
            pareto1 = filter_by_index_and_value(pareto,2,j)
            pareto1 = remove_duplicates(pareto1)
            total_pareto = total_pareto + pareto1
        return total_pareto #pareto_performance,pareto_design


# by no using remove duplicates
def repeat_auto_mapping(image_row, image_col, kernel, inchannel, outchannel, array_row, array_col, array_limit):
    PW_h = kernel
    PW_w = kernel
    # reg pareto set
    s1 = []
    s2 = []
    pareto = []
    pareto_performance = []
    pareto_design = []
    flex_window = []
    total_pareto = []
    s1_fc = []
    s2_fc = []
    # fc layer
    if kernel == 1:
        s1_fc, flex_window = calculate_performance_fc(image_row, image_row, kernel, inchannel, outchannel, array_row, array_col)
        s1_fc.append(flex_window)

        return [s1_fc]
    # conv layer
    else:
        for i in range(int(image_row - kernel)):
            for j in range(int(image_col - kernel)):
                PW_h = kernel + i
                PW_w = kernel + j
                if PW_w*PW_h>array_col:
                    continue
                ICt = max(math.floor(array_row/PW_h/PW_w),1)
                OCt = max(math.floor(array_col/(i+1)/(j+1)),1)
                if OCt>=outchannel:
                    OCt = outchannel
                if ICt>=inchannel:
                    ICt = inchannel
                s1, flex_window = calculate_performance(image_row, image_row, kernel, inchannel, outchannel,PW_h, PW_w, ICt, OCt, array_row, array_col)
                # s1.append(total_transferdata)
                # s1.append(total_computecycle)
                # s1.append(total_array)
                s1.append(flex_window)
                s2 = s1[:]
                # pareto.append(s2)
                pareto = update_pareto_set(s2,pareto)
                
                # pareto_performance, pareto_design = split_listdata(pareto, 3)
                s1.clear()
                # print("The total amount of data transmitted :",total_transferdata)
                # print("Total calculation cycle :",total_computecycle)
                # print("The total array :",total_array)
        
                total_pareto = total_pareto + pareto
        return total_pareto #pareto_performance,pareto_design


def Automated_Exploration_Framework(Network, array_row, array_col, max_numarray, mode, alpha: float = 0.5):
    net,conv_array, fc_array = calculate_min_array(Network,array_row,array_col)
    minnum_array = conv_array + fc_array
    # If it is smaller than the minimum array, special mapping is required
    # If greater than or equal to the minimum array, automatic mapping search is required
    net_array = []

    # result = []
    variable_array = 0
    for j in range(len(net)):
        net_array.append(net[j][9])
        if net[j][9]!=1:
            variable_array = variable_array + 1
    net_design = {}
    if minnum_array <= max_numarray:
        remain_array = max_numarray - minnum_array
        for i in range(len(net)):
            if net[i][8] == 1:
                perf = auto_mapping(int(net[i][0]),int(net[i][1]),int(net[i][3]),\
                    net[i][2],net[i][5],array_row,array_col,net[i][9]+remain_array)
                net_design['conv_layer'+str(i)] = perf
            else:
                perf = auto_mapping(int(net[i][0]),int(net[i][1]),int(net[i][3]),\
                    net[i][2],net[i][5],array_row,array_col,net[i][9]+remain_array)
                net_design['fc_layer'+str(i)] = perf
            # print(perf)
        if mode == "compute_first":
            result = select_design_by_mode(net_design, max_numarray, mode)
        elif mode == "cycle_first":
            result = select_design_by_mode(net_design, max_numarray, mode)
        else:
            result = select_design_by_mode(net_design, max_numarray, mode, alpha)
    elif minnum_array-variable_array<=max_numarray<minnum_array:
        print("------------Special mapping-----------------")
    else:
        print("------------The number of storage arrays is too small, please reset--------")
        print("------------The minimum required storage array is:",minnum_array)

    return result

def remove_duplicate_schemes(design_schemes):
    """
    去除每一层设计方案中的重复方案。

    参数:
        design_schemes (list): 一个列表，其中每个元素代表一层的设计方案，每个方案是一个列表，包含传输数据量、计算周期、资源限制和其他参数。

    返回:
        list: 去除重复方案后的设计方案列表。
    """
    unique_schemes = []
    for layer in design_schemes:
        seen = set()
        unique_layer = []
        for scheme in layer:
            # 将整个方案转换为元组，以便可以存储在集合中
            scheme_tuple = (scheme[0], scheme[1], scheme[2], tuple(scheme[3]))
            if scheme_tuple not in seen:
                seen.add(scheme_tuple)
                unique_layer.append(scheme)
        unique_schemes.append(unique_layer)
    return unique_schemes


def process_neural_network_design(design_schemes, resource_limit, resource_min, cycle_limit, transmission_limit, output_file="result2.csv", max_rows=1000):
    """
    筛选符合资源限制条件的神经网络设计方案组合，计算Total data transmission和Total compute cycles，并将结果写入CSV文件。

    参数:
        design_schemes (list): 各层的设计方案列表，每个元素是一个层的多个设计方案。
        resource_limit (int): 总资源上限。
        resource_min (int): 总资源下限。
        cycle_limit (int): Total compute cycles上限。
        transmission_limit (int): Total data transmission上限。
        output_file (str): 输出的CSV文件名，默认为 "result.csv"。
        max_rows (int): 最大写入的记录数量，默认为 1000。

    返回:
        None
    """
    # 获取每一层的可行方案
    valid_schemes_per_layer = []
    for layer in design_schemes:
        valid_layer_schemes = []
        for scheme in layer:
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
        valid_schemes_per_layer.append(valid_layer_schemes)

    # 打开CSV文件以准备写入
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        header = ["Scheme ID"]
        for i in range(len(design_schemes)):
            header.extend([f"Layer{i+1}_data_transmission", f"Layer{i+1}_compute_cycles", f"Layer{i+1}_resource_limit", f"Layer{i+1}_mapping_design"])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 生成所有可能的方案组合并实时筛选和写入
        count = 0
        # 使用 itertools.product 生成组合
        for combo in itertools.product(*valid_schemes_per_layer):
            if count >= max_rows:
                break  # 达到最大记录数时停止

            total_resource = sum(scheme[2] for scheme in combo)
            total_transmission = sum(scheme[0] for scheme in combo)
            total_cycles = sum(scheme[1] for scheme in combo)

            # 检查是否满足所有限制条件
            if (total_resource <= resource_limit and
                total_resource >= resource_min and
                total_cycles <= cycle_limit and
                total_transmission <= transmission_limit):

                # 写入数据行
                row = [count + 1]
                for scheme in combo:
                    row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                row.extend([total_transmission, total_cycles, total_resource])
                writer.writerow(row)
                count += 1

    print(f"结果已写入 {output_file} 文件中！共找到 {count} 条记录（最多 {max_rows} 条）。")





def check_and_enter_directory(folder_name):
    """
    Check if a folder exists; if not, create it, then navigate into the directory
    :param folder_name: Name of the folder to check and enter
    """
    # Check if the folder exists
    if os.path.exists(folder_name):
        # Check if it's a directory (not a file)
        if os.path.isdir(folder_name):
            print(f"Folder '{folder_name}' exists, entering the directory")
            # Navigate into the directory
            os.chdir(folder_name)
        else:
            print(f"Error: '{folder_name}' is a file, not a folder")
    else:
        # Create the folder
        print(f"Folder '{folder_name}' does not exist, creating the folder")
        os.makedirs(folder_name)
        # Navigate into the created directory
        os.chdir(folder_name)
        print(f"Created and entered folder '{folder_name}'")



def select_design_scheme(file_name, mode, ratio, resource_limit):
    # 读取文件
    df = pd.read_csv(file_name)

    # 提取最后三列数据
    selected_df = df.iloc[:, -3:]
    selected_df.columns = ['Total data transmission', 'Total compute cycles', 'resource limit']

    # 筛选出资源使用小于等于资源限制的方案
    feasible_df = selected_df[selected_df['resource limit'] <= resource_limit]

    if mode == 'H':
        # H模式：选择当前资源使用中计算周期最小的（然后再同周期下选择数据传输最小的）
        min_cycle_df = feasible_df[feasible_df['Total compute cycles'] == feasible_df['Total compute cycles'].min()]
        result = min_cycle_df[min_cycle_df['Total data transmission'] == min_cycle_df['Total data transmission'].min()].iloc[0]
    elif mode == 'L':
        # L模式：选择当前资源使用中数据传输最小的（然后再同数据传数量中下选择周期最小的）
        min_transmission_df = feasible_df[feasible_df['Total data transmission'] == feasible_df['Total data transmission'].min()]
        result = min_transmission_df[min_transmission_df['Total compute cycles'] == min_transmission_df['Total compute cycles'].min()].iloc[0]
    elif mode == 'C':
        # C模式：选择周期最小的，并且资源使用最小，然后在这个基础上选择数据传输最小的
        min_cycle_resource_df = feasible_df[feasible_df['Total compute cycles'] == feasible_df['Total compute cycles'].min()]
        min_resource_df = min_cycle_resource_df[min_cycle_resource_df['resource limit'] == min_cycle_resource_df['resource limit'].min()]
        result = min_resource_df[min_resource_df['Total data transmission'] == min_resource_df['Total data transmission'].min()].iloc[0]
    elif mode == 'U':
        # U模式：对数据传输量和计算周期进行量化，赋予权重选取和最小的
        min_transmission = feasible_df['Total data transmission'].min()
        max_transmission = feasible_df['Total data transmission'].max()
        min_cycle = feasible_df['Total compute cycles'].min()
        max_cycle = feasible_df['Total compute cycles'].max()

        normalized_transmission = (feasible_df['Total data transmission'] - min_transmission) / (max_transmission - min_transmission)
        normalized_cycle = (feasible_df['Total compute cycles'] - min_cycle) / (max_cycle - min_cycle)

        score = ratio[0] * normalized_transmission + ratio[1] * normalized_cycle
        feasible_df['score'] = score

        result = feasible_df[feasible_df['score'] == feasible_df['score'].min()].iloc[0][:-1]
    else:
        result = None

    return result


def return_to_home_directory():
    current_dir = os.getcwd()
    # Get the parent directory of the current working directory
    parent_dir = os.path.dirname(current_dir)

    try:
        # Change to the parent directory
        os.chdir(parent_dir)
        print(f"Successfully changed to the parent directory: {parent_dir}")
    except FileNotFoundError:
        print("The parent directory does not exist. Maybe you are already at the root directory or the path is incorrect.")
    except NotADirectoryError:
        print("The specified path is not a directory.")




def move_csv_by_name(unwanted_str, target_folder):
    """
    移动当前目录下文件名包含未指定字符部分的CSV文件到指定文件夹
    
    参数:
        unwanted_str: 字符串，文件名中不应包含的字符部分
        target_folder: 字符串，目标文件夹名称
    """
    # 获取当前工作目录
    current_dir = os.getcwd()
    
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"已创建目标文件夹: {target_folder}")
    
    # 遍历当前目录下的所有文件
    for filename in os.listdir(current_dir):
        # 检查是否为CSV文件
        if filename.endswith('.csv'):
            # 检查文件名中是否包含未指定的字符部分
            if unwanted_str in filename:
                # 构建源文件和目标文件的完整路径
                source_path = os.path.join(current_dir, filename)
                target_path = os.path.join(current_dir, target_folder, filename)
                
                # 移动文件
                shutil.move(source_path, target_path)
                print(f"已移动文件: {filename} -> {target_folder}/")
    
    print("文件移动操作完成")