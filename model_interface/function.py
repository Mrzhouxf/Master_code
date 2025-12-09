import csv
import os
from typing import List, Union
from model_interface.im2 import *
import itertools
import threading
import pandas as pd
import math
import shutil
import sys
from typing import List, Tuple, Any

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
    # print('inchannel:',inchannel)
    # print('pw_h:',PW_h)
    # print('pw_w:',PW_w)
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
    # print(os.getcwd())
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
                if PW_w*PW_h>array_row:
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
        total_pareto = remove_duplicates(total_pareto)
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
                if PW_w*PW_h>array_row:
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
            # pareto1 = remove_duplicates(pareto1)
            total_pareto = total_pareto + pareto1
        # total_pareto = remove_duplicates(total_pareto)
        # if len(total_pareto)==1:
        #     total_pareto = total_pareto
        # else:
        #     total_pareto = total_pareto*2
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
            # if count >= max_rows:
            #     break  # 达到最大记录数时停止

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

    parent_dir = os.path.dirname(parent_dir)
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







# 增加递归深度限制以应对深层网络（例如 20 层）
# 默认的 Python 递归深度通常是 1000，这里我们将其设置为更高
sys.setrecursionlimit(2000) 

def process_neural_network_design_optimized(
    design_schemes: List[List[Tuple[float, float, float, Any]]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_optimized.csv", 
    max_rows: int = 1000000 
) -> None:
    """
    使用基于剪枝的回溯搜索来高效筛选符合约束条件的神经网络设计方案组合。
    
    参数与原函数相同，但性能针对多层网络进行了优化。
    """
    
    # 步骤 1: 单层预筛选
    # 剔除单个资源消耗就超过总资源上限的方案，虽然原函数已做，但保持流程一致。
    valid_schemes_per_layer = []
    for layer in design_schemes:
        valid_layer_schemes = []
        for scheme in layer:
            # scheme[2] 是单个方案的资源消耗
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
        
        # 优化：如果某一层没有可用的方案，则整个网络无解
        if not valid_layer_schemes:
             print(f"警告：第 {len(valid_schemes_per_layer) + 1} 层没有符合 'resource_limit' 的方案。")
             print(f"结果已写入 {output_file} 文件中！共找到 0 条记录。")
             return
             
        valid_schemes_per_layer.append(valid_layer_schemes)

    total_layers = len(valid_schemes_per_layer)
    global_count = 0
    
    # 步骤 2: CSV 文件初始化
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头 (与原函数保持一致)
        header = ["Scheme ID"]
        for i in range(total_layers):
            header.extend([
                f"Layer{i+1}_data_transmission", 
                f"Layer{i+1}_compute_cycles", 
                f"Layer{i+1}_resource_limit", 
                f"Layer{i+1}_mapping_design"
            ])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 步骤 3: 剪枝回溯搜索主函数
        def search_scheme(
            layer_index: int, 
            current_combo: List[Tuple[float, float, float, Any]],
            current_transmission: float,
            current_cycles: float,
            current_resource: float
        ):
            nonlocal global_count
            
            # 剪枝操作 (Pruning):
            # 实时检查当前的累积值是否超限。
            if (current_resource > resource_limit or
                current_cycles > cycle_limit or
                current_transmission > transmission_limit):
                return # 发现超限，立即返回，剪掉整个分支

            # 递归终止条件: 已经为所有层选择了方案
            if layer_index == total_layers:
                
                # 最终检查 resource_min
                if current_resource >= resource_min:
                    # 达到 max_rows 限制
                    if global_count >= max_rows:
                        return

                    # 写入数据行
                    global_count += 1
                    row = [global_count]
                    for scheme in current_combo:
                        row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                    row.extend([current_transmission, current_cycles, current_resource])
                    writer.writerow(row)
                return

            # 递归步骤: 遍历当前层的每一个可行方案
            for scheme in valid_schemes_per_layer[layer_index]:
                
                # 递归调用自身，进入下一层
                search_scheme(
                    layer_index + 1,
                    current_combo + [scheme], # 添加当前选择的方案到组合中
                    current_transmission + scheme[0],
                    current_cycles + scheme[1],
                    current_resource + scheme[2]
                )
                
        # 启动搜索
        # 初始调用：从第 0 层开始，所有累积值都为 0
        search_scheme(0, [], 0.0, 0.0, 0.0)

    print(f"结果已写入 {output_file} 文件中！共找到 {global_count} 条记录（最多 {max_rows} 条）。")



def process_neural_network_design_real_time_monitoring(
    design_schemes: List[List[Tuple[float, float, float, Any]]], 
    resource_limit: float, 
    resource_min: float, 
    cycle_limit: float, 
    transmission_limit: float, 
    output_file: str = "result_optimized.csv", 
    max_rows: int = 1000000 
) -> None:
    """
    使用基于动态下限剪枝（Dynamic Lower Bound Pruning）的回溯搜索，
    高效筛选符合约束条件的神经网络设计方案组合。

    参数与原函数相同，但性能针对多层网络进行了优化。
    """
    
    # 步骤 1: 单层预筛选 & 预计算最小资源
    valid_schemes_per_layer = []
    min_resource_per_layer = [] # 存储每层方案中的最小资源消耗
    
    for i, layer in enumerate(design_schemes):
        valid_layer_schemes = []
        current_min_resource = float('inf')
        
        for scheme in layer:
            # scheme[2] 是单个方案的资源消耗
            if scheme[2] <= resource_limit:
                valid_layer_schemes.append(scheme)
                # 记录该层的最小资源
                if scheme[2] < current_min_resource:
                    current_min_resource = scheme[2]
        
        # 优化：如果某一层没有可用的方案，则整个网络无解
        if not valid_layer_schemes:
            print(f"警告：第 {i + 1} 层没有符合 'resource_limit' 的方案。")
            print(f"结果已写入 {output_file} 文件中！共找到 0 条记录。")
            return
            
        valid_schemes_per_layer.append(valid_layer_schemes)
        min_resource_per_layer.append(current_min_resource)

    total_layers = len(valid_schemes_per_layer)
    global_count = 0
    
    # 步骤 2: 计算最小资源后缀和 (R_min_k -> N)
    # min_resource_suffix_sum[i] 存储的是 i 层到最后一层 (total_layers - 1) 的最小资源消耗总和。
    min_resource_suffix_sum = [0.0] * (total_layers + 1)
    # 逆序计算后缀和
    for i in range(total_layers - 1, -1, -1):
        min_resource_suffix_sum[i] = min_resource_suffix_sum[i+1] + min_resource_per_layer[i]
        
    # 检查总资源下限：如果所有层的最小资源总和 R_min_0 -> N 已经超过 resource_limit，则无解。
    if min_resource_suffix_sum[0] > resource_limit:
        print(f"警告：所有层的最小资源需求总和 ({min_resource_suffix_sum[0]:.2f}) 已超过 'resource_limit' ({resource_limit:.2f})。")
        print(f"结果已写入 {output_file} 文件中！共找到 0 条记录。")
        return

    # 步骤 3: CSV 文件初始化
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入表头
        header = ["Scheme ID"]
        for i in range(total_layers):
            header.extend([
                f"Layer{i+1}_data_transmission", 
                f"Layer{i+1}_compute_cycles", 
                f"Layer{i+1}_resource_limit", 
                f"Layer{i+1}_mapping_design"
            ])
        header.extend(["Total data transmission", "Total compute cycles", "Total resource"])
        writer.writerow(header)

        # 步骤 4: 剪枝回溯搜索主函数 (动态下限优化)
        def search_scheme(
            layer_index: int, 
            current_combo: List[Tuple[float, float, float, Any]],
            current_transmission: float,
            current_cycles: float,
            current_resource: float
        ):
            nonlocal global_count
            
            # **上界剪枝 (Upper Bound Pruning):** # 实时检查当前的累积值是否超限。
            if (current_cycles > cycle_limit or
                current_transmission > transmission_limit):
                return # 发现超限，立即返回，剪掉整个分支

            # 递归终止条件: 已经为所有层选择了方案
            if layer_index == total_layers:
                
                # 最终检查 resource_min
                if current_resource >= resource_min:
                    # 达到 max_rows 限制
                    if global_count >= max_rows:
                        return

                    # 写入数据行
                    global_count += 1
                    row = [global_count]
                    for scheme in current_combo:
                        row.extend([scheme[0], scheme[1], scheme[2], scheme[3]])
                    row.extend([current_transmission, current_cycles, current_resource])
                    writer.writerow(row)
                return

            # **动态下限剪枝 (Dynamic Lower Bound Pruning - 资源优化核心)**
            # 计算剩余所有层的最小资源总和
            remaining_min_resource = min_resource_suffix_sum[layer_index + 1]
            
            # 递归步骤: 遍历当前层的每一个可行方案
            for scheme in valid_schemes_per_layer[layer_index]:
                
                new_resource = current_resource + scheme[2]
                
                # **核心优化剪枝:** 检查当前资源 + 剩余层的最小资源是否会超限
                # 如果 new_resource + remaining_min_resource > resource_limit，
                # 那么无论剩余层怎么选，总资源都一定会超限，所以剪枝。
                if new_resource + remaining_min_resource > resource_limit:
                    # 由于 valid_schemes_per_layer 是无序的，这里无法直接 break，必须 continue。
                    # 如果能按资源升序排列，则可以优化为 break。
                    continue

                # 递归调用自身，进入下一层
                search_scheme(
                    layer_index + 1,
                    current_combo + [scheme], 
                    current_transmission + scheme[0],
                    current_cycles + scheme[1],
                    new_resource
                )
                
        # 启动搜索
        # 初始调用：从第 0 层开始，所有累积值都为 0
        search_scheme(0, [], 0.0, 0.0, 0.0)

    print(f"结果已写入 {output_file} 文件中！共找到 {global_count} 条记录（最多 {max_rows} 条）。")



## 逐层映射，最普通的映射（资源不足时，不能全局映射的时候）


def map_nn_layers_by_sequence(layers_data, resource_limit):
    """
    根据资源限制对神经网络层进行分批次映射，支持单个层资源超限时进行拆分。
    此版本确保：
    1. 第一层的数据始终从片外读取。
    2. 最后一层的数据始终写入片外。

    Args:
        layers_data (list): 包含每个层输入数据量和资源数的列表，
                            例如：[[input_data_size, resource_cost], ...]
        resource_limit (int): 可用的总资源数。

    Returns:
        list: 映射方案，每个元素代表一个批次，包含：
              - 'layers': 该批次映射的层索引列表（从0开始）。
                          如果一个层被拆分，则会多次出现该层索引。
              - 'off_chip_read_data': 该批次从片外读取的数据量（字节）。
              - 'off_chip_write_data': 该批次向片外写入的数据量（字节）。
              - 'is_split_part': 布尔值，表示是否是某个被拆分的层的子批次。
              - 'split_part_info': 如果是拆分批次，包含 (current_part, total_parts)
    """
    mapping_plan = []
    num_layers = len(layers_data)
    current_layer_index = 0
    is_first_batch_overall = True # 标记是否是整个网络的第一个批次
    total_data_transmission = 0

    while current_layer_index < num_layers:
        batch_layers = []
        current_batch_resource_cost = 0
        
        # --- 确定当前批次的片外读取数据量 ---
        off_chip_read_data = 0
        if is_first_batch_overall:
            if num_layers > 0:
                off_chip_read_data = layers_data[0][0] # 整个网络的第一层的输入数据
            is_first_batch_overall = False 
        else:
            # 对于非第一个批次，读取的数据是其当前层（即上一层）的输出
            off_chip_read_data = layers_data[current_layer_index][0]


        # --- 尝试将多个完整层放入当前批次 ---
        temp_current_layer_index = current_layer_index # 使用临时变量进行试探
        while temp_current_layer_index < num_layers:
            layer_resource_cost = layers_data[temp_current_layer_index][1]
            
            # 如果当前层自身的资源就超限，且批次是空的，则不能尝试将其与之前的层合并，需要单独处理
            if layer_resource_cost > resource_limit and not batch_layers:
                break 

            if current_batch_resource_cost + layer_resource_cost <= resource_limit:
                batch_layers.append(temp_current_layer_index)
                current_batch_resource_cost += layer_resource_cost
                temp_current_layer_index += 1
            else:
                break # 当前层无法放入当前批次 (可能需要拆分，或等待下一个批次)
        
        # --- 如果当前批次有完整层，则作为一个批次输出 ---
        if batch_layers:
            current_batch_write_data = 0
            # 确定该批次结束后需要写入片外的数据量
            # 如果后面还有层，则写入下一层的输入数据
            if temp_current_layer_index < num_layers:
                current_batch_write_data = layers_data[temp_current_layer_index][0]
            else:
                # 如果这是处理最后一层的批次 (或包含最后一层)
                # 且它不是一个拆分的层，那么其输出肯定要写入片外
                if batch_layers[-1] == num_layers - 1: # 批次包含了最后一层
                     # 写入数据量为最后一层的输入数据量，或者更精确的输出数据量（这里用输入代替）
                     current_batch_write_data = layers_data[num_layers - 1][0] 

            mapping_plan.append({
                'layers': batch_layers,
                'off_chip_read_data': off_chip_read_data,
                'off_chip_write_data': current_batch_write_data,
                'is_split_part': False,
                'split_part_info': None
            })
            current_layer_index = temp_current_layer_index # 更新到下一个待处理的层
            total_data_transmission = total_data_transmission + off_chip_read_data +current_batch_write_data
        
        # --- 如果批次为空（即没有完整层能放入），并且当前层自身资源超限，则进行拆分 ---
        elif current_layer_index < num_layers: # current_layer_index 未处理完
            layer_to_split_index = current_layer_index
            layer_resource_cost = layers_data[layer_to_split_index][1]
            layer_input_size = layers_data[layer_to_split_index][0]
            
            # 计算需要多少个子批次来处理这一层
            num_split_parts = math.ceil(layer_resource_cost / resource_limit)
            
            # 确定该层总的输出数据量（作为下一层的输入，如果存在）
            actual_total_output_for_split_layer = 0
            is_last_layer_in_network = (layer_to_split_index == num_layers - 1)

            if not is_last_layer_in_network:
                actual_total_output_for_split_layer = layers_data[layer_to_split_index+1][0]
            else:
                # 如果是被拆分的最后一层，其总输出量假定与输入量相同
                actual_total_output_for_split_layer = layer_input_size
            
            output_data_per_part = actual_total_output_for_split_layer / num_split_parts if num_split_parts > 0 else 0

            # 为这个被拆分的层生成子批次
            for i in range(num_split_parts):
                # 每个子批次都读取该层的完整输入数据（从片外）
                current_split_read_data = layer_input_size
                
                # 特殊处理第一个批次的第一个被拆分的层：它的 read_data 应该用 layers_data[0][0]
                if layer_to_split_index == 0 and i == 0:
                     current_split_read_data = layers_data[0][0]

                # 只有最后一个子批次才写入完整的输出，之前的子批次写入部分结果
                current_split_write_data = output_data_per_part
                
                # 如果是当前被拆分层的最后一个子批次
                if i == num_split_parts - 1:
                    current_split_write_data = output_data_per_part
                    
                    # 额外检查：如果是整个网络的最后一层，它的输出无论如何都要写入片外
                    # 这里的逻辑已经涵盖了，因为 actual_total_output_for_split_layer 会被赋值
                    # 只要不是0，就表示写入片外
                    if is_last_layer_in_network:
                        # 确保它不是0，除非实际输出就是0
                        if current_split_write_data == 0 and actual_total_output_for_split_layer > 0:
                             current_split_write_data = actual_total_output_for_split_layer
                
                # 再次确认：如果是整个网络的最后一层，且是其最后一个子批次，即使实际输出是0，也可能需要一个表示写入片外的动作
                # 这里假设0代表“无需写回”，如果必须写回一个空值，需要明确
                # 目前逻辑是，如果实际输出是0，就写0。如果实际输出不是0，就写实际输出。
                # 你的要求是“最后一层的数据肯定是要写入片外的”，那么如果其输出是0，也写0。

                mapping_plan.append({
                    'layers': [layer_to_split_index],
                    'off_chip_read_data': current_split_read_data,
                    'off_chip_write_data': current_split_write_data,
                    'is_split_part': True,
                    'split_part_info': (i + 1, num_split_parts)
                })
                total_data_transmission = total_data_transmission + current_split_read_data +current_split_write_data
            current_layer_index += 1 # 整个被拆分的层处理完毕，前进到下一层
    return mapping_plan,total_data_transmission


# 直接存储全局方案，再进行搜索

def generate_batch_options_from_each_layer(layers_data, resource_limit):
    """
    生成从每个未映射的层开始的所有可行批次映射方案。

    Args:
        layers_data (list): 包含每个层输入数据量和资源数的列表，
                            例如：[[input_data_size, resource_cost], ...]
        resource_limit (int): 可用的总资源数。

    Returns:
        dict: 键是层的索引（从0开始），值是一个列表，包含从该层开始的所有可行批次方案。
              每个方案是一个字典，结构与之前的批次方案类似，但会增加 'is_split_part' 标记。
              如果一个层自身超限，它的解决方案列表将只有一个拆分方案。
    """
    num_layers = len(layers_data)
    all_layer_options = {}

    # 辅助函数：计算拆分层的子批次信息
    def get_split_layer_parts(layer_index, initial_read_data):
        layer_resource_cost = layers_data[layer_index][1]
        layer_input_size = layers_data[layer_index][0]
        num_split_parts = math.ceil(layer_resource_cost / resource_limit)
        
        actual_total_output_for_split_layer = 0
        is_last_layer_in_network = (layer_index == num_layers - 1)

        if not is_last_layer_in_network:
            actual_total_output_for_split_layer = layers_data[layer_index+1][0]
        else:
            actual_total_output_for_split_layer = layer_input_size 
        
        output_data_per_part = actual_total_output_for_split_layer / num_split_parts if num_split_parts > 0 else 0

        split_parts_options = []
        for i in range(num_split_parts):
            current_split_read_data = layer_input_size
            if layer_index == 0 and i == 0:
                 current_split_read_data = layers_data[0][0]

            current_split_write_data = output_data_per_part
            if i == num_split_parts - 1:
                current_split_write_data = output_data_per_part
                if is_last_layer_in_network:
                    if current_split_write_data == 0 and actual_total_output_for_split_layer > 0:
                         current_split_write_data = actual_total_output_for_split_layer
            
            # 如果是最后一层的最后一个子批次，即使实际输出是0，也确保其写回数据量设置为其参考输出
            if is_last_layer_in_network and i == num_split_parts - 1 and current_split_write_data == 0:
                 current_split_write_data = output_data_per_part

            split_parts_options.append({
                'layers': [layer_index],
                'off_chip_read_data': initial_read_data if i == 0 else current_split_read_data, # 只有第一个子批次读入片外
                'off_chip_write_data': current_split_write_data,
                'is_split_part': True,
                'split_part_info': (i + 1, num_split_parts),
                'note': f"Split part {i+1} for Layer {layer_index}"
            })
        return split_parts_options

    for start_layer_idx in range(num_layers):
        current_layer_options = []
        
        # 确定当前批次的片外读取数据量（基于这个批次从哪个层开始）
        current_batch_read_data = layers_data[start_layer_idx][0]
        if start_layer_idx == 0: # 如果批次从第一层开始，那么读取第一层的输入
            current_batch_read_data = layers_data[0][0]

        # 检查当前层的资源消耗是否超出资源限制
        if layers_data[start_layer_idx][1] > resource_limit:
            # 如果当前层自身资源超限，唯一的方案就是拆分它
            split_options = get_split_layer_parts(start_layer_idx, current_batch_read_data)
            current_layer_options.extend(split_options)
            all_layer_options[start_layer_idx] = current_layer_options
            continue # 处理下一个起始层
        
        # 如果当前层可以单独放入批次，则尝试生成所有可能的组合
        current_batch_resource_cost = 0
        temp_batch_layers = [] # 存储当前正在构建的批次中的层
        
        for i in range(start_layer_idx, num_layers):
            layer_resource_cost = layers_data[i][1]

            # 如果当前层自身超限，不能将其作为常规组合的一部分，且之前的层已经打包，
            # 那么这个组合到此为止，下一轮循环会处理超限层
            if layer_resource_cost > resource_limit and len(temp_batch_layers) > 0:
                break
            # 如果当前层自身超限，且没有层打包，那么它会被上面 if condition 处理为拆分
            if layer_resource_cost > resource_limit:
                 # 这段代码不应该执行到这里，因为前面已经处理了单层超限的情况
                 # 为了鲁棒性，这里可以加上断言或者额外的错误处理
                 break

            if current_batch_resource_cost + layer_resource_cost <= resource_limit:
                temp_batch_layers.append(i)
                current_batch_resource_cost += layer_resource_cost

                # 构建一个方案：将 temp_batch_layers 作为一个批次
                batch_write_data = 0
                if i + 1 < num_layers:
                    batch_write_data = layers_data[i + 1][0]
                elif i == num_layers - 1: # 如果这是最后一层
                    batch_write_data = layers_data[num_layers - 1][0] # 确保最后一层输出
                
                # 创建一个副本，因为 temp_batch_layers 会继续添加
                option_layers = list(temp_batch_layers) 
                current_layer_options.append({
                    'layers': option_layers,
                    'off_chip_read_data': current_batch_read_data,
                    'off_chip_write_data': batch_write_data,
                    'is_split_part': False,
                    'split_part_info': None,
                    'note': f"Maps Layers {option_layers[0]} to {option_layers[-1]}"
                })
            else:
                break # 资源不足，无法再添加更多层
        
        all_layer_options[start_layer_idx] = current_layer_options

    return all_layer_options


# 使用动态规划来实现快速搜索得出可靠方案

def find_optimal_network_mapping(layers_data, resource_limit):
    """
    根据每个层的可行批次方案，使用动态规划寻找整个网络的最小传输延迟映射方案。

    Args:
        layers_data (list): 包含每个层输入数据量和资源数的列表。
        resource_limit (int): 可用的总资源数。

    Returns:
        tuple: (min_total_latency, optimal_mapping_plan)
               min_total_latency (float): 最小的总片外读写延迟。
               optimal_mapping_plan (list): 包含选定批次方案的列表。
    """
    num_layers = len(layers_data)
    
    # 获取从每个层开始的所有可行批次方案
    all_options = generate_batch_options_from_each_layer(layers_data, resource_limit)

    # 动态规划数组：dp[i] 表示从 Layer i 开始到网络结束的最小总延迟
    dp = [float('inf')] * (num_layers + 1)
    # best_path[i] 存储从 Layer i 开始时选择的最佳第一个批次方案
    # 如果是常规批次，存储该批次字典；如果是拆分层，存储包含所有子批次方案的列表
    best_path = [None] * (num_layers + 1)

    # 基础情况：网络结束后的延迟为0
    dp[num_layers] = 0

    # 从后往前计算 dp 数组
    for i in range(num_layers - 1, -1, -1):
        min_cost_for_i = float('inf')
        best_option_for_i = None

        # 遍历从 Layer i 开始的所有可行批次方案
        if i in all_options:
            for option in all_options[i]:
                # 方案可以是常规批次或拆分层的一部分
                if not option['is_split_part']:
                    # --- 处理常规批次方案 ---
                    current_batch_cost = option['off_chip_read_data'] + option['off_chip_write_data']
                    next_layer_idx = option['layers'][-1] + 1
                    
                    if next_layer_idx <= num_layers: # 确保索引有效
                        total_cost = current_batch_cost + dp[next_layer_idx]
                        
                        if total_cost < min_cost_for_i:
                            min_cost_for_i = total_cost
                            best_option_for_i = option
                else:
                    
                    # --- 处理拆分层方案 ---
                    # 只有当这是拆分方案的第一个子批次时，才进行聚合计算
                    # 避免对同一拆分层的每个子批次都重复计算总成本
                    if option['split_part_info'][0] == 1: # 只有拆分方案的第一个子批次需要计算整个拆分层的成本
                        total_split_batch_cost = 0
                        split_options_list = []
                        # 收集该层所有的拆分子批次方案并计算总成本
                        # 注意：这里假设 all_options[i] 中的拆分方案是连续排列的
                        for split_option in all_options[i]:
                            if split_option['is_split_part'] and split_option['layers'][0] == i:
                                total_split_batch_cost += (split_option['off_chip_read_data'] + split_option['off_chip_write_data'])
                                split_options_list.append(split_option)
                            else:
                                # 如果在拆分方案中遇到非拆分方案或不同层的拆分方案，则停止
                                # 这种情况不应该发生，因为 all_options[i] 应该是针对 Layer i 的
                                pass

                        next_layer_idx_after_split = i + 1
                        
                        if next_layer_idx_after_split <= num_layers: # 确保索引有效
                            total_cost_for_split_layer = total_split_batch_cost + dp[next_layer_idx_after_split]
                            
                            if total_cost_for_split_layer < min_cost_for_i:
                                min_cost_for_i = total_cost_for_split_layer
                                best_option_for_i = split_options_list # 存储整个拆分方案列表
                        # print(total_split_batch_cost)
        
        # 记录当前 i 的最小成本和最佳选择
        dp[i] = min_cost_for_i
        best_path[i] = best_option_for_i

    # 回溯构建最优映射方案
    optimal_mapping_plan = []
    current_layer_idx = 0
    while current_layer_idx < num_layers:
        chosen_option = best_path[current_layer_idx]
        if chosen_option is None:
            # 这种情况不应该发生，除非网络无法完全映射
            print(f"Error: No optimal path found from Layer {current_layer_idx}")
            break
        
        if isinstance(chosen_option, list): # 选择了拆分层的方案
            optimal_mapping_plan.extend(chosen_option)
            current_layer_idx += 1 # 整个拆分层处理完毕
        else: # 选择了常规批次方案
            optimal_mapping_plan.append(chosen_option)
            current_layer_idx = chosen_option['layers'][-1] + 1
            
    return dp[0], optimal_mapping_plan


# 暴力寻找，但是会中途剪枝,这个函数存在问题

def find_optimal_network_mapping_brute_force(layers_data, resource_limit):
    num_layers = len(layers_data)
    all_options = generate_batch_options_from_each_layer(layers_data, resource_limit)

    min_total_latency = float('inf')
    optimal_mapping_plan = []

    def explore_mappings(current_layer_idx, current_latency, current_plan):
        nonlocal min_total_latency, optimal_mapping_plan

        if current_layer_idx == num_layers:
            if current_latency < min_total_latency:
                min_total_latency = current_latency
                optimal_mapping_plan = list(current_plan)
            return

        if current_layer_idx > num_layers or current_latency >= min_total_latency:
            return

        if current_layer_idx in all_options and all_options[current_layer_idx]:
            # 检查当前层的选项是拆分层还是常规批次
            is_split_layer_options = all_options[current_layer_idx][0]['is_split_part']

            if is_split_layer_options:
                # 聚合这个拆分层的所有子批次成本
                total_split_batch_cost = 0
                split_options_list = []
                for split_option in all_options[current_layer_idx]:
                    total_split_batch_cost += (split_option['off_chip_read_data'] + split_option['off_chip_write_data'])
                    split_options_list.append(split_option)
                
                next_layer_idx_after_split = current_layer_idx + 1
                
                current_plan.extend(split_options_list)
                explore_mappings(next_layer_idx_after_split, current_latency + total_split_batch_cost, current_plan)
                
                for _ in split_options_list:
                    current_plan.pop()
            else:
                # 处理常规批次方案
                for option in all_options[current_layer_idx]:
                    # 再次检查，确保它是常规批次（虽然逻辑上如果is_split_layer_options为False，这里应该都是常规批次）
                    if not option['is_split_part']:
                        batch_cost = option['off_chip_read_data'] + option['off_chip_write_data']
                        next_layer_idx = option['layers'][-1] + 1
                        
                        current_plan.append(option)
                        explore_mappings(next_layer_idx, current_latency + batch_cost, current_plan)
                        current_plan.pop()
        # else: 如果 current_layer_idx 不在 all_options 中，说明该层无法映射，这条路径无效，剪枝（已由 current_layer_idx > num_layers 或 current_latency >= min_total_latency 隐式处理）

    explore_mappings(0, 0, [])

    return min_total_latency, optimal_mapping_plan





# 把csv文件转成二维矩阵

def read_csv_to_2d_array(filepath):
    """
    读取CSV文件并将其内容解析为二维数组（列表的列表）。

    Args:
        filepath (str): CSV文件的完整路径。

    Returns:
        list: 包含CSV文件内容的二维数组。
              如果文件不存在或读取失败，返回空列表。
    """
    data = []
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                # 默认情况下，csv.reader 将所有内容读取为字符串。
                # 如果你需要将数字转换为浮点数或整数，可以在这里进行转换。
                # 例如：row_as_numbers = [float(item) if item.replace('.', '', 1).isdigit() else item for item in row]
                # 这里我们保持原始的字符串形式，因为layers_data的第一个元素是浮点数，第二个是整数
                # 我们可以尝试转换一下，如果转换失败就保留字符串
                processed_row = []
                for item in row:
                    try:
                        # 尝试转换为浮点数
                        processed_row.append(float(item))
                    except ValueError:
                        # 如果不是浮点数，尝试转换为整数
                        try:
                            processed_row.append(int(item))
                        except ValueError:
                            # 如果都不是数字，则保留为字符串
                            processed_row.append(item)
                data.append(processed_row)
    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
    except Exception as e:
        print(f"读取CSV文件时发生错误: {e}")
    return data