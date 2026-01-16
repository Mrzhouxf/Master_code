from model_interface.noc import *
import numpy as np
import os
import math


def generate_traces_nop(bus_width, netname, noc_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    Interconnect_path = '/home/zxf1/master_code/Interconnect'                                    # 根目录
    create_folder(netname + '_NoP_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
    file_path = Interconnect_path + '/' + netname + '_NoP_traces'           # trace 总目录

 
    # ---------------- 按 Chiplet 遍历 ----------------
    for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
        create_folder('Package_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
        chiplet_dir_name = file_path + '/Package_' + str(chip_id)           # 当前 Chiplet 目录

        # ---------------- 按层遍历 ----------------
        for i in range(len(noc_records[chip_id])):
            # 初始化：trace 第一行占位，后续会删除
            trace = np.array([[0, 0, 0]])
            timestamp = 1                                                     # 时间戳从 1 开始

            # 计算本层需要生成的 packet 数量
            num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
            num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

            # 提取源/目的 tile 区间
            src_tile_begin = noc_records[chip_id][i][0][0]
            src_tile_end   = noc_records[chip_id][i][0][-1]
            dest_tile_begin = noc_records[chip_id][i][1][0]
            dest_tile_end   = noc_records[chip_id][i][1][-1]

            # ---------------- 三重循环生成 trace ----------------
            for pack_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in range(dest_tile_begin, dest_tile_end + 1):
                    for src_tile_idx in range(src_tile_begin, src_tile_end + 1):
                        # 追加一行：源tile，目的tile，时间戳
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                    # 同目的不同源之间时间戳+1（除最后一个目的）
                    if dest_tile_idx != dest_tile_end:
                        timestamp += 1
                # 完成一个 packet 后时间戳再+1
                timestamp += 1

            # ---------------- 文件写出 ----------------
            filename = 'trace_file_layer_' + str(i) + '.txt'
            trace = np.delete(trace, 0, 0)                      # 删除初始占位行
            os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
            np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
            # 回到顶层，准备下一层
            os.chdir("..")
            os.chdir("..")
            os.chdir("..")



def generate_traces_nop_GA(bus_width, netname, noc_records, scale):
    """
    为片上网络(NoC)生成通信 trace 文件
    每个 Chiplet 每层生成一个 txt,记录 (src, dest, timestamp) 三列
    """
    # ---------------- 目录准备 ----------------
    Interconnect_path = '/home/zxf1/master_code/Genetic_A'                                    # 根目录
    create_folder(netname + '_NoP_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
    file_path = Interconnect_path + '/' + netname + '_NoP_traces'           # trace 总目录

 
    # ---------------- 按 Chiplet 遍历 ----------------
    for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
        create_folder('Package_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
        chiplet_dir_name = file_path + '/Package_' + str(chip_id)           # 当前 Chiplet 目录

        # ---------------- 按层遍历 ----------------
        for i in range(len(noc_records[chip_id])):
            # 初始化：trace 第一行占位，后续会删除
            trace = np.array([[0, 0, 0]])
            timestamp = 1                                                     # 时间戳从 1 开始

            # 计算本层需要生成的 packet 数量
            num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
            num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

            # 提取源/目的 tile 区间
            src_tile_begin = noc_records[chip_id][i][0][0]
            src_tile_end   = noc_records[chip_id][i][0][-1]
            dest_tile_begin = noc_records[chip_id][i][1][0]
            dest_tile_end   = noc_records[chip_id][i][1][-1]

            # ---------------- 三重循环生成 trace ----------------
            for pack_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in range(dest_tile_begin, dest_tile_end + 1):
                    for src_tile_idx in range(src_tile_begin, src_tile_end + 1):
                        # 追加一行：源tile，目的tile，时间戳
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                    # 同目的不同源之间时间戳+1（除最后一个目的）
                    if dest_tile_idx != dest_tile_end:
                        timestamp += 1
                # 完成一个 packet 后时间戳再+1
                timestamp += 1

            # ---------------- 文件写出 ----------------
            filename = 'trace_file_layer_' + str(i) + '.txt'
            trace = np.delete(trace, 0, 0)                      # 删除初始占位行
            os.chdir(chiplet_dir_name)                          # 进入本 Chiplet 目录
            np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
            # 回到顶层，准备下一层
            os.chdir("..")
            os.chdir("..")
            os.chdir("..")