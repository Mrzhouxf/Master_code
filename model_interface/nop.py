from noc import *
import numpy as np
import os
import math

def generate_traces_nop(bus_width, netname, noc_records, scale, mode = 'sqm'):
    """
    为片上网络（NoP - Network on Package）生成跨芯粒（Chiplet）通信 trace
    所有层统一放在 ./Interconnect/<netname>_NoP_traces/Chiplet_NoP/ 目录下
    每层输出一个 txt，三列：src_tile_idx, dest_tile_idx, timestamp
    """
    if mode == 'sqm':
        # ---------------- 1. 目录准备 ----------------
        Interconnect_path = './Interconnect'                                    # 根目录
        create_folder(netname + '_NoP_traces', Interconnect_path)               # 创建 NoP 总目录
        file_path = Interconnect_path + '/' + netname + '_NoP_traces'           # 完整路径

        create_folder('Chiplet_NoP', file_path)                                 # 创建跨芯粒子目录
        chiplet_dir_name = file_path + '/Chiplet_NoP'                           # 保存路径

        # ---------------- 2. 按层遍历 ----------------
        for i in range(len(noc_records)):
            # 初始化：第一行占位，后续删除
            trace = np.array([[0, 0, 0]])
            timestamp = 1                                                       # 时间戳从 1 开始

            # 计算本层 packet 数量（先按 bus_width 切分，再按 scale 降采样）
            num_packets_this_layer = math.ceil(noc_records[i][2][0] / bus_width)
            num_packets_this_layer = math.ceil(num_packets_this_layer / scale)

            # 提取源/目的 tile 区间
            src_tile_begin = noc_records[i][0][0]
            src_tile_end   = noc_records[i][0][-1]
            dest_tile_begin = noc_records[i][1][0]
            dest_tile_end   = noc_records[i][1][-1]

            # ---------------- 3. 三重循环生成 trace ----------------
            for pack_idx in range(0, num_packets_this_layer):
                for dest_tile_idx in range(dest_tile_begin, dest_tile_end + 1):
                    for src_tile_idx in range(src_tile_begin, src_tile_end + 1):
                        # 追加一行：源 tile, 目的 tile, 时间戳
                        trace = np.append(trace, [[src_tile_idx, dest_tile_idx, timestamp]], axis=0)

                    # 同目的不同源之间时间戳 +1（跳过最后一个目的）
                    if dest_tile_idx != dest_tile_end:
                        timestamp += 1
                # 完成一个 packet 后时间戳再 +1
                timestamp += 1

            # ---------------- 4. 文件写出 ----------------
            filename = 'trace_file_layer_' + str(i) + '.txt'
            trace = np.delete(trace, 0, 0)                      # 删除初始占位行
            os.chdir(chiplet_dir_name)                          # 进入输出目录
            np.savetxt(filename, trace, fmt='%i')               # 保存为整数文本
            # 回到顶层，准备下一层
            os.chdir("..")
            os.chdir("..")
            os.chdir("..")
# mode is GA

    elif mode == 'GA':

        create_folder('Genetic_A')
        Interconnect_path = './Genetic_A'                                    # 根目录
        create_folder(netname + '_NoC_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
        file_path = Interconnect_path + '/' + netname + '_NoC_traces'           # trace 总目录

        # ---------------- 按 Chiplet 遍历 ----------------
        for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
            create_folder('Chiplet_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
            chiplet_dir_name = file_path + '/Chiplet_' + str(chip_id)           # 当前 Chiplet 目录

            # ---------------- 按层遍历 ----------------
            for i in range(len(noc_records[chip_id])):
                # 初始化：trace 第一行占位，后续会删除
                trace = np.array([[0, 0, 0]])
                timestamp = 1                                                     # 时间戳从 1 开始

                # 计算本层需要生成的 packet 数量
                num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
                num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

                # 提取源/目的 tile 区间
                src_tile = noc_records[chip_id][i][0]
                # src_tile_end   = noc_records[chip_id][i][0][-1]
                dest_tile = noc_records[chip_id][i][1]
                dest_tile_end   = noc_records[chip_id][i][1][-1]

                # ---------------- 三重循环生成 trace ----------------
                for pack_idx in range(0, num_packets_this_layer):
                    for dest_tile_idx in dest_tile:
                        for src_tile_idx in src_tile:
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

    #mode is ours

    else:
        # ---------------- 目录准备 ----------------
        create_folder('Ours')
        Interconnect_path = './Ours'                                    # 根目录
        create_folder(netname + '_NoC_traces', Interconnect_path)               # 创建 ./Interconnect/<netname>_NoC_traces/
        file_path = Interconnect_path + '/' + netname + '_NoC_traces'           # trace 总目录

        # ---------------- 按 Chiplet 遍历 ----------------
        for chip_id in sorted(noc_records.keys()):                              # 保证 Chiplet 顺序
            create_folder('Chiplet_' + str(chip_id), file_path)                 # 创建 ./Chiplet_<id>/
            chiplet_dir_name = file_path + '/Chiplet_' + str(chip_id)           # 当前 Chiplet 目录

            # ---------------- 按层遍历 ----------------
            for i in range(len(noc_records[chip_id])):
                # 初始化：trace 第一行占位，后续会删除
                trace = np.array([[0, 0, 0]])
                timestamp = 1                                                     # 时间戳从 1 开始

                # 计算本层需要生成的 packet 数量
                num_packets_this_layer = math.ceil(noc_records[chip_id][i][2] / bus_width)
                num_packets_this_layer = math.ceil(num_packets_this_layer / scale) # 再按 scale 降采样

                # 提取源/目的 tile 区间
            
                src_tile = noc_records[chip_id][i][0]
                # src_tile_end   = noc_records[chip_id][i][0][-1]
                dest_tile = noc_records[chip_id][i][1]
                dest_tile_end   = noc_records[chip_id][i][1][-1]

                # ---------------- 三重循环生成 trace ----------------
                for pack_idx in range(0, num_packets_this_layer):
                    for dest_tile_idx in dest_tile:
                        for src_tile_idx in src_tile:
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
