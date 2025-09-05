import math

def im2col (image_col, image_row, filter_col, filter_row, in_channel, out_channel, array_row, array_col) :

    col_slide = image_col - filter_col + 1
    row_slide = image_row - filter_row + 1
    
    col_cycle = math.ceil(out_channel/array_col)
    row_cycle = math.ceil(filter_row*filter_col*in_channel/array_row)
    total_cycle = col_slide * row_slide * row_cycle * col_cycle
    total_datasize = col_slide*row_slide*filter_row*filter_col*in_channel
    
    return total_cycle, total_datasize


def SDK (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :
    
    row_vector = filter_row * filter_col * in_channel
    col_vector = out_channel
    
    used_row = math.ceil(row_vector/array_row)
    used_col = math.ceil(col_vector/array_col)
    
    new_array_row = array_row * used_row
    new_array_col = array_col * used_col

    # initialize
    cycle = []
    w = []
    w.append(filter_row*filter_col)
    cycle.append(used_row*used_col*(image_row-filter_row+1)*(image_col-filter_col+1))
    
    i=0
    while True :
        i += 1
        pw_row = filter_row + i - 1 
        pw_col = filter_col + i - 1
        pw = pw_row * pw_col
        if pw*in_channel <= new_array_row and i * i * out_channel <= new_array_col :
            parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            parallel_window_col = math.ceil((image_col - (filter_col + i) + 1)/i) + 1
            
            if parallel_window_row * parallel_window_row * used_row * used_col <= cycle[0] :
                del cycle[0]
                del w[0]
                cycle.append(parallel_window_row * parallel_window_col * used_row * used_col)
                w.append(pw)
                total_datasize = parallel_window_col*parallel_window_row*pw*in_channel
            
        else :
            break
        
    
    return cycle, w, total_datasize


def vw_sdk (image_col, image_row, filter_col, filter_row, in_channel, out_channel, \
                    array_row, array_col) :

    i = 0 # initialize # overlap col
    j = 1 # overlap row

    reg_total_cycle = [] # initialize
    reg_overlap_row = []
    reg_overlap_col = []
    reg_row_cycle = []
    reg_col_cycle = []
    reg_ICt = []
    reg_OCt = []
    
    while True :
        try :
            i += 1
            if (i + filter_col) > image_col : 
                i = 1
                j += 1
                if j + filter_row > image_row : 
                    break

            # for parallel_window computing
            reg_N_parallel_window_row = math.ceil((image_row - (filter_row + i) + 1)/i) + 1
            reg_N_parallel_window_col = math.ceil((image_col - (filter_col + j) + 1)/j) + 1
            
            # for cycle computing
            # Tiled IC
            if in_channel == 3 :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                if ICt > in_channel :
                    ICt = 3
                row_cycle = math.ceil(in_channel / ICt)
            else :
                ICt = math.floor(array_row /((filter_row + i - 1)*(filter_col + j - 1)))
                row_cycle = math.ceil(in_channel / ICt)
            
            # Tiled OC
            OCt =  math.floor(array_col / (i * j))
            col_cycle = math.ceil(out_channel / OCt)
    
            reg_N_of_computing_cycle = reg_N_parallel_window_row * reg_N_parallel_window_col \
                                    * row_cycle * col_cycle
            
            if i == 1 : # initialize
                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

            if reg_total_cycle[0] > reg_N_of_computing_cycle :
                del reg_total_cycle[0]
                del reg_overlap_row[0]
                del reg_overlap_col[0]
                del reg_row_cycle[0]
                del reg_col_cycle[0]
                del reg_ICt[0]
                del reg_OCt[0]

                reg_total_cycle.append(reg_N_of_computing_cycle)
                reg_overlap_row.append(i)
                reg_overlap_col.append(j)
                reg_row_cycle.append(row_cycle)
                reg_col_cycle.append(col_cycle)
                reg_ICt.append(ICt)
                reg_OCt.append(OCt)

    
        except ZeroDivisionError :
            continue

    return reg_total_cycle[0], reg_overlap_col[0], reg_overlap_row[0], reg_row_cycle[0], reg_col_cycle[0], reg_ICt[0], reg_OCt[0] 