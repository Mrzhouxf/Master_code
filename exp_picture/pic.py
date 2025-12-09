import os
import csv
import pandas as pd
from model_interface.function import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False # 确保负号正常显示

def auto_mapping1(image_row, image_col, kernel, inchannel, outchannel, array_row, array_col, array_limit):
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
    remove_process = []
    
    # fc layer
    if kernel == 1:
        s1_fc, flex_window = calculate_performance_fc(image_row, image_row, kernel, inchannel, outchannel, array_row, array_col)
        s1_fc.append(flex_window)
        remove_process.append(len([[s1_fc]]))
        remove_process.append(len([[s1_fc]]))
        return [s1_fc],remove_process
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

                s1.append(flex_window)
                s2 = s1[:]
                # pareto.append(s2)
                pareto = update_pareto_set(s2,pareto)
                
                # pareto_performance, pareto_design = split_listdata(pareto, 3)
                s1.clear()

        for j in range(1,array_limit+1):
            pareto1 = filter_by_index_and_value(pareto,2,j)

            total_pareto = total_pareto + pareto1

        remove_process.append(len(total_pareto))
        total_pareto1 = remove_duplicates(total_pareto)
        remove_process.append(len(total_pareto1))

        return total_pareto,remove_process,total_pareto1

def run_experiment_for_network_and_array(net_name, array_row, array_col):
    """
    Run experiment for specific neural network and array configuration
    """
    print(f"Running experiment: {net_name}, Array: {array_row}x{array_col}")
    
    # Read network structure
    net, net_c, net_fc = calculate_min_array('NetWork_' + net_name + '.csv', array_row, array_col)
    
    fangan = []
    zonghe = []
    total_pareto_data = []  # Store the third set of data
    
    # Run mapping for each layer
    for layer_idx, layer in enumerate(net):
        print(f"  Processing layer {layer_idx + 1}/{len(net)}")
        result = auto_mapping1(layer[0], layer[1], layer[3], layer[2], layer[5], array_row, array_col, layer[-1] + 5)
        fangan.append(result[0])
        zonghe.append(result[1])
        if len(result) > 2:  # If there is third set of data
            total_pareto_data.append(result[2])
        else:
            total_pareto_data.append(None)
    
    return fangan, zonghe, total_pareto_data

def create_subplot(ax, zonghe_data, total_pareto_data, array_size, net_name):
    """
    Create a single subplot
    """
    # Extract valid data
    valid_layers = []
    before_pruning = []
    after_pruning = []
    pareto_count = []
    
    for layer_idx, (z_data, t_data) in enumerate(zip(zonghe_data, total_pareto_data)):
        if len(z_data) >= 2:  # Only process layers with before/after pruning data
            valid_layers.append(layer_idx)
            before_pruning.append(z_data[0])
            after_pruning.append(z_data[1])
            pareto_count.append(len(t_data) if t_data else 0)
    
    if not valid_layers:
        ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
        return
    
    x = np.arange(len(valid_layers))
    width = 0.25
    
    # Draw bar chart
    bars1 = ax.bar(x - width, before_pruning, width, label='Before Pruning', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, after_pruning, width, label='After Pruning', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, pareto_count, width, label='Pareto Solutions', color='#2ca02c', alpha=0.8)
    
    # Set chart properties
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Number of Solutions')
    ax.set_title(f'{net_name.upper()} - Array {array_size}', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show labels for values > 0
                ax.text(bar.get_x() + bar.get_width()/2., height + max(before_pruning + after_pruning + pareto_count)*0.01,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

def create_detailed_results_pdf(net_name, all_data, output_dir='duplicate'):
    """
    Create PDF with detailed subplots for a specific neural network
    """
    pdf_filename = f'{output_dir}/{net_name}_detailed_results.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # Determine subplot layout based on number of network layers
        sample_data = next(iter(all_data.values())) if all_data else None
        if sample_data:
            num_layers = len(sample_data[0])
            
            # Decide layout based on number of layers
            if num_layers <= 10:
                # Few layers: 2x2 layout
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.flatten()
            else:
                # Many layers: 4x1 layout
                fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        else:
            # Default 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        
        # Set main title
        fig.suptitle(f'{net_name.upper()} Neural Network Mapping Optimization Results', fontsize=16, fontweight='bold', y=0.98)
        
        # Create each subplot
        for idx, (array_size, (zonghe_data, total_pareto_data)) in enumerate(all_data.items()):
            if idx < len(axes):
                create_subplot(axes[idx], zonghe_data, total_pareto_data, array_size, net_name)
        
        # Hide extra subplots
        for idx in range(len(all_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for main title
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"Detailed PDF report generated: {pdf_filename}")
    return pdf_filename

def create_summary_pdf(net_name, all_data, output_dir='duplicate'):
    """
    Create PDF with summary statistics for a specific neural network
    """
    pdf_filename = f'{output_dir}/{net_name}_summary.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # Create summary statistics
        array_sizes = []
        avg_before = []
        avg_after = []
        reduction_ratios = []
        
        for array_size, (zonghe_data, _) in all_data.items():
            valid_layers = [z for z in zonghe_data if len(z) >= 2]
            if valid_layers:
                before = sum(z[0] for z in valid_layers) / len(valid_layers)
                after = sum(z[1] for z in valid_layers) / len(valid_layers)
                reduction = ((before - after) / before * 100) if before > 0 else 0
                
                array_sizes.append(array_size)
                avg_before.append(before)
                avg_after.append(after)
                reduction_ratios.append(reduction)
        
        # Create summary bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if array_sizes:
            x = np.arange(len(array_sizes))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, avg_before, width, label='Average Before Pruning', color='#1f77b4')
            bars2 = ax.bar(x + width/2, avg_after, width, label='Average After Pruning', color='#ff7f0e')
            
            # Display values above bars
            for i, (b, a, r) in enumerate(zip(avg_before, avg_after, reduction_ratios)):
                ax.text(i - width/2, b + max(avg_before)*0.01, f'{b:.1f}', ha='center', va='bottom')
                ax.text(i + width/2, a + max(avg_after)*0.01, f'{a:.1f}', ha='center', va='bottom')
                ax.text(i, -max(avg_before)*0.1, f'Reduction: {r:.1f}%', ha='center', va='top', fontweight='bold')
            
            ax.set_xlabel('Array Configuration')
            ax.set_ylabel('Average Number of Solutions')
            ax.set_title(f'{net_name.upper()} - Summary of Average Results Across Array Configurations', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(array_sizes)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"Summary PDF report generated: {pdf_filename}")
    return pdf_filename

def run_network_experiments(net_name, array_configs, output_dir='duplicate'):
    """
    Run experiments for a specific neural network and generate PDF reports
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Use only four array configurations (skip 128x128)
    selected_arrays = array_configs[1:]  # Skip the first one (128x128)
    
    # Collect all data
    all_data = {}
    for array_row, array_col in selected_arrays:
        print(f"Running {net_name} with {array_row}x{array_col}")
        try:
            fangan, zonghe, total_pareto = run_experiment_for_network_and_array(net_name, array_row, array_col)
            all_data[f"{array_row}x{array_col}"] = (zonghe, total_pareto)
            
            # Save raw data to CSV
            csv_filename = f'{output_dir}/zonghe_{net_name}_{array_row}x{array_col}.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Layer_Index', 'Before_Pruning', 'After_Pruning', 'Total_Pareto'])
                for layer_idx, (z_data, t_data) in enumerate(zip(zonghe, total_pareto)):
                    if len(z_data) >= 2:
                        writer.writerow([layer_idx, z_data[0], z_data[1], len(t_data) if t_data else 0])
                    else:
                        writer.writerow([layer_idx, z_data[0], '', len(t_data) if t_data else 0])
                        
        except Exception as e:
            print(f"Error: {net_name} with {array_row}x{array_col}: {str(e)}")
    
    # Create two separate PDF files
    detailed_pdf = create_detailed_results_pdf(net_name, all_data, output_dir)
    summary_pdf = create_summary_pdf(net_name, all_data, output_dir)
    
    return detailed_pdf, summary_pdf

def main():
    """
    Main function - Run all experiments and generate PDF reports
    """
    # Define array configurations (including 128x128, but will skip later)
    array_configs = [
        (128, 128),
        (128, 256), 
        (256, 256),
        (256, 512),
        (512, 512)
    ]
    
    # Define neural networks
    networks = ['vgg16', 'Resnet20']
    
    print("Starting neural network mapping optimization experiments...")
    print("=" * 50)
    
    # Generate PDF reports for each neural network
    for net_name in networks:
        print(f"\nProcessing neural network: {net_name.upper()}")
        print("-" * 30)
        
        try:
            detailed_pdf, summary_pdf = run_network_experiments(net_name, array_configs)
            print(f"✓ Completed: {net_name}")
            print(f"  - Detailed results: {detailed_pdf}")
            print(f"  - Summary: {summary_pdf}")
        except Exception as e:
            print(f"✗ Error processing {net_name}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("All experiments completed! PDF reports are saved in the 'duplicate' folder")

if __name__ == "__main__":
    main()