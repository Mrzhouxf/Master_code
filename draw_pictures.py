import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 读取 Excel 数据
df = pd.read_excel('auto_vgg161.xlsx', sheet_name='Sheet1')

# 设置全局字体大小 
plt.rcParams.update({ 'font.size': 14, # 基础字体大小 
                     'axes.titlesize': 16, # 子图标题大小 
                     'axes.labelsize': 15, # 坐标轴标签大小 
                     'legend.fontsize': 11, # 图例字体大小 
                     'xtick.labelsize': 13, # x轴刻度标签大小 
                     'ytick.labelsize': 13, # y轴刻度标签大小 
                     'font.weight':'bold',
                     'axes.titleweight':'bold'
                     })

# 定义颜色映射，假设 Total resource 有五种不同的值（可根据实际数据调整）
resource_values = df['Total resource'].unique()
color_map = plt.cm.get_cmap('tab10')
colors = color_map(range(len(resource_values)))
resource_color = {rv: c for rv, c in zip(resource_values, colors)}

# 创建一个 2x2 的子图布局
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# 定义一个函数来提亮颜色
def darken_color(color, factor=0.3):
    r, g, b, a = color
    r = max(0, r * factor)
    g = max(0, g * factor)
    b = max(0, b * factor)
    return (r, g, b, a)
def brighten_color(color, factor=1.5):
    r, g, b, a = color
    r = min(1, r * factor)
    g = min(1, g * factor)
    b = min(1, b * factor)
    return (r, g, b, a)

# 绘制四个子图的函数
def plot_subplot(ax, title, label, special_points=[],is_bottom_right=False):
    # 绘制不同 Total resource 的散点
    for rv in resource_values:
        subset = df[df['Total resource'] == rv]
        brightened_color = darken_color(resource_color[rv])
        ax.scatter(subset['Total compute cycles'], subset['Total data transmission'],
                   c=[brightened_color], label=f'Total resource: {rv}', alpha=0.6, edgecolors='w')

    # 标记特殊点 resource_color[target_resource]
    for i,(point, marker, color, label_text) in enumerate(special_points):
        x, y = point
        target_point = df[(df['Total compute cycles'] == x) & (df['Total data transmission'] == y)]
        if not target_point.empty:
            target_resource = target_point['Total resource'].values[0]
            ax.scatter(target_point['Total compute cycles'], target_point['Total data transmission'],
                       c=color, marker=marker, s=150, label=label_text)
            target_coord = (int(target_point['Total compute cycles'].values[0]), int(target_point['Total data transmission'].values[0]))
            print(f"Coordinates of the marked point {label_text}:", target_coord)
            if is_bottom_right:
                # 第一个点（D形）向上偏移，第二个点（p形）向右偏移
                if i == 0:
                    ax.annotate(f'{target_coord}', target_coord, 
                                textcoords="offset points", xytext=(0, 20), ha='center')  # 上移
                else:
                    ax.annotate(f'{target_coord}', target_coord, 
                                textcoords="offset points", xytext=(20, 0), ha='left')   # 右移
            else:
                # 其他子图保持原有偏移
                ax.annotate(f'{target_coord}', target_coord, 
                            textcoords="offset points", xytext=(0, 10), ha='center')
            # ax.annotate(f'{target_coord}', target_coord, textcoords="offset points", xytext=(0, 10), ha='center')
        else:
            print(f"Point {point} not found")

    # 设置坐标轴范围
    ax.set_xlim(75000, 91000)
    ax.set_ylim(30000000, 41000000)

    # 添加标签和标题
    ax.set_xlabel('Total compute cycles')
    ax.set_ylabel('Total data transmission')
    ax.set_title(title)

    # 添加图例到左上角
    ax.legend(loc='upper left',framealpha=0.7)

    # ax.set_alpha(0.1)

# 第一个子图（原左图）
special_points_1 = [
    ((77102, 35187024), '*', 'blue', 'OURS'),
    ((77102, 32972352), '^', 'red', 'VW-SDK')
]
plot_subplot(axes[0, 0], 'Global optimal point found by different methods', 'Old marked points', special_points_1)

# 第二个子图（原右图）
min_compute_cycle = df['Total compute cycles'].min()
min_transfer_at_min_cycle = df[df['Total compute cycles'] == min_compute_cycle]['Total data transmission'].min()
new_point_1 = df[(df['Total compute cycles'] == min_compute_cycle) & (
        df['Total data transmission'] == min_transfer_at_min_cycle)]
coord_new1 = (new_point_1['Total compute cycles'].values[0], new_point_1['Total data transmission'].values[0])

min_transfer = df['Total data transmission'].min()
min_cycle_at_min_transfer = df[df['Total data transmission'] == min_transfer]['Total compute cycles'].min()
new_point_2 = df[(df['Total compute cycles'] == min_cycle_at_min_transfer) & (
        df['Total data transmission'] == min_transfer)]
coord_new2 = (new_point_2['Total compute cycles'].values[0], new_point_2['Total data transmission'].values[0])

special_points_2 = [
    (coord_new1, '*', 'green', 'High-throughput'),
    (coord_new2, '^', 'orange', 'Low communication latency')
]
plot_subplot(axes[0, 1], 'High throughput and Low latency points', 'New marked points', special_points_2)

# 第三个子图
special_points_3 = [
    ((77102, 35187024), 'o', 'red', 'VW-SDK'),
    ((77102, 32225856), 's', 'blue', 'OURS')
]
plot_subplot(axes[1, 0], 'Points with less data transmission during the similar cycles', 'Specific points', special_points_3)

# 第四个子图：寻找总计算周期*0.5+总数据传输量*0.5的最小点和总计算周期*0.7+总数据传输量*0.3的最小点
# 数据量化到 0 到 1
x_min = df['Total compute cycles'].min()
x_max = df['Total compute cycles'].max()
y_min = df['Total data transmission'].min()
y_max = df['Total data transmission'].max()
df['Total compute cycles_norm'] = (df['Total compute cycles'] - x_min) / (x_max - x_min)
df['Total data transmission_norm'] = (df['Total data transmission'] - y_min) / (y_max - y_min)

# 计算权重得分
df['score_05_05'] = 0.5 * df['Total compute cycles_norm'] + 0.5 * df['Total data transmission_norm']
df['score_07_03'] = 0.7 * df['Total compute cycles_norm'] + 0.3 * df['Total data transmission_norm']

# 找到得分最小的点
point_05_05 = df[df['score_05_05'] == df['score_05_05'].min()]
point_07_03 = df[df['score_07_03'] == df['score_07_03'].min()]

# 反量化坐标
point_05_05_x = point_05_05['Total compute cycles_norm'].values[0] * (x_max - x_min) + x_min
point_05_05_y = point_05_05['Total data transmission_norm'].values[0] * (y_max - y_min) + y_min
point_07_03_x = point_07_03['Total compute cycles_norm'].values[0] * (x_max - x_min) + x_min
point_07_03_y = point_07_03['Total data transmission_norm'].values[0] * (y_max - y_min) + y_min

special_points_4 = [
    ((point_05_05_x, point_05_05_y), 'D', 'gold', 'cycles * 0.5 + transmission * 0.5'),
    ((point_07_03_x, point_07_03_y), 'p', 'teal', 'cycles * 0.7 + transmission * 0.3')
]
plot_subplot(axes[1, 1], 'Optimization points under different weights', 'Points with minimum weighted scores', special_points_4,is_bottom_right=True)

# 显示网格
for i in range(2):
    for j in range(2):
        axes[i, j].grid(True, linestyle='--', alpha=0.7)

# 调整布局
with PdfPages('scatter_plots161.pdf') as pdf:
    pdf.savefig(fig)
plt.tight_layout()
plt.show()
    






# 原始数据，去掉 Im2col
data1 = {
    'Model': ['ResNet20', 'SqueezeNet', 'ResNet110', 'VGG16'],
    'SDK': [481900, 1381648, 2711020, 46322224],
    'VW - SDK': [384488, 1294576, 2100968, 35187024],
    'OURS': [356528, 1088320, 1923248, 32972352]
}

# 新数据，去掉 Im2col
data2 = {
    'Model': ['ResNet20', 'SqueezeNet', 'ResNet110', 'VGG16'],
    'SDK': [1120, 14843, 6130, 114697],
    'VW - SDK': [804, 7802, 4344, 77102],
    'OURS': [814, 7840, 4414, 77102]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 创建一个包含两个子图的画布
fig, axes = plt.subplots(1, 2, figsize=(24, 8))

# 第一个子图：原始数据
ax1 = axes[0]
# 绘制除VGG16外其他模型的柱状图
df1_except_vgg16 = df1[df1['Model'] != 'VGG16']
bar_width = 0.2
index = range(len(df1_except_vgg16))
for i, col in enumerate(df1_except_vgg16.columns[1:]):
    if col == 'OURS':
        ax1.bar([pos + i * bar_width for pos in index], df1_except_vgg16[col], width=bar_width,
                label=col, color='red', alpha=0.7)
    else:
        ax1.bar([pos + i * bar_width for pos in index], df1_except_vgg16[col], width=bar_width,
                label=col, alpha=0.7)

# 设置主坐标轴标签等
ax1.set_ylabel('Total data transmission(In addition to VGG16)', color='blue')
# 设置标题并指定y参数确保对齐
ax1.set_title('Comparison of total data transmission of different models under various methods', y=1.05)
# 调整 x 轴刻度位置，包含 VGG16 的位置
x_ticks_pos = [pos + bar_width * (len(df1.columns) - 2) / 2 for pos in index] + [len(df1_except_vgg16) + bar_width * (len(df1.columns) - 2) / 2]
# 调整 x 轴刻度标签，包含 VGG16
x_tick_labels = list(df1_except_vgg16['Model']) + ['VGG16']
ax1.set_xticks(x_ticks_pos)
ax1.set_xticklabels(x_tick_labels)
ax1.tick_params(axis='y', labelcolor='blue')

# 创建副坐标轴
ax1_secondary = ax1.twinx()
# 绘制VGG16的柱状图
vgg16_data1 = df1[df1['Model'] == 'VGG16'].set_index('Model')
for i, col in enumerate(vgg16_data1.columns):
    if col == 'OURS':
        ax1_secondary.bar([len(df1_except_vgg16) + i * bar_width], vgg16_data1[col].values[0], width=bar_width,
                          color='red', alpha=0.7)
    else:
        ax1_secondary.bar([len(df1_except_vgg16) + i * bar_width], vgg16_data1[col].values[0], width=bar_width,
                          alpha=0.7)

# 设置副坐标轴标签等
ax1_secondary.set_ylabel('Total data transmission(VGG16)', color='green')
ax1_secondary.tick_params(axis='y', labelcolor='green')

# 添加图例（保持原位置）
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_secondary.get_legend_handles_labels()
ax1_secondary.legend(lines + lines2, labels + labels2, loc='upper left')

# 第二个子图：新数据
ax2 = axes[1]
# 绘制除VGG16外其他模型的柱状图
df2_except_vgg16 = df2[df2['Model'] != 'VGG16']
bar_width = 0.2
index = range(len(df2_except_vgg16))
for i, col in enumerate(df2_except_vgg16.columns[1:]):
    if col == 'OURS':
        ax2.bar([pos + i * bar_width for pos in index], df2_except_vgg16[col], width=bar_width,
                label=col, color='red', alpha=0.7)
    else:
        ax2.bar([pos + i * bar_width for pos in index], df2_except_vgg16[col], width=bar_width,
                label=col, alpha=0.7)

# 设置主坐标轴标签等
ax2.set_ylabel('Total compute cycles(In addition to VGG16)', color='blue')
# 设置标题并指定相同的y参数确保对齐
ax2.set_title('Comparison of total compute cycles of different models under various methods', y=1.05)
# 调整 x 轴刻度位置，包含 VGG16 的位置
x_ticks_pos = [pos + bar_width * (len(df2.columns) - 2) / 2 for pos in index] + [len(df2_except_vgg16) + bar_width * (len(df2.columns) - 2) / 2]
# 调整 x 轴刻度标签，包含 VGG16
x_tick_labels = list(df2_except_vgg16['Model']) + ['VGG16']
ax2.set_xticks(x_ticks_pos)
ax2.set_xticklabels(x_tick_labels)
ax2.tick_params(axis='y', labelcolor='blue')

# 创建副坐标轴
ax2_secondary = ax2.twinx()
# 绘制VGG16的柱状图
vgg16_data2 = df2[df2['Model'] == 'VGG16'].set_index('Model')
for i, col in enumerate(vgg16_data2.columns):
    if col == 'OURS':
        ax2_secondary.bar([len(df2_except_vgg16) + i * bar_width], vgg16_data2[col].values[0], width=bar_width,
                          color='red', alpha=0.7)
    else:
        ax2_secondary.bar([len(df2_except_vgg16) + i * bar_width], vgg16_data2[col].values[0], width=bar_width,
                          alpha=0.7)

# 设置副坐标轴标签等
ax2_secondary.set_ylabel('Total compute cycles(VGG16)', color='green')
ax2_secondary.tick_params(axis='y', labelcolor='green')

# 添加图例（保持原位置）
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_secondary.get_legend_handles_labels()
ax2_secondary.legend(lines + lines2, labels + labels2, loc='upper left')

# 调整布局避免标签重叠等问题
plt.tight_layout()

# 保存为 PDF
pdf_path = 'experimental_results2.pdf'
with PdfPages(pdf_path) as pdf:
    pdf.savefig(fig)

print(f"图表已保存为 {pdf_path}")
    