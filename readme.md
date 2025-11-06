# 一、单层映射方案

在main.ipynb文件里面

## 1.神经网络每层方案

由automapping函数来保存每层在可用资源下的非支配解集

#二、全局映射方案

high_speed_main.py low_speed_main.py文件都是用来进行全局搜索的

##2.全局搜索函数分为三类：

###2.1 基础版本

在function.py文件下，暴力、回溯、剪枝加回溯；

###2.2 多进程版本

在parallel_optimized_search.py文件下，也是有暴力、回溯、剪枝+回溯三种的多进程版本；

### 2.3 消融实验可直接运行

```
python ablation_exp.py --network vgg16 --resource 60
```

# 三、布局通信优化设计

## 3.通信比较（遗传算法 VS 传统顺序布局）

### 3.1 遗传算法

```python
cd master_code 
cd Genetic_A
python main_GA_sim.py -network Resnet20 
```

### 3.2传统算法

```
cd master_code 
cd Interconnect
python main_sqm_sim.py -network Resnet20 
```

