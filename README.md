# Project_1-Dimensionality-Reduction
### Project Requirement

- 对AwA数据库的feature进行降维（可以自己找如何抽取Feature）
  - Zero-Shot learning/ one-shot/ few-shot：从北极熊分类器变成斑马分类器
  - 数据库还提供了attribute
  - 分出Training Sample（60%）和Test Sample（40%）
  - 决定如何Validation：K-fold cross-validation （把训练集分成k份，一份用于validation）

- 用Linear SVM用来分类图像（deep-learning feature）

- 至少有三种的降维方式
- 探索最优的降维的方式和维度
- 书写报告
  - 实验结果尽量多
  - 实验结果考虑越多分数越高
  
  
 姓名| 任务| 第一阶段 | 第二阶段 |
-|-|-|-
王皓轩 | 数据集整理 | 将图像数据类型数值化，并划分训练集、验证集和测试集合（找个时间一起看一下数据集，如果比较复杂可找另一个人分担）| SNE和t-SNE |
陈鸿滨 | 三种降维方式 | 先完成selection、PCA和LDA，将三种降维方式分别用类封装起来，方便调用 | Sparse Coding |
李杰宇 | 串联整个代码 | 完成Linear SVM分类器，并整合三种降维方式，把接口都写好、参数都调好 | MDS，整理新加入的降维器 |
李东岳 | 报告书写 | 写报告，包括绘制必要的图表，代码完成前可以先把框架写好，等待填数据 | LLE和补充实验的框架 |

# DDL：
+ 这周末数据集处理完，接口写好
+ 下周四之前第一阶段完成，讨论下接下来要做的所有实验
+ 下周末之前完成第二阶段，讨论接下来要做的补充实验
+ 4-6前完成report

# 试验结果：
降维方法 | 2 | 4 | 8 | 16 | 32 | 64（LDA：49） | 128 | 256 | 512 | 1024 |
-|-|-|-|-|-|-|-|-|-|-
LDA|23.85(23.80)|48.85(45.56)|74.57(70.88)|85.84（81.74）|90.65（86.87）|91.54（88.53）|-|-|-|-|
PCA|23.31|46.32|54.67|41.39|39.96|51.71|71.80|88.20|92.71|92.92|
t-SNE|9.33||||||||||
Sparse Coding|5.00|10.18|15.46||||||||
LLE|||||||||||
Auto-Encoder|||||||||||

