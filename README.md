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
t-SNE|9.33|11.8|||||||||
Sparse Coding|5.00|10.18|15.46|20.20|35.03|50.08|66.87|81.70|88.93|91.02|
LLE|||||||||||
Auto-Encoder|||||||||||

##Feature Selection
###SelectKBest
score_function | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
-|-|-|-|-|-|-|-|-|-|-|-
f_classif|15.8|28.73|43.02|63.07|78.61|87.05|89.83|91.6|92.47|93|
chi2|10.49|19.57|39.93|60.87|79.08|86.41|89.91|92.04|92.79|93.08|
mutual_info_classif|18.18|32.41|53.84|71.14|81.68|86.88|89.99|91.91|92.55|93.02|

降维花费时间mutual_info_classif大概是前两者的50-150倍

###Forward Selection
evaluation 使用数据为三分之一的训练集。

选取特征顺序为feature_importances的降序（使用Extra-Trees模型得到的特征对分类的重要度）：
max_dims = 1000(实验发现并不需要到1000维)
| aim_acc | 0.7   | 0.75  | 0.8   | 0.85  | 0.9   |
| ------- | ----- | ----- | ----- | ----- | ----- |
| step    | 2     | 2     | 2     | 5     | 5     |
| dims    | 26    | 36    | 48    | 80    | 320   |
| eva_acc | 71.77 | 75.23 | 80.15 | 85.72 | 90.24 |
| acc     | 72.41 | 77.56 | 81.7  | 85.85 | 90.47 |

选取特征顺序为Random:（max_dims = 1000)
| aim_acc | 0.7   | 0.75  | 0.8   | 0.85  | 0.9  |
| ------- | ----- | ----- | ----- | ----- | ---- |
| step    | 2     | 2     | 2     | 5     | 5    |
| dims    | 56    | 66    | 86    | 165   | 1000 |
| eva_acc | 70.2  | 76.03 | 80.56 | 85.32 | 87.7 |
| acc     | 71.08 | 76.63 | 80.94 | 85.6  | 90.9 |

