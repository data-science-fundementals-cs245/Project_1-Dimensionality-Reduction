# Feature Selection Results Analysis

## 1. Forward Selection

###A. 实验设置

实验中采用的 *evaluation* 函数为使用**部分数据集**跑svm（使用gridsearch在一定范围内自动搜寻适用于该维度设置下的最佳参数，gridsearch的参数设置与全数据集跑的svm相同），实验设置与全数据集跑的svm实验相同，即按6:4划分训练集和测试集，并使用相同的k-fold cross-validation ，evaluation的评判标准是 *evaluation* 函数中模型在测试集的准确度。

在试验中，超参数step代表每次加入特征子集的特征个数，实验中step的选择主要综合考虑特征选择的细粒度和时间成本，所以aim_acc为0.85和0.9的实验step选择了5。

超参数aim_acc表示目标准确度，一旦evaluation的准确度大于等于aim_acc，FS算法即停止

超参数max_dims表示一旦特征子集的特征个数达到max_dims，FS算法即停止

在特征选择的过程中，采用两种不同的选择策略，一是根据feature_importances降序选择（即先选择对分类贡献度高的特征），而是random（随机选择）。

### B. 结果分析

对结果的分析主要通过控制变量和对比来切入，不妨将Forward Selection 结果的三个表格从上到下标号为**a, b, c**。

- eval_acc 与acc相比总会小一点，可能是由于其只使用了部分数据集，样本比较少（模型欠拟合？）
- 表格**a,b**对比，可以分析不同特征选取策略的影响。random策略要达到同样的aim_acc需要更多的特征个数，且random策略无法达到比较高的eval_acc（可能原因：选取的特征包含一些不重要的特征影响分类，且使用的样本比较少，两者影响下使eval_acc的上限略低于acc）
- 表格**a,c**对比，可以分析evaluation数据集规模对evaluation准确度的影响。表格**c**使用了更小的数据集，可以看出其eval_acc与acc的差距比较大（小得多或者偶尔eval_acc>acc），相比之下，表格**a** 的eval_acc就更加接近acc
- 在运行FS算法的过程中，一开始eval_acc 随着dims的增加上升得很快，但eval_acc逐渐变大接近最佳的acc时，eval_acc随着dims的增加不停抖动

## 2. SelectKBest

单变量特征选择 (Univariate feature selection)，单变量特征选择的原理是分别单独的计算每个变量的某个统计指标，根据该指标来判断哪些指标重要，剔除那些不重要的指标。SelectKBest选出K个最重要的特征。

三种不同的评分函数取得效果有些小区别。

降维的目标维度很低时，mutual_info_classif（互信息）的效果好，当维度较高时，chi2的效果略好一些。造成这些差别的原理我不懂啊！



从整体上看，单变量特征选择比Forward Selection的效果好。