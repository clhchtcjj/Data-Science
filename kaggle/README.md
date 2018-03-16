# 数据挖掘建模过程和常用技术总结



> 本文主要参考知乎大树先生[Kaggle Titanic 生存预测 -- 详细流程吐血梳理](https://zhuanlan.zhihu.com/p/31743196)和《Python数据分析与挖掘实战》以期对数据分析、挖掘建模有一个初步认识。
>
> 文件Titanic.py基本都是大树先生的源码，只不过加了一些便于自己理解的内容。



## 1. 定义挖掘目标

针对具体的数据挖掘应用需求，首先要明确本次的挖掘目标是什么？系统完成后期望达到什么样的效果。

因此，我们需要分析应用领域，包括应用中的各种知识和应用目标，了解相关领域情况，熟悉背景知识，弄清用户需求。想要充分发挥数据挖掘的价值，必须对目标有一个清晰明确的定义。

- 任务理解
- 指标确定



## 2. 数据采集

在明确了需要进行数据挖掘的目标后，接下来就是从业务系统中抽取一个与挖掘目标相关的样本数据子集。

抽取数据的标准：1 相关性；2 可靠性；3 有效性

通过对样本数据的精选，不仅能减少数据处理量，节省系统资源，还可以使得我们想要寻找的规律更加凸显出来。

- 建模抽样
- 质量把控
- 实时采集



## 3. 数据探索

当我们拿到一个数据样本集后：

- 首先，需要验证它是否达到我们原来设想的要求；
- 其次，要探寻样本中是否存在明显的规律和趋势；
- 接着，观察是否出现从未设想过的数据状态；
- 还有，属性之间是否存在什么相关性；

这一阶段的工作，是保证最终挖掘模型质量的必须。

下面说说，在这一阶段，我常用的方法。

我主要使用python，用到的包有matplotlib、numpy、pandas和seaborn。

#### 导入相关包：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，在大多数情况下使用seaborn就能做出很具有吸引力的图。
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

#### 读取数据：

```python
train_data = pd.read_csv(r'data/train.csv')
test_data = pd.read_csv(r'data/test.csv')
```

#### 观测数据

```python
sns.set_style('whitegrid')
train_data.head() # 默认显示前五条记录
train_data.info()
test_data.info()
```

info()主要显示结果如下，从中我们可以发现那些列是包含缺失值

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```

#### 数据探索

1. 首先需要对缺失值处理，常用方法有：均值/中位数/众数插补；使用固定值；最近邻插补；**回归方法（根据已有的数据和其他有关的变量的数据建立拟合模型来预测缺失的数值）；插值法（构建插值函数：拉格朗日插值法和牛顿插值法）**

   显然，用回归分析对某一变量进行缺失值处理时，要先对与其相关的变量进行处理。

   ```python
   # 固定值
   # 对Cabin填充一个表示缺失值的值
   train_data['Cabin'] = train_data.Cabin.fillna('U0')

   # 众数
   train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().iloc[0] #可能有多个众数，所以要用.iloc[0]

   # 回归
   # Age属性对预测结果有很大的影响，所以我们采用模型来预测缺失值，如回归、随机森林
   # 首先，选取对预测Age有影响的特征：Age、Survived、Fare、Parch、Sibsp、Pclass构建训练集
   from sklearn.ensemble import RandomForestRegressor

   age_df = train_data[['Age','Survived','Fare','Parch','SibSp','Pclass']]
   age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
   age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
   X = age_df_notnull.values[:,1:]
   Y = age_df_notnull.values[:,0]
   # 这里使用随机森林进行年龄预测
   RF = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
   RF.fit(X,Y)
   pre_ages = RF.predict(age_df_isnull.values[:,1:])
   # 前面是条件后面是列名
   train_data.loc[(train_data['Age'].isnull()),['Age']] = pre_age
   ```

2. 通过图表，分析数据异常情况、趋势以及属性间相关性

   **2.1 异常值分析**

   （1）简单统计分析

   ​	统计量的分析，观测是否超出常识范围（年龄，性别等）

   （2）3$\delta$ 原则（$\delta$ 为标准差）

   ​	如果数据服从正态分布，那么一组测定值与平均值的偏差不会超过3$\delta$ 

   （3）箱型图

   ​	箱型图提供了识别异常值的一个标准：异常值通常被定义为小于$Q_L-1.5IQR$或大于$Q_U+1.5IQR$ 其中：$Q_L$为下四分位数，$Q_U$为上四分位数$ 为上下四分位数之差。

   ​	主要包含五个数据节点，将一组数据从大到小排列，分别计算出他的上边缘，上四分位数，中位数，下四分位数，下边缘。

   ​	![](http://img.mp.sohu.com/upload/20170812/874e638096164060b0a4e5a64bb450ea_th.png)

   ​	具体用法如下：seaborn.boxplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, fliersize=5, linewidth=None, whis=1.5, notch=False, ax=None, **kwargs)

   ​	参数说明：

   ​		**x, y, hue**: names of variables in data or vector data, optional #设置 x,y 以及颜色控制的变量

   ​		**data** : DataFrame, array, or list of arrays, optional #设置输入的数据集

   ​		**order, hue_order** : lists of strings, optional #控制变量绘图的顺序

   （4）琴型图

   ​	Violinplot 结合了箱线图与核密度估计图的特点，它表征了在一个或多个分类变量情况下，连续变量数据的分布并进行了比较，它是一种观察多个数据分布有效方法。

   ![](http://img.mp.sohu.com/upload/20170812/8c4f735de6ff4a2389a2057d1af7bb47_th.png)

   ​	使用方法：seaborn.violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None, bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box', split=False, orient=None, linewidth=None, color=None, palette=None, saturation=0.75, ax=None, **kwargs)

   ​		**split**: bool, optional #琴形图是否从中间分开两部分

   ​		**scale**: {“area”, “count”, “width”}, optional #用于调整琴形图的宽带。

   ​			    area——每个琴图拥有相同的面域；

   ​                            count——根据样本数量来调节宽度；

   ​                            width——每个琴图则拥有相同的宽度。

   **异常值处理方法：**

   - 删除含有异常值的记录

   - 视为缺失值

   - 平均值修正

   - 不处理

     ​

   **2.2 数据特征分析**

   （1）分布分析：对于定量数据，欲了解其分布形式是对称的还是非对称的，发现某些特大或特小的可疑值，可通过绘制频率分布表、绘制频率分布直方图、绘制茎叶图进行直观的分析；对于定性分类数据，可有用饼图、条形图直观的显示分布情况


```
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
```

![](https://pic2.zhimg.com/80/v2-2e386d4191e85b34fde4f574b83a6447_hd.jpg)

```python
# 查看分布
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
```

![](https://pic2.zhimg.com/v2-0794672a98de3cb38d739a70ed65ba15_r.jpg)

```
# 不同年龄下的平均生存率
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Age_int'] = train_data["Age"].astype(int)
average_age = train_data[['Age_int','Survived']].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age)
```

![](https://pic2.zhimg.com/v2-d686344e2923fe07b25a503ea625385d_r.jpg)

（2）对比分析

（3）统计量分析

- 集中趋势度量：均值、中位数、众数
- 离中趋势度量：极差、标准差、变异系数、四分位数间距

（4）周期性分析

（5）贡献度分析：帕累托法则

   **2.3 相关性分析**

（1）直接绘制散点图

​	判断两个变量是否具有线性相关最简单直观的方法是直接绘制散点图

（2）绘制散点图矩阵

​	同时需要考虑多个变量间的相关关系时，一一绘制是十分麻烦的。此时可以利用散点图矩阵绘制各变量间的散点图，从而快速发现多个变量间的主要相关性，这在进行多元线性回归时显得尤其重要。

​	使用方法：seaborn.pairplot(data, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='hist', markers=None, size=2.5, aspect=1, dropna=True, plot_kws=None, diag_kws=None, grid_kws=None)

​	参数说明：

​		vars : 与data使用，否则使用data的全部变量。参数类型：numeric类型的变量list。

​		{x, y}_vars : 与data使用，否则使用data的全部变量。参数类型：numeric类型的变量list。

​		dropna : 是否剔除缺失值。参数类型：boolean, optional

​		kind : {‘scatter’, ‘reg’}, optional Kind of plot for the non-identity relationships.

​		diag_kind : {‘hist’, ‘kde’}, optional。Kind of plot for the diagonal subplots.

​		size : 默认 6，图的尺度大小（正方形）。参数类型：numeric

​		hue : 使用指定变量为分类变量画图。参数类型：string (变量名)

​		hue_order : list of strings Order for the levels of the hue variable in the palette

​		palette : 调色板颜色

​		markers : 使用不同的形状。参数类型：list

​		aspect : scalar, optional。Aspect * size gives the width (in inches) of each facet.

​		{plot, diag, grid}_kws : 指定其他参数。参数类型：dicts

更详细的可以参考https://www.jianshu.com/p/6e18d21a4cad

```
g = sns.pairplot(combine_train_test_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
    u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived', palette = 'seismic',size=1.2, diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[]) 设置x轴图例为空值
```

![](https://pic1.zhimg.com/80/v2-aa36fc3830eda1a330cc255cdcdbb161_hd.jpg)

（3）计算皮尔逊相关系数

​	使用方法：seaborn.heatmap(data, vmin=None, vmax=None, cmap=None, center=None, robust=False, annot=None, fmt='.2g', annot*kws=None, linewidths=0, linecolor='white', cbar=True, cbar*kws=None, cbar_ax=None, square=False, ax=None, xticklabels=True, yticklabels=True, mask=None, **kwargs)

​	参数说明：

​		data：矩阵数据集，可以使numpy的数组（array），如果是pandas的dataframe，则df的index/column信息会分别对应到heatmap的columns和rows

​		linewidths,热力图矩阵之间的间隔大小

​		vmax,vmin, 图例中最大值和最小值的显示值，没有该参数时默认不显示

​		cmap：matplotlib的colormap名称或颜色对象；如果没有提供，默认为cubehelix map (数据集为连续数据集时) 或 RdBu_r (数据集为离散数据集时)

​		center：将数据设置为图例中的均值数据，即图例中心的数据值；通过设置center值，可以调整生成的图像颜色的整体深浅；设置center数据时，如果有数据溢出，则手动设置的vmax、vmin会自动改变

​		xticklabels: 如果是True，则绘制dataframe的列名。如果是False，则不绘制列名。如果是列表，则绘制列表中的内容作为xticklabels。 如果是整数n，则绘制列名，但每个n绘制一个label。 默认为True。

​		yticklabels: 如果是True，则绘制dataframe的行名。如果是False，则不绘制行名。如果是列表，则绘制列表中的内容作为yticklabels。 如果是整数n，则绘制列名，但每个n绘制一个label。 默认为True。默认为True

​		annotate的缩写，annot默认为False，当annot为True时，在heatmap中每个方格写入数据

​		annot_kws，当annot为True时，可设置各个参数，包括大小，颜色，加粗，斜体字等

```python
Correlation = pd.DataFrame(combined_train_test[
 ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 
  'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
```

![](https://pic3.zhimg.com/v2-67e0f1fec4c1208099f1121a4302feb3_r.jpg)

还有Spearman秩相关系数、判定系数

## 4. 数据预处理

#### 处理缺失值

#### 处理异常值

#### *数据集成*：多个源头的数据源进行汇总，去除冗余

#### 数据变换

（1）简单函数变换

​	平方、开方、取对数、差分运算

​	简单函数变换常用来将不具有正态分布的数据变成具有正态分布的数据。在时间序列分析中，有时简单的对数变化或差分运算就可以将非平稳序列转换成平稳序列。

（2）规范化

​	最小-最大规范化、零-均值规范化、小数定标规范化

```python
# 数据归一化
scale_age_fare = preprocessing.StandardScaler().fit(combine_train_test_df[['Age','Fare', 'Name_length']])
combine_train_test_df[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combine_train_test_df[['Age','Fare', 'Name_length']])

```

（3）连续属性离散化

​	等宽法、等频法、基于聚类的分析方法

```python
# 2. Emarked
combine_train_test_df['Embarked'].fillna(combine_train_test_df['Embarked'].mode().iloc[0],inplace=True)
combine_train_test_df['Embarked'] = pd.factorize(combine_train_test_df['Embarked'])[0]
# 获取one-hot编码
emb_dummies_df = pd.get_dummies(combine_train_test_df['Embarked'],prefix=combine_train_test_df[['Embarked']].columns[0])
combine_train_test_df = pd.concat([combine_train_test_df,emb_dummies_df],axis=1)
---------
# 4. Name
# 从名字中提取称呼
combine_train_test_df['Title'] = combine_train_test_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# # print(combine_train_test_df['Title'])
title_dict = {}
title_dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
print(title_dict)
combine_train_test_df['Title'] = combine_train_test_df['Title'].map(title_dict)
print(combine_train_test_df['Title'])
# one-hot
combine_train_test_df.loc[:,'Title'] = pd.factorize(combine_train_test_df['Title'])[0]
emb_dummies_df = pd.get_dummies(combine_train_test_df['Title'],prefix=combine_train_test_df[['Title']].columns[0])
combine_train_test_df = pd.concat([combine_train_test_df,emb_dummies_df],axis=1)
print(combine_train_test_df['Title'])
------------------------------------------------
from sklearn.preprocessing import LabelEncoder
# Pclass
# 建立Pclass类别转换函数
def pclass_fare_category(df,pclass1_mean_fare,plcass2_mean_fare,pclass3_mean_fare):
    if df['Pclass'] == 1:
        if df['Fare'] <= pclass1_mean_fare:
            return 'Pclass1_low'
        else:
            return 'Pclass1_high'
    elif df['Pclass'] == 2:
        if df['Fare'] <= plcass2_mean_fare:
            return 'Pclass2_low'
        else:
            return 'Pclass2_high'
    if df['Pclass'] == 3:
        if df['Fare'] <= pclass3_mean_fare:
            return 'Pclass3_low'
        else:
            return 'Pclass3_high'
# print(combine_train_test_df['Pclass'])
Pclass1_mean_fare = combine_train_test_df["Fare"].groupby(by=combine_train_test_df['Pclass']).mean().get([1]).values[0]
Pclass2_mean_fare = combine_train_test_df["Fare"].groupby(by=combine_train_test_df['Pclass']).mean().get([2]).values[0]
Pclass3_mean_fare = combine_train_test_df["Fare"].groupby(by=combine_train_test_df['Pclass']).mean().get([3]).values[0]
combine_train_test_df['Pclass_Fare_Category'] = combine_train_test_df.apply(pclass_fare_category, args=(
 Pclass1_mean_fare, Pclass2_mean_fare, Pclass3_mean_fare), axis=1)
combine_train_test_df['Pclass_Fare_Category']
pclass_level = LabelEncoder()
pclass_level.fit(np.array(
 ['Pclass1_low', 'Pclass1_high', 'Pclass2_low', 'Pclass2_high', 'Pclass3_low', 'Pclass3_high']))
combine_train_test_df['Pclass_Fare_Category'] = pclass_level.transform(combine_train_test_df['Pclass_Fare_Category'])
pclass_dummies_df = pd.get_dummies(combine_train_test_df['Pclass_Fare_Category']).rename(columns=lambda x: 'Pclass_' + str(x))
combine_train_test_df = pd.concat([combine_train_test_df, pclass_dummies_df], axis=1)
combine_train_test_df['Pclass'] = pd.factorize(combine_train_test_df['Pclass'])[0]
------------------------
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'],bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
```

（4）属性构造

（5）小波变化？？

#### 数据归约

（1）属性归约

​	合并属性、逐步前向选择、逐步后向删除、决策树回归、主成分分析

（2）数值归约

​	有参数法：只需要存放参数

​	无参数法：直方图、聚类、采样

（3）通过模型预先训练选择重要特征

```python
# 选择特征
# 利用不同的模型来对特性进行选择，选出较为重要的特征
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_top_n_features(titanic_train_data_X,titanic_train_data_Y,top_n_features):
    # RF
    rf = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20,30]}
    rf_grid = model_selection.GridSearchCV(rf,rf_param_grid,n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X,titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature':list(titanic_train_data_X),'importance':rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:10]))
    
     # AdaBoost
    ada_est =AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:10]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:10]))
    
    # GradientBoosting
    gb_est =GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [500], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': gb_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature']
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:10]))
    
    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature']
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:10]))
    
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n , features_importance

feature_to_pick = 30
feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
```

## 5. 挖掘建模 & 模型的评价

样本抽取完成并经预处理后，接下来要考虑的问题是：本次建模属于数据挖掘应用中的哪类问题？选用哪种算法进行模型构建？

列举本数据的建模方法：

Boosting、Bagging、Stacking、Blending（Blending 和 Stacking 很相似，但同时它可以防止信息泄露的问题）

以Stacking为例

这里我们使用了两层的模型融合，Level 1使用了：RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM ，一共7个模型，Level 2使用了XGBoost使用第一层预测的结果作为特征对最终的结果进行预测。

**模型的评价：基于模型评价的结果可以自动选择最好的模型，此外，还能根据业务对模型进行解释和应用。**

**Level 1：**

Stacking框架是堆叠使用基础分类器的预测作为对二级模型的训练的输入。 然而，我们不能简单地在全部训练数据上训练基本模型，产生预测，输出用于第二层的训练。如果我们在Train Data上训练，然后在Train Data上预测，就会造成标签泄露。为了避免标签泄露，我们需要对每个基学习器使用K-fold，将K个模型对Valid Set的预测结果拼起来，作为下一层学习器的输入。

所以这里我们建立输出K-fold预测的方法：

```python
# Stacking-Level1
from sklearn.model_selection import KFold

ntrain = titanic_train_data_X.shape[0]
ntest = titanic_test_data_X.shape[0]
SEED = 0
NFOLDS = 7
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)

def get_out_folds(clf,x_train,y_train,x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS,ntest))
    for i ,(train_index,test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.fit(x_tr,y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i,:] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1) 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt',max_depth=6, 
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)
ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)
et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)
gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2, max_depth=5, verbose=0)
dt = DecisionTreeClassifier(max_depth=8)
knn = KNeighborsClassifier(n_neighbors = 2)
svm = SVC(kernel='linear', C=0.025)

x_train = titanic_train_data_X.values
x_test = titanic_test_data_X.values
y_train = titanic_train_data_Y.values

rf_oof_train, rf_oof_test = get_out_folds(rf, x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_out_folds(ada, x_train, y_train, x_test) # AdaBoost 
et_oof_train, et_oof_test = get_out_folds(et, x_train, y_train, x_test) # Extra Trees
gb_oof_train, gb_oof_test = get_out_folds(gb, x_train, y_train, x_test) # Gradient Boost
dt_oof_train, dt_oof_test = get_out_folds(dt, x_train, y_train, x_test) # Decision Tree
knn_oof_train, knn_oof_test = get_out_folds(knn, x_train, y_train, x_test) # KNeighbors
svm_oof_train, svm_oof_test = get_out_folds(svm, x_train, y_train, x_test) # Support Vector

print("Training is complete")
```

*Level 2：*

我们利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。

```python
x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)
from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')
```

#### 通过学习曲线评测模型

**偏差**

在训练误差和测试误差收敛并且相当高时，这实质上表示模型具有偏差。无论我们向其提供多少数据，模型都无法表示基本关系，因而出现系统性的高误差。

高偏差示例：此时，增加样本数将没有效果。因为模型本身出了问题。可能的问题是模型过于简单。

![高偏差](https://images0.cnblogs.com/blog/300615/201408/261834238914556.png)

**方差**

如果训练误差与测试误差之间的差距很大，这实质上表示模型具有高方差。与偏差模型不同的是，如果有更多可供学习的数据，或者能简化表示数据的最重要特征的模型，则通常可以改进具有方差的模型。

高方差示例：此时，增加训练样本数有可能会有很好的效果。

![](https://images0.cnblogs.com/blog/300615/201408/261842369856464.png)

欠拟合：验证集和训练集的loss都非常低

过拟合：训练集loss小，验证集loss大

```python
# 观测学习曲线
from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                       n_jobs=1, train_sizes=np.linspace(.1,1.0,5), verbose=0):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
    estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt    

X = x_train
Y = y_train

# RandomForest
rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2, 
              'max_features' : 'sqrt','verbose': 0}

# AdaBoost
ada_parameters = {'n_estimators':500, 'learning_rate':0.1}

# ExtraTrees
et_parameters = {'n_jobs': -1, 'n_estimators':500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}

# GradientBoosting
gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}

# DecisionTree
dt_parameters = {'max_depth':8}

# KNeighbors
knn_parameters = {'n_neighbors':2}

# SVM
svm_parameters = {'kernel':'linear', 'C':0.025}

# XGB
gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma':0.9, 'subsample':0.8, 
               'colsample_bytree':0.8, 'objective': 'binary:logistic', 'nthread':-1, 'scale_pos_weight':1}     

title = "Learning Curves"
plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, Y, cv=None,  n_jobs=4, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
plt.show()
plot_learning_curve(AdaBoostClassifier(**ada_parameters), title, X, Y, cv=None,  n_jobs=4, train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
plt.show()
```

![](https://pic1.zhimg.com/80/v2-bc61d3b4b1b4718012162927ec7a7a61_hd.jpg)