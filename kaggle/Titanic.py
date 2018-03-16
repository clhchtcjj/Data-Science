
# coding: utf-8

# In[4]:


__author__ = 'CLH'

'''
    概览数据
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，在大多数情况下使用seaborn就能做出很具有吸引力的图。
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')

# 观察数据
train_data = pd.read_csv(r'data/train.csv')
test_data = pd.read_csv(r'data/test.csv')

sns.set_style('whitegrid')
train_data.head()


# In[5]:

train_data.info()
print("-"*40)
test_data.info()


# In[13]:

# 结论：从info信息上来看，Age、Cabin、Embarked、Fare包含缺失值
# 下面对缺失值进行处理
# 1. Age
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
train_data.loc[train_data['Age'].isnull(),['Age']] = pre_ages
# #2. Cabin
# # 对Cabin填充一个表示缺失值的值
# train_data['Cabin'] = train_data.Cabin.fillna('U0')
# # 3. Embarked
# train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
# # train_data.info()
# # 绘制存活比例
# train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')


# In[15]:

#2. Cabin
# 对Cabin填充一个表示缺失值的值
train_data['Cabin'] = train_data.Cabin.fillna('U0')
# 3. Embarked
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data.info()
# 绘制存活比例
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')


# In[16]:

# 分析数据关系
# 1. 性别与是否生成的关系
train_data.groupby(['Sex','Survived'])['Survived'].count()


# In[17]:

train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
# 发现女性比男性生存率高


# In[18]:

# 2. 船舱等级和生存与否的关系
train_data.groupby(['Pclass','Survived'])['Pclass'].count()


# In[19]:

train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()


# In[22]:

train_data.groupby(['Sex','Pclass','Survived'])['Survived'].count()


# In[20]:

train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()


# In[23]:

# 3. 年龄与存活与否的关系
fig, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age",hue="Survived",data=train_data,split=True,ax=ax[0])
ax[0].set_title("Pclass and Age vs Survived")
ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age",hue="Survived",data=train_data,split=True,ax=ax[1])
ax[1].set_title("Sex and Age vs Survived")
ax[1].set_yticks(range(0,110,10))
plt.show()


# In[26]:

# 分析总体的年龄分布
plt.subplot(121)
train_data['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train_data.boxplot(column="Age",showfliers=True)
plt.show()


# In[27]:

# 通过这个曲线图可以看出不同特征值时的分布密度  
facet = sns.FacetGrid(train_data,hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train_data['Age'].max()))
facet.add_legend()


# In[31]:

# 不同年龄下的平均生存率
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Age_int'] = train_data["Age"].astype(int)
average_age = train_data[['Age_int','Survived']].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age)


# In[32]:

train_data['Age'].describe()


# In[33]:

# 按照年龄，将乘客划分为儿童，少年，成年和老年。分析这四个群体的生还情况
bins = [0, 12, 18, 65, 100]
train_data['Age_group'] = pd.cut(train_data['Age'],bins)
by_age = train_data.groupby('Age_group')['Survived'].mean()
by_age


# In[34]:

by_age.plot(kind='bar')


# In[35]:

# 称呼与存活与否的关系 Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
pd.crosstab(train_data['Title'],train_data['Sex'])


# In[36]:

train_data[['Title','Survived']].groupby(['Title']).mean().plot.bar()


# In[38]:

# 观测名字长度与生成率之间的关系
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
train_data['Name_length'] = train_data['Name'].apply(len)
name_length = train_data[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=name_length)
# 从图形上看，名字长度与生成与否确实也是存在一定相关性


# In[40]:

# 有无兄弟姐妹和存活与否的关系 SibSp
sibsp_df = train_data[train_data['SibSp']!=0]
no_sibsp_df = train_data[train_data['SibSp']==0]
plt.figure(figsize=(10,5))
plt.subplot(121)
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('sibsp')

plt.subplot(122)
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_sibsp')


# In[41]:

# 有无父母子女和存活与否的关系
parch_df = train_data[train_data['Parch'] != 0]
no_parch_df = train_data[train_data['Parch'] == 0]

plt.figure(figsize=(10,5))
plt.subplot(121)
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('parch')

plt.subplot(122)
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived', 'Survived'], autopct = '%1.1f%%')
plt.xlabel('no_parch')


# In[42]:

# 亲友的人数与存活与否的关系 SibSp & Parch
fig,ax = plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')


# In[44]:

train_data['Family_Size'] = train_data['Parch'] + train_data['SibSp'] + 1
train_data[['Family_Size','Survived']].groupby(['Family_Size']).mean().plot.bar()
# 亲友少和亲友太多，会导致存活率低。。。。


# In[46]:

# 票价分布与存活与否关系
plt.figure(figsize=(10,5))
train_data['Fare'].hist(bins=70)
train_data.boxplot(column='Fare',by='Pclass',showfliers=False)


# In[47]:

train_data['Fare'].describe()


# In[49]:

# 绘制生存与否与票价均值与方差的关系
fare_not_survived = train_data['Fare'][train_data['Survived']==0]
fare_survivred = train_data['Fare'][train_data['Survived']==1]

average_fare = pd.DataFrame([fare_not_survived.mean(),fare_survivred.mean()])
std_fare = pd.DataFrame([fare_not_survived.std(),fare_survivred.std()])
average_fare.plot(yerr=std_fare, kind='bar',legend=False)
# 票价与是否生还有一定的相关性，生还者的平均票价要大于未生还者的平均票价


# In[50]:

# 船舱类型和存活与否的关系 Cabin 由于缺失值比较多，不对此进行分析
# 港口和存活与否的关系 Embarked 
sns.countplot('Embarked',hue='Survived',data=train_data)
plt.title('Embarked and Survived')
sns.factorplot('Embarked','Survived',data= train_data,size=3,aspect=2)
plt.title('Embarked and Survived rate')


# In[52]:

# 数据分析结束之后，开始对数据进行转换
# 类别变量转换
# 1. Embark===dummy
embark_dummies = pd.get_dummies(train_data['Embarked'])
embark_dummies
train_data = train_data.join(embark_dummies)
train_data.drop(['Embarked'],axis=1,inplace=True)
train_data.head()


# In[62]:

# 2. Cabin===Factorizing可以创建一些数字，来表示类别变量，对每一个类别映射一个ID，这种映射最后只生成一个特征，不像dummy那样生成多个特征。
import re
train_data['Cabin'][train_data.Cabin.isnull()]='U0'
train_data['CabinLetter'] = train_data['Cabin'].map(lambda x: re.compile('([a-zA-Z]+)').search(x).group())
train_data['CabinLetter'] = pd.factorize(train_data['CabinLetter'])[0]
train_data['CabinLetter']


# In[63]:

# 定量转换
# Age
# 1. Scaling:将很大范围的数值映射到一个很小的范围
from sklearn import preprocessing
assert np.size(train_data['Age']) == 891

scaler = preprocessing.StandardScaler()
train_data['Age_scaled'] = scaler.fit_transform(train_data['Age'].values.reshape(-1,1))
train_data['Age_scaled'].head()


# In[65]:

# Binning: 将连续数据离散化，存储的值被分布到一些“桶”或“箱”中。
train_data['Fare_bin'] = pd.qcut(train_data['Fare'],5)
train_data['Fare_bin'].head()
# 在数据Binning化后，要么factorize要么dummies
train_data['Fare_bin_id'] = pd.factorize(train_data['Fare_bin'])[0]
fare_bin_dummies_df = pd.get_dummies(train_data['Fare_bin']).rename(columns=lambda x:'Fare_'+str(x))
trian_data = pd.concat([train_data,fare_bin_dummies_df],axis=1)



# In[151]:

# 特征工程
# 在训练模型之前，要对训练数据和测试数据同时进行预处理
train_df_org = pd.read_csv(r'data/train.csv')
test_df_org = pd.read_csv(r'data/test.csv')
test_df_org['Survived'] = 0
combine_train_test_df = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']


# In[102]:

combine_train_test_df.info()


# In[152]:

# 处理缺失值和离散化
# 1. Cabin
combine_train_test_df['Cabin'].fillna('U0')


# In[153]:

# 2. Emarked
combine_train_test_df['Embarked'].fillna(combine_train_test_df['Embarked'].mode().iloc[0],inplace=True)
combine_train_test_df['Embarked'] = pd.factorize(combine_train_test_df['Embarked'])[0]
# 获取one-hot编码
emb_dummies_df = pd.get_dummies(combine_train_test_df['Embarked'],prefix=combine_train_test_df[['Embarked']].columns[0])
combine_train_test_df = pd.concat([combine_train_test_df,emb_dummies_df],axis=1)


# In[154]:

# 3. Sex
combine_train_test_df['Sex'] = pd.factorize(combine_train_test_df['Sex'])[0]
emb_dummies_df = pd.get_dummies(combine_train_test_df['Sex'],prefix=combine_train_test_df[['Sex']].columns[0])
combine_train_test_df = pd.concat([combine_train_test_df,emb_dummies_df],axis=1)


# In[89]:

# combine_train_test_df = combine_train_test_df.drop(['Sex_0','Sex_1','Sex_2'],axis=1)


# In[91]:

# combine_train_test_df = combine_train_test_df.drop(['Sex_-1'],axis=1)


# In[94]:

# combine_train_test_df.head(5)


# In[189]:



# combine_train_test_df = combine_train_test_df.drop(['Title_0'],axis=1)
# combine_train_test_df.info()
# combine_train_test_df.head(5)
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


# In[190]:

combine_train_test_df.info()


# In[191]:

# 增加名字长度特征
combine_train_test_df['Name_length'] = combine_train_test_df['Name'].apply(len)


# In[192]:

# Fare
combine_train_test_df['Fare'] = combine_train_test_df[['Fare']].fillna(combine_train_test_df.groupby('Pclass').transform(np.mean))


# In[193]:

combine_train_test_df.info()


# In[158]:

# 将票价分配到每个人
combine_train_test_df['Group_Ticket'] = combine_train_test_df['Fare'].groupby(by=combine_train_test_df['Ticket']).transform('count')
combine_train_test_df['Fare'] = combine_train_test_df['Fare'] / combine_train_test_df['Group_Ticket']
combine_train_test_df.drop(['Group_Ticket'],axis=1,inplace=True)
# 使用binning给票价分等级
combine_train_test_df['Fare_bin'] = pd.qcut(combine_train_test_df['Fare'],5)
combine_train_test_df['Fare_bin_id'] = pd.factorize(combine_train_test_df['Fare_bin'])[0]
Fare_bin_dummies_df = pd.get_dummies(combine_train_test_df['Fare_bin_id']).rename(columns=lambda x:'Fare_'+str(x))
combine_train_test_df = pd.concat([combine_train_test_df,Fare_bin_dummies_df],axis=1)
combine_train_test_df.drop(['Fare_bin'],axis=1,inplace=True)


# In[160]:


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




# In[161]:

# Parch and SibSp
def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_family'
    else:
        return 'Large_family'
combine_train_test_df['Family_Size'] = combine_train_test_df['Parch']+combine_train_test_df['SibSp']+1
combine_train_test_df['Family_Size_Category'] = combine_train_test_df['Family_Size'].map(family_size_category)
le_family = LabelEncoder()
le_family.fit(np.array(['Single','Small_family','Large_family']))
combine_train_test_df['Family_Size_Category'] = le_family.transform(combine_train_test_df['Family_Size_Category'])
family_size_dummies_df = pd.get_dummies(combine_train_test_df['Family_Size_Category'],
                                     prefix=combine_train_test_df[['Family_Size_Category']].columns[0])
combine_train_test_df = pd.concat([combine_train_test_df, family_size_dummies_df], axis=1)



# In[162]:

# Age
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor

missing_age_df = pd.DataFrame(combine_train_test_df[
 ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']])

missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
# print(missing_age_df)
def fill_missing_age(missing_age_train,missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'],axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'],axis=1)
    
    # RF
    rf = RandomForestRegressor()
    rf_gram_grid = {'n_estimators':[100,200],'max_depth':[5,6,7],'random_state':[0]}
    rf_grid = model_selection.GridSearchCV(rf, rf_gram_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
    rf_grid.fit(missing_age_X_train,missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_grid.best_score_))
    print('RF Train Error for "Age" Feature Regressor' + str(rf_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test['Age_RF'] = rf_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])
    
    # gbdt
    gbdt = GradientBoostingRegressor()
    gbdt_gram_grid = {'n_estimators':[100,200],'max_depth':[5,6,7],'max_features':[3,4],'learning_rate':[0.01,0.02]}
    gbdt_grid = model_selection.GridSearchCV(gbdt,gbdt_gram_grid,cv=10,n_jobs=25,verbose=1,scoring='neg_mean_squared_error')
    gbdt_grid.fit(missing_age_X_train,missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbdt_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbdt_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(gbdt_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbdt_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test
combine_train_test_df.loc[(combined_train_test.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)


# In[195]:

# combine_train_test_df['Ticket_Letter'] = combine_train_test_df['Ticket'].str.split().str[0]
# combine_train_test_df['Ticket_Letter'] = combine_train_test_df['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)

# combine_train_test_df['Ticket_Letter'] = pd.factorize(combine_train_test_df['Ticket_Letter'])[0]
# combine_train_test_df['Cabin'] = combine_train_test_df['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)

combine_train_test_df.info()# = combine_train_test_df.drop(['Title_-1'],axis=1)
print(combine_train_test_df['Title'])


# In[196]:

# 分析特征之间的相关性
Correlation = pd.DataFrame(combine_train_test_df[
 ['Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass', 
  'Pclass_Fare_Category', 'Age', 'Ticket_Letter', 'Cabin']])
colormap = plt.cm.viridis
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features',y=1.05,size=15)
sns.heatmap(Correlation.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# In[197]:

g = sns.pairplot(combine_train_test_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
    u'Family_Size', u'Title', u'Ticket_Letter']], hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])


# In[198]:

# 数据预处理
# 数据归一化
scale_age_fare = preprocessing.StandardScaler().fit(combine_train_test_df[['Age','Fare', 'Name_length']])
combine_train_test_df[['Age','Fare', 'Name_length']] = scale_age_fare.transform(combine_train_test_df[['Age','Fare', 'Name_length']])
# 丢弃无用数据
combined_data_backup_df = combine_train_test_df
combine_train_test_df.drop(['PassengerId', 'Embarked', 'Sex', 'Name', 'Title', 'Fare_bin_id', 'Pclass_Fare_Category', 
                       'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],axis=1,inplace=True)
# 将训练数据和测试数据分开
train_data = combine_train_test_df[:891]
test_data = combine_train_test_df[891:]

titanic_train_data_X = train_data.drop(['Survived'],axis=1)
titanic_train_data_Y = train_data['Survived']
titanic_test_data_X = test_data.drop(['Survived'],axis=1)


# In[205]:

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
    
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt], 
                               ignore_index=True).drop_duplicates()
    
    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et, 
                                   feature_imp_sorted_gb, feature_imp_sorted_dt],ignore_index=True)
    
    return features_top_n , features_importance

feature_to_pick = 30
feature_top_n, feature_importance = get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])


# In[211]:

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


# In[212]:

x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)


# In[214]:

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


# In[208]:

from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 4, min_child_weight= 2, gamma=0.9, subsample=0.8, 
                     colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)
predictions = gbm.predict(x_test)


# In[210]:

StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
StackingSubmission.to_csv('StackingSubmission.csv',index=False,sep=',')


# In[ ]:



