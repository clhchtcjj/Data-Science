# -*- coding: utf-8 -*-
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


# 观察数据
train_data = pd.read_csv(r'data/train.csv')
test_data = pd.read_csv(r'data/test.csv')

sns.set_style('whitegrid')
print(train_data.head())

