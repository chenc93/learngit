#encoding=utf8
'''
OLS回归分析（R）
参考文档：https://www.cnblogs.com/ljhdo/p/4807068.html
机器学习之线性回归附Python代码 http://blog.sciencenet.cn/blog-1966190-1119186.html
Python环境下的8种简单线性回归算法 https://www.jiqizhixin.com/articles/2018-01-01
Python Statsmodels 统计包之 OLS 回归 https://zhuanlan.zhihu.com/p/22692029
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #可以拟合一元多项式
import scipy.optimize as opt #拟合曲线的包
import statsmodels.api as sm
from statsmodels.formula.api import ols #线性回归

filename = 'nihe.csv'
data = pd.read_csv(filename, header=0)
data1 = data[['x','y1']]
data2 = data[['x','y2']]
lm_s_1 = ols('y1 ~ x', data=data1).fit()
lm_s_1.params#显示参数
lm_s_1.summary()