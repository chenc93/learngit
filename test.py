#encoding=utf8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  #科学计算包
from scipy import stats

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

file_name = 'data.csv'
df = pd.read_csv(file_name, header=0)
print df.columns.values
sample = df['b_a_show_click']
sample_mean = sample.mean()
sample_std = sample.std()
print '样本均值：%.5f'% sample_mean
print '样本均值：%.5f'% sample_std


# fig,axes=plt.subplots(2,2) #创建一个一行三列的画布
# sns.distplot(sample,ax=axes[0,0])
# sns.distplot(df['show_click_a'],ax=axes[0,1])
# sns.distplot(df['show_click_b'],ax=axes[1,0])
# # plt.title('样本集分布')
# plt.show()

# a代表无标签，b代表有标签
# 单样本t检验 -- ttest_1samp
# H0:b-a<=0  H1：b-a>0
s_mean = 0.0014
t, p_2tailed = stats.ttest_1samp(sample, s_mean)
print 't值:%.5f,双尾p值：%.5f '%(t, p_2tailed)
p_1taied = p_2tailed/2
print '单样本t检验'
print 'H0:b-a<=0  H1：b-a>0'
print 't值:%.5f,右单尾p值：%.5f '%(t,p_1taied)


print "*"*100

# H0:a>=b  H1：a<b

# 方差齐性检验
print stats.levene(df['show_click_a'], df['show_click_b'])

# 两独立样本t检验 -- ttest_ind
# equal_var代表方差不相等
print df[['show_click_a','show_click_b']].mean(axis=0)
print df[['show_click_a','show_click_b']].std(axis=0)
t_2, p_2 = stats.ttest_ind(df['show_click_a'], df['show_click_b'], equal_var=False)
p_real = p_2/2
print '两独立样本t检验'
print 'H0:a>=b  H1：a<b'
print 't值:%.5f,左单尾p值：%.5f '%(t_2, p_real)

p = 0.05

if p_real > p:
    print '接受原假设，即无标签下载率优于有标签'
else:
    print '拒绝原假设，即无标签下载率低于有标签'
#
# def real_p(t, p):
#     if t>0:
#         p_real = 1-p/2
#     else:
#         p_real = p/2
#     return p_real
print '*'*100
# 配对样本t检验 -- ttest_rel
t_3, p_3 = stats.ttest_rel(df['show_click_a'], df['show_click_b'])
p_real = p_3/2
print '配对样本t检验'
print 'H0:a>=b  H1：a<b'
print 't值:%.5f,左单尾p值：%.5f '%(t_3, p_real)
