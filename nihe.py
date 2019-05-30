#encoding=utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

data = pd.read_csv('nihe.csv', header=0)
#
# plt.plot(data['x'], data['y1'])
# plt.plot(data['x'], data['y2'])
# plt.show()


#定义目标函数
#幂函数
def func(x, a, b):
    return x**a + b


#指数函数 pow() 方法返回 xy（x的y次方） 的值
def func1(x, a, b):
    return a*pow(x, b)

#三个参数拟合
def func2(x, a, b, c):
    return a * np.exp(b * x) + c



def getIndexes(y_predict, y_data):
    n = y_data.size
    # SSE为和方差
    SSE = ((y_data - y_predict) ** 2).sum()
    # MSE为均方差
    MSE = SSE / n
    # RMSE为均方根,越接近0，拟合效果越好
    RMSE = np.sqrt(MSE)

    # 求R方，0<=R<=1，越靠近1,拟合效果越好
    u = y_data.mean()
    SST = ((y_data - u) ** 2).sum()
    SSR = SST - SSE
    R_square = SSR / SST
    return SSE, MSE, RMSE, R_square

#
# pfit为估计参数值的期望
# perr为估计参数值的标准差
# print('pfit(case 1)', popt_1)
# print('perr(case 1)', np.sqrt(np.diag(pcov_1)))
# print('pfit(case 2)', popt_2)
# print('perr(case 2)', np.sqrt(np.diag(pcov_2)))
#
# print getIndexes(model1, data['y1'])
# print getIndexes(model1, data['y2'])
# plt.plot(model1, label='model1')
# plt.plot(model2, label='model2')
# plt.plot(data['x'], data['y1'], label='y1')
# plt.plot(data['x'], data['y2'], label='y2')
# plt.legend()
# plt.show()
#
#
print '*'*30+'多项式'+'*'*30

f1 = np.polyfit(data['x'], data['y1'], 3)#模型1
p1 = np.poly1d(f1) #公式
y1val = p1(data['x'])#模型1的拟合值
f2 = np.polyfit(data['x'], data['y2'], 3)#模型1
p2 = np.poly1d(f2) #公式
y2val = p2(data['x'])#模型1的拟合值
print 'f1' ,f1
print 'p1', p1
# print 'y1val', y1val
print 'f2' ,f2
print 'p2', p2
# print 'y2val', y2val
print 'y1: ', getIndexes(y1val, data['y1'])
print 'y2: ', getIndexes(y2val, data['y2'])


# 利用curve_fit作简单的拟合，popt为拟合得到的参数,pcov是参数的协方差矩阵
print '*'*30+'幂函数 x**a + b '+'*'*30
popt_1_1, pcov_1_1 = opt.curve_fit(func, data['x'], data['y1'])
popt_1_2, pcov_1_2 = opt.curve_fit(func, data['x'], data['y2'])
model1_1 = func(data['x'], *popt_1_1) #拟合值
model1_2 = func(data['x'], *popt_1_2)
print 'y1: ', popt_1_1
print 'y2: ', popt_1_2
# print np.sqrt(np.diag(pcov_1))各参数的标准差
print 'y1: ', getIndexes(model1_1, data['y1'])
print 'y2: ', getIndexes(model1_2, data['y2'])

print '*'*30+'指数函数 a*pow(x, b) '+'*'*30
popt_2_1, pcov_2_1 = opt.curve_fit(func1, data['x'], data['y1'])
popt_2_2, pcov_2_1 = opt.curve_fit(func1, data['x'], data['y2'])
model2_1 = func1(data['x'], *popt_2_1) #拟合值
model2_2 = func1(data['x'], *popt_2_2)
print 'y1: ', popt_2_1
print 'y2: ', popt_2_2
# print np.sqrt(np.diag(pcov_1))各参数的标准差
print 'y1: ', getIndexes(model2_1, data['y1'])
print 'y2: ', getIndexes(model2_2, data['y2'])

print '*'*30+'三个参数拟合 a * np.exp(b * x) + c '+'*'*30
popt_3_1, pcov_3_1 = opt.curve_fit(func2, data['x'], data['y1'])
popt_3_2, pcov_3_2 = opt.curve_fit(func2, data['x'], data['y2'])
model3_1 = func2(data['x'], *popt_3_1) #拟合值
model3_2 = func2(data['x'], *popt_3_2)
print 'y1: ', popt_3_1
print 'y2: ', popt_3_2
# print np.sqrt(np.diag(pcov_1))各参数的标准差
print 'y1: ', getIndexes(model3_1, data['y1'])
print 'y2: ', getIndexes(model3_2, data['y2'])

# 作图
fig = plt.figure(figsize=(10,10))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
ax1.plot(data['x'], y1val, label='m_1')
ax1.plot(data['x'], y2val, label='m_2')
ax1.plot(data['x'], data['y1'], label='y1')
ax1.plot(data['x'], data['y2'], label='y2')
ax1.set_title('polynomial')

ax2.plot(data['x'], model1_1, label='m1_1')
ax2.plot(data['x'], model1_2, label='m1_2')
ax2.plot(data['x'], data['y1'], label='y1')
ax2.plot(data['x'], data['y2'], label='y2')
ax2.set_title('x**a + b ')

ax3.plot(data['x'], model2_1, label='m1_1')
ax3.plot(data['x'], model2_2, label='m1_2')
ax3.plot(data['x'], data['y1'], label='y1')
ax3.plot(data['x'], data['y2'], label='y2')
ax3.set_title('a*pow(x, b)')

ax4.plot(data['x'], model3_1, label='m1_1')
ax4.plot(data['x'], model3_2, label='m1_2')
ax4.plot(data['x'], data['y1'], label='y1')
ax4.plot(data['x'], data['y2'], label='y2')
ax4.set_title('a * np.exp(b * x) + c')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.show()