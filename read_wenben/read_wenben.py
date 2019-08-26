#encoding=utf8

import json
import pandas as pd

data=[]
with open('背景商品.txt') as fp:
    for line in fp:
        data.append(line)
        # print data

data1 = ''.join(data) #转成字符串

s = json.loads(data1)
# data2 = ''.join(s['productInfos'])
data2= s['productInfos']
print len(data2)
print data2[0]['title']
print data2[0]['productId']
title =[]
productid=[]
for i in range(len(data2)):
    title.append(data2[i]['title'])
    productid.append(data2[i]['productId'])


print productid
print title

rel = zip(productid,title)
print rel
rel_df = pd.DataFrame(rel,columns=['productid','title'])
print rel_df