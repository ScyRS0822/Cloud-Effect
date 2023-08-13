# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
import joblib
import lightgbm as lgb


def drawLST(MYresult, FYresult):
    N = len(MYresult)
    #我的平均值，和风云的平均值
    MYmean = np.mean(MYresult)
    FYmean = np.mean(FYresult)
    #我的标准差，和风云的标准差
    MYsd = (np.sum((MYresult-MYmean)**2)/(N-1))**0.5
    FYsd = (np.sum((FYresult-FYmean)**2)/(N-1))**0.5
    #相关系数R2
    R2 = (np.sum((MYresult-MYmean)*(FYresult-FYmean))/(N-1)/MYsd/FYsd)**2
    
    #偏差bias
    bias = np.sum(MYresult-FYresult)/N
    #均方根误差RMSE
    RMSE = (np.sum((MYresult-FYresult)**2)/N)**0.5
    plt.figure(figsize=(4,4))
    plt.hist2d(FYresult, MYresult, cmin = 0.5,bins=500,norm=LogNorm(),cmap = 'Spectral_r', range=[(180, 350), (180, 350)]) #,norm=LogNorm(),
    plt.colorbar(shrink=0.824)
    x = np.linspace(200,350,4) #list(range(0,11,2))
    y = np.linspace(200,350,4)
    #在图中添加一条Y=X的线，还可以起到控制x、y坐标轴的显示范围的作用！
    X = [180,350]
    Y = [180,350]
    plt.plot(X,Y,c='black',linewidth = 0.9, linestyle='--')
    plt.xticks(x,['200','250','300','350'])
    plt.yticks(y,['200','250','300','350'])
    #在途中添加注释文本信息，前两个数代表文本在图中的像元空间xy坐标
    plt.text(190, 335, r'$N=%d$' % N, size=12)
    plt.text(190, 323, r'$R²=%.2f$' % R2, size=12)
    plt.text(190, 311, r'$bias=%.2f$' % bias, size=12)
    #plt.text(0, 0.5, r'$MRE=%.3f$' % MRE, size=12)
    plt.text(190, 299, r'$RMSE=%.2f$' % RMSE, size=12)
    plt.text(250, 210, '(b) Hourly, Sea', size=12)
    plt.ylabel('Estimated SST (K)', size=12)
    plt.xlabel('ERA5 SST (K)', size=12)
    plt.text(361, 353, 'n', size=12)
    plt.axis('scaled')
    plt.savefig('SeaHourly.jpg', dpi=500, bbox_inches = 'tight')
    plt.show()

data = pd.read_csv('data/TestData/Sea2019.txt', sep='\t', header=None, skiprows=1)
data = np.array(data)
data[:,4] = 90-np.abs(data[:,4])
data[:,0] = np.where(data[:,0]<0.001,0,data[:,0])

Xdata = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]))
Ydata = data[:,6].reshape(-1,1).ravel()

# Random Forest
RF = joblib.load('RFsea.pkl')
RFpred = RF.predict(Xdata)
# drawLST(RFpred,Ydata,'RF')

# LightGBM
LGBM = lgb.Booster(model_file='LGBMsea.txt')
LGBMpred = LGBM.predict(Xdata)
# drawLST(LGBMpred,Ydata,'LGBM')

# CatBoost
CB = joblib.load('CBsea.joblib')
CBpred = CB.predict(Xdata)
# drawLST(CBpred,Ydata,'CB')

drawLST((RFpred+LGBMpred+CBpred)/3,Ydata)




