# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from matplotlib.colors import LogNorm
import joblib
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
import catboost as cb

def drawLST(MYresult, FYresult,title):
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
    # 设置标题、x标签、y标签
    if len(title)<5:
        plt.title(title,size=12)
    else:
        plt.title('(RF+CB+LGBM)/3',size=12)
    plt.ylabel('Estimated LST (K)', size=12)
    plt.xlabel('ERA5 LST (K)', size=12)
    cbar = plt.colorbar(shrink=0.8)
    plt.text(361, 353, 'n', size=12)
    plt.axis('scaled')
    plt.savefig(title+'.jpg', dpi=1000, bbox_inches = 'tight')
    plt.show()
    
data = pd.read_csv('data/TrainData/Sea2018.txt', sep='\t', header=None, skiprows=1)
data = np.array(data)

data[:,0] = np.where(data[:,0]<1,0,data[:,0])
data[:,4] = 90-np.abs(data[:,4])


Xdata = np.column_stack((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5]))
Ydata = data[:,6].reshape(-1,1).ravel()


# Random Forest, 2016
RF = RandomForestRegressor(n_estimators=40,random_state=2023,min_samples_split=30,n_jobs=4)
RF.fit(Xdata, Ydata)
RFpred = RF.predict(Xdata)
drawLST(RFpred,Ydata,'RF')
print(RF.feature_importances_)
joblib.dump(RF, 'RFsea.pkl')


# LightGBM, 2017
params = {'objective': 
          'regression', 
          'num_leaves': 250, 
          'n_estimators': 45, 
          'min_child_samples': 30, 
          'metric': 'mse', 
          'max_depth': -1, 
          'learning_rate': 0.45, 
          'feature_fraction': 0.9, 
          'boosting_type': 'dart'}
LGBM = lgb.LGBMRegressor(**params)
LGBM.fit(Xdata, Ydata)
LGBMpred = LGBM.predict(Xdata)
drawLST(LGBMpred,Ydata,'LGBM')
print(LGBM.feature_importances_)
LGBM.booster_.save_model('LGBMsea.txt')


# CatBoost, 2018
CB = cb.CatBoostRegressor(eval_metric='RMSE',verbose=20,learning_rate=0.10,depth=12,iterations=250)
CB.fit(Xdata,Ydata)
CBpred = CB.predict(Xdata)
drawLST(CBpred,Ydata,'CB')
print(CB.get_feature_importance())
joblib.dump(CB, 'CBsea.joblib')






