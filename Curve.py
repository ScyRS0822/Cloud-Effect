# -*- coding: utf-8 -*-
import os
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cv2
import joblib
import lightgbm as lgb
from netCDF4 import Dataset
from sklearn.impute import SimpleImputer

def getRMSE(LSTall,LST):
    squared_diff = (LSTall - LST) ** 2
    rmse = np.sqrt(np.mean(squared_diff))
    print(rmse)

# 备选(31,110),(32,110),(30,112),(25.3,114),(34,114.5),(35.3,116.3)

lat0 = 35.3
lon0 = 116.3
lat = int((90-lat0)/0.25)
lon = int(lon0/0.25)

data = Dataset(r"F:\PythonProject\3PolesLST\data\Curve20200701.nc")

LST = np.array(data['skt'])[16:16+72+1,lat,lon]
SWNRall = np.array(data['msnswrf'])[15:15+72+1,lat,lon]
SWNRclr = np.array(data['msnswrfcs'])[15:15+72+1,lat,lon]
LWDRall = np.array(data['msdwlwrf'])[15:15+72+1,lat,lon]
TCC = np.array(data['tcc'])[16:16+72+1,lat,lon]
CWV = np.array(data['tcwv'])[16:16+72+1,lat,lon]
Lat = (90 - np.abs(lat0))*np.ones(73)
DOY = 183*np.ones(73)

hour = np.linspace(0,72,73)

BBE = Dataset(r"H:\GLASS\BBE_avhrr_005d_v41_2020\2020\GLASS03B02.V40.A2020185.2022140.hdf")
BBE = np.array(BBE['BBE'])
BBE = np.where(BBE>1.01,0.985,BBE)
BBE = np.where(BBE<0.85,0.985,BBE)
BBE = np.hstack((BBE[:,3600:],BBE[:,:3600]))
BBE = cv2.resize(BBE,(1440,721),interpolation=cv2.INTER_AREA)
BBE = BBE[lat,lon]*np.ones(73)

LAI = Dataset(r"H:\GLASS\Lai-avhrr-v40-2020\2020\GLASS01B01.V60.A2020185.2022138.hdf")
LAI = np.array(LAI['LAI'])
LAI = np.where(LAI>100,0,LAI)
LAI = np.hstack((LAI[:,3600:],LAI[:,:7200]))
LAI = cv2.resize(LAI,(1440,721),interpolation=cv2.INTER_AREA)
LAI = LAI[lat,lon]*np.ones(73)

RF = joblib.load('RFland.pkl')
LGBM = lgb.Booster(model_file='LGBMland.txt')
CB = joblib.load('CBland.joblib')


LWDRclr = np.array(data['msdwlwrfcs'])[16:16+72+1,lat,lon]
LWURclr = LWDRclr - np.array(data['msnlwrfcs'])[16:16+72+1,lat,lon]
LSTclr = ((LWURclr-(1-BBE)*LWDRclr)/BBE/5.67e-8)**0.25

LSTdata = {'ERA5 LST':LST,'LST$_{clr}$':LSTclr,'TCC':TCC}

# 画两条LST线外加一条TCC线
def drawRF(hour, ydata):
    """画双轴折线图
    :param lx x轴数据集合
    :param dy y轴数据字典
    """
    # 设置图片可以显示中文和特殊字符
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    yname1 = list(ydata.keys())[0]
    yname2 = list(ydata.keys())[1]
    yname3 = list(ydata.keys())[2]
    
    plt.figure(figsize=(15, 5), dpi=500)
    
    plt.plot(hour, ydata.get(yname1), label=yname1, linewidth=3, color='red',linestyle='-')
    plt.plot(hour, ydata.get(yname2), label=yname2, linewidth=3, color='darkgrey',linestyle='-')
    
    time = np.linspace(0,72,13)
    # plt.yticks([290,295,300,305,310],['290','295','300','305','310'],size=14)
    plt.xticks(time,['0','6','12','18','24(0)','6','12','18','24(0)','6','12','18','24'],size=15)
    plt.xlabel('Local Time (h)',fontsize=20)
    
    plt.ylabel('LST (K)', fontsize=20)
    plt.yticks(size=15,ticks=[290,295,300,305,310])

    plt.legend(loc=[0.15,0.07], fontsize=15)
    
    # 调用twinx后可绘制次坐标轴
    plt.twinx()
    plt.plot(hour, ydata.get(yname3), label=yname3, linewidth=3, color='blue',linestyle=":")
    plt.ylabel(yname3, fontsize=20)
    plt.yticks(size=15)
    
    plt.legend(loc='upper right', fontsize=15)
    
    plt.savefig('Daily1.jpg', dpi=500,bbox_inches='tight')
    plt.show()

drawRF(hour,LSTdata)


# 用机器学习模型画LST曲线
LWDRclr = np.array(data['msdwlwrfcs'])[15:15+72+1,lat,lon]
LWURclr = LWDRclr - np.array(data['msnlwrfcs'])[15:15+72+1,lat,lon]

Xall = np.column_stack((SWNRall, LWDRall, TCC, CWV, Lat, DOY, BBE, LAI))
Xall = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xall)
RFall = RF.predict(Xall)
LGBMall = LGBM.predict(Xall)
CBall = CB.predict(Xall)

Xclr = np.column_stack((SWNRclr, LWDRclr, np.zeros(73), CWV, Lat, DOY, BBE, LAI))
Xclr = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xclr)
RFclr = RF.predict(Xclr)
LGBMclr = LGBM.predict(Xclr)
CBclr = CB.predict(Xclr)

LSTall = (RFall + CBall + LGBMall)/3
LSTclr = (RFclr + CBclr + LGBMclr)/3

LSTdata = {'ERA5 LST':LST,'LST$_{all}$':LSTall,'LST$_{clr}$':LSTclr,'TCC':TCC}

# 画三条LST线外加一条TCC线
def drawRF(hour, ydata):
    """画双轴折线图
    :param lx x轴数据集合
    :param dy y轴数据字典
    """
    # 设置图片可以显示中文和特殊字符
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    yname1 = list(ydata.keys())[0]
    yname2 = list(ydata.keys())[1]
    yname3 = list(ydata.keys())[2]
    yname4 = list(ydata.keys())[3]
    
    plt.figure(figsize=(15, 5), dpi=500)
    
    plt.plot(hour, ydata.get(yname1), label=yname1, linewidth=2.5, color='red',linestyle='-')
    plt.plot(hour, ydata.get(yname2), label=yname2, linewidth=2.5, color='black',linestyle='-')
    plt.plot(hour, ydata.get(yname3), label=yname3, linewidth=2.5, color='darkgrey',linestyle='-')
    plt.ylim(290, 310)
    
    time = np.linspace(0,72,13)
    plt.xticks(time,['0','6','12','18','24(0)','6','12','18','24(0)','6','12','18','24'],size=15)
    plt.xlabel('Local Time (h)',fontsize=20)
    
    plt.ylabel('LST (K)', fontsize=20)
    plt.yticks(size=15,ticks=[290,295,300,305,310])

    plt.legend(loc=[0.15,0.07], fontsize=15)
    
    # 调用twinx后可绘制次坐标轴
    plt.twinx()
    plt.plot(hour, ydata.get(yname4), label=yname4, linewidth=2.5, color='blue',linestyle=":")
    plt.ylabel(yname4, fontsize=20)
    plt.yticks(size=15)
    
    plt.legend(loc='upper right', fontsize=15)
    
    plt.savefig('Daily2.jpg', dpi=500,bbox_inches='tight')
    plt.show()
    
drawRF(hour, LSTdata)
getRMSE(LSTall, LST)


