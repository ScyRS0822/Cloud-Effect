# -*- coding: utf-8 -*-

import os
import cv2
import joblib
import numpy as np
import lightgbm as lgb
from netCDF4 import Dataset
from sklearn.impute import SimpleImputer
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from osgeo import gdal,osr

def ARRAY2TIF(array,outfile,res):
    Lonmin,Latmax,Lonmax,Latmin = [-180,90,180,-90]
    Num_lat = 721  # int(180/res)
    Num_lon = 1440 # int(360/res)
    Lat_res = (Latmax - Latmin) / (float(Num_lat))
    Lon_res = (Lonmax - Lonmin) / (float(Num_lon))
    # 设置影像的显示范围。Lat_re前需要添加负号
    geotransform = (Lonmin, Lon_res, 0.0, Latmax, 0.0, -Lat_res)
    
    driver = gdal.GetDriverByName('GTiff')
    out_tif = driver.Create(outfile, Num_lon, Num_lat, 1, gdal.GDT_Float32)
    out_tif.SetGeoTransform(geotransform)
    # 定义投影
    prj = osr.SpatialReference()
    prj.ImportFromEPSG(4326)
    out_tif.SetProjection(prj.ExportToWkt())
    # 数据导出
    out_tif.GetRasterBand(1).WriteArray(array)  # 将数据写入内存
    out_tif.FlushCache()  # 将数据写入到硬盘
    out_tif = None  # 关闭tif文件

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
    # 设置标题、x标签、y标签
    if len(title)<5:
        plt.title(title,size=12)
    else:
        plt.title('(RF+CB+LGBM)/3',size=12)
    plt.ylabel('Estimated Ts (K)', size=12)
    plt.xlabel('ERA5 Ts (K)', size=12)
    plt.text(361, 353, 'n', size=12)
    plt.axis('scaled')
    if len(title)<5:
        plt.savefig(title+'.jpg', dpi=1000, bbox_inches = 'tight')
    else:
        plt.savefig('RF-CB-LGBM.jpg', dpi=1000, bbox_inches = 'tight')
    plt.show()

year = '1981'

SoilType = Dataset(r"F:\PythonProject\RFforLST\RFforLST\SoilType.nc")
SoilType = np.array(SoilType['slt'])[0]
mask = np.where(SoilType<0.5,np.nan,1)

files = os.listdir('H:/ERA5/'+year)
BBEfiles = [f for f in os.listdir('H:/GLASS/BBE_avhrr_005d_v41_'+year+'/'+year) if f.endswith('.hdf')]
LAIfiles = [f for f in os.listdir('H:/GLASS/Lai-avhrr-v40-'+year+'/'+year) if f.endswith('.hdf')]

LAT = np.linspace(90,-90, 721)
LAT = LAT.reshape(-1, 1)
LAT = np.repeat(LAT, 1440, axis=1)
LAT = 90-np.abs(LAT)

RFland = joblib.load('RFland.pkl')
LGBMland = lgb.Booster(model_file='LGBMland.txt')
CBland = joblib.load('CBland.joblib')
print(CBland.get_feature_importance())

RFsea = joblib.load('RFsea.pkl')
LGBMsea = lgb.Booster(model_file='LGBMsea.txt')
CBsea = joblib.load('CBsea.joblib')

LSTyearly = 0

RFyearlyAll = 0
LGBMyearlyAll = 0
CByearlyAll = 0

RFyearlyClr = 0
LGBMyearlyClr = 0
CByearlyClr = 0

for file in files[1:]:
    print(files.index(file))
    
    BBE = BBEfiles[int(len(BBEfiles)*(files.index(file)-1)/len(files))]
    BBE = Dataset('H:/GLASS/BBE_avhrr_005d_v41_'+year+'/'+year+'/'+BBE)
    BBE = np.array(BBE['BBE'])
    BBE = np.where(BBE<0.85,0.985,BBE)
    BBE = np.where(BBE>1.01,0.985,BBE)
    BBE = np.hstack((BBE[:,3600:],BBE[:,:3600]))
    BBE = cv2.resize(BBE,(1440,721),interpolation=cv2.INTER_AREA)
    
    LAI = LAIfiles[int(len(LAIfiles)*(files.index(file)-1)/len(files))]
    LAI = Dataset('H:/GLASS/Lai-avhrr-v40-'+year+'/'+year+'/'+LAI)
    LAI = np.array(LAI['LAI'])
    LAI = np.where(LAI>100,0,LAI)
    LAI = np.hstack((LAI[:,3600:],LAI[:,:7200]))
    LAI = cv2.resize(LAI,(1440,721),interpolation=cv2.INTER_AREA)
    
    TCCC0 = np.zeros((721,1440))
    
    data = Dataset('H:/ERA5/'+year+'/'+file)
    LST = np.array(data['skt'])
    SWNR = np.array(data['msnswrf'])
    LWDR = np.array(data['msdwlwrf'])
    TCC = np.array(data['tcc'])
    CWV = np.array(data['tcwv'])
    SWNRC = np.array(data['msnswrfcs'])
    LWDRC = np.array(data['msdwlwrfcs'])
    
    DOY = np.ones((721,1440))*(files.index(file))
    DOY[361:,:] = -DOY[361:,:]
    
    for hour in range(24):
        LST0 = LST[hour]
        LSTyearly = LSTyearly + LST0
        TCC0 = TCC[hour]
        CWV0 = CWV[hour]
        
        if hour == 0:
            temp = Dataset('H:/ERA5/'+year+'/'+files[files.index(file)-1])
            SWNR0 = np.array(temp['msnswrf'])[-1]
            LWDR0 = np.array(temp['msdwlwrf'])[-1]
            SWNRC0 = np.array(temp['msnswrfcs'])[-1]
            LWDRC0 = np.array(temp['msdwlwrfcs'])[-1]
        else:
            SWNR0 = SWNR[hour-1]
            LWDR0 = LWDR[hour-1]
            SWNRC0 = SWNRC[hour-1]
            LWDRC0 = LWDRC[hour-1]
        SWNR0 = np.where(SWNR0<0.001,0,SWNR0)
        SWNRC0 = np.where(SWNRC0<0.001,0,SWNRC0)
        row,col = SWNR0.shape
        
        for cover in ['Land','Sea']:
            if cover == 'Land':
                mask = np.where(SoilType<0.5,np.nan,1)
            else:
                mask = np.where(SoilType<0.5,1,np.nan)

            SWNRmask = SWNR0*mask
            LWDRmask = LWDR0*BBE
            LWDRmask = LWDRmask*mask
            TCCmask = TCC0*mask
            CWVmask = CWV0*mask
            LATmask = LAT*mask
            BBEmask = BBE*mask
            DOYmask = DOY*mask
            SWNRCmask = SWNRC0*mask
            LWDRCmask = LWDRC0*BBE
            LWDRCmask = LWDRCmask*mask
            TCCCmask = TCCC0*mask
            LAImask = LAI*mask

            SWNRmask = SWNRmask.flatten()
            LWDRmask = LWDRmask.flatten()
            TCCmask = TCCmask.flatten()
            CWVmask = CWVmask.flatten()
            LATmask = LATmask.flatten()
            BBEmask = BBEmask.flatten()
            DOYmask = DOYmask.flatten()
            SWNRCmask = SWNRCmask.flatten()
            LWDRCmask = LWDRCmask.flatten()
            TCCCmask = TCCCmask.flatten()
            LAImask = LAImask.flatten()
            
            if cover == 'Land':
                
                Xall = np.column_stack((SWNRmask,LWDRmask,TCCmask,CWVmask,LATmask, DOYmask,BBEmask, LAImask))
                Xall = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xall)
                RFall = RFland.predict(Xall)
                LGBMall = LGBMland.predict(Xall)
                CBall = CBland.predict(Xall)
                
                Xclr = np.column_stack((SWNRCmask,LWDRCmask,TCCCmask,CWVmask,LATmask, DOYmask,BBEmask, LAImask))
                Xclr = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xclr)
                RFclr = RFland.predict(Xclr)
                LGBMclr = LGBMland.predict(Xclr)
                CBclr = CBland.predict(Xclr)
                
            else:
                
                Xall = np.column_stack((SWNRmask,LWDRmask,TCCmask,CWVmask,LATmask,DOYmask))
                Xall = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xall)
                RFall = RFsea.predict(Xall)
                LGBMall = LGBMsea.predict(Xall)
                CBall = CBsea.predict(Xall)
                
                Xclr = np.column_stack((SWNRCmask,LWDRCmask,TCCCmask,CWVmask,LATmask,DOYmask))
                Xclr = SimpleImputer(missing_values=np.nan, strategy='constant',fill_value=-9999).fit_transform(Xclr)
                RFclr = RFsea.predict(Xclr)
                LGBMclr = LGBMsea.predict(Xclr)
                CBclr = CBsea.predict(Xclr)
            
            RFall = np.reshape(RFall,(row,col))
            LGBMall = np.reshape(LGBMall,(row,col))
            CBall = np.reshape(CBall,(row,col))
            RFclr = np.reshape(RFclr,(row,col))
            LGBMclr = np.reshape(LGBMclr,(row,col))
            CBclr = np.reshape(CBclr,(row,col))
            
            if cover == 'Land':
                RFnan = RFall[0,0]
                LGBMnan = LGBMall[0,0]
                CBnan = CBall[0,0]
                RFnanC = RFclr[0,0]
                LGBMnanC = LGBMclr[0,0]
                CBnanC = CBclr[0,0]
            else:
                RFnan = RFall[720,0]
                LGBMnan = LGBMall[720,0]
                CBnan = CBall[720,0]
                RFnanC = RFclr[720,0]
                LGBMnanC = LGBMclr[720,0]
                CBnanC = CBclr[720,0]
            
            RFall = np.where(RFall==RFnan,0,RFall)
            LGBMall = np.where(LGBMall==LGBMnan,0,LGBMall)
            CBall = np.where(CBall==CBnan,0,CBall)
            RFclr = np.where(RFclr==RFnanC,0,RFclr)
            LGBMclr = np.where(LGBMclr==LGBMnanC,0,LGBMclr)
            CBclr = np.where(CBclr==CBnanC,0,CBclr)
            
            RFyearlyAll = RFyearlyAll + RFall
            LGBMyearlyAll = LGBMyearlyAll + LGBMall
            CByearlyAll = CByearlyAll + CBall
            RFyearlyClr = RFyearlyClr + RFclr
            LGBMyearlyClr = LGBMyearlyClr + LGBMclr
            CByearlyClr = CByearlyClr + CBclr

# 时间平均
LSTyearly = LSTyearly/365/24
RFyearlyAll = RFyearlyAll/365/24
LGBMyearlyAll = LGBMyearlyAll/365/24
CByearlyAll = CByearlyAll/365/24
RFyearlyClr = RFyearlyClr/365/24
LGBMyearlyClr = LGBMyearlyClr/365/24
CByearlyClr = CByearlyClr/365/24

# 模型平均
MLyearlyAll = (RFyearlyAll + LGBMyearlyAll + CByearlyAll)/3
MLyearlyClr = (RFyearlyClr + LGBMyearlyClr + CByearlyClr)/3

# 保存下来
LSTyearly = np.hstack((LSTyearly[:,720:],LSTyearly[:,:720]))
MLyearlyAll = np.hstack((MLyearlyAll[:,720:],MLyearlyAll[:,:720]))
MLyearlyClr = np.hstack((MLyearlyClr[:,720:],MLyearlyClr[:,:720]))
CRE = MLyearlyAll - MLyearlyClr

ARRAY2TIF(LSTyearly,'LST2019.tif',0.25)
ARRAY2TIF(MLyearlyAll,'MLall2019.tif',0.25)
ARRAY2TIF(MLyearlyClr,'MLclr2019.tif',0.25)
ARRAY2TIF(CRE,'CRE2019.tif',0.25)

# 查看精度
drawLST(MLyearlyAll.flatten(),LSTyearly.flatten(),'RF-CB-LGBM')


            
            
            