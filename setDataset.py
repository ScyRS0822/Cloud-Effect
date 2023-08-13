# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
from netCDF4 import Dataset

year = '2017'

SoilType = Dataset(r"F:\PythonProject\RFforLST\RFforLST\SoilType.nc")
SoilType = np.array(SoilType['slt'])[0]
mask = np.where(SoilType<0.5,np.nan,1)

LSTs = []
SWNRs = []
LWDRs = []
TCCs = []
CWVs = []
LATs = []
BBEs = []
DOYs = []
LAIs = []

files = os.listdir('H:/ERA5/'+year)
BBEfiles = [f for f in os.listdir('H:/GLASS/BBE_avhrr_005d_v41_'+year+'/'+year) if f.endswith('.hdf')]
LAIfiles = [f for f in os.listdir('H:/GLASS/Lai-avhrr-v40-'+year+'/'+year) if f.endswith('.hdf')]

LAT = np.linspace(90,-90, 721)
LAT = LAT.reshape(-1, 1)
LAT = np.repeat(LAT, 1440, axis=1)

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
    
    data = Dataset('H:/ERA5/'+year+'/'+file)
    LST = np.array(data['skt'])
    SWNR = np.array(data['msnswrf'])
    LWDR = np.array(data['msdwlwrf'])
    TCC = np.array(data['tcc'])
    CWV = np.array(data['tcwv'])
    
    DOY = np.ones((721,1440))*(files.index(file))
    DOY[361:,:] = -DOY[361:,:]
    
    for hour in range(24):
        LST0 = LST[hour]
        TCC0 = TCC[hour]
        CWV0 = CWV[hour]
        if hour == 0:
            temp = Dataset('H:/ERA5/'+year+'/'+files[files.index(file)-1])
            SWNR0 = np.array(temp['msnswrf'])[-1]
            LWDR0 = np.array(temp['msdwlwrf'])[-1]
        else:
            SWNR0 = SWNR[hour-1]
            LWDR0 = LWDR[hour-1]

        LSTmask = LST0*mask
        SWNRmask = SWNR0*mask
        LWDRmask = LWDR0*BBE
        LWDRmask = LWDRmask*mask
        TCCmask = TCC0*mask
        CWVmask = CWV0*mask
        LATmask = LAT*mask
        BBEmask = BBE*mask
        DOYmask = DOY*mask
        LAImask = LAI*mask

        LSTmask = LSTmask.flatten()
        SWNRmask = SWNRmask.flatten()
        LWDRmask = LWDRmask.flatten()
        TCCmask = TCCmask.flatten()
        CWVmask = CWVmask.flatten()
        LATmask = LATmask.flatten()
        BBEmask = BBEmask.flatten()
        DOYmask = DOYmask.flatten()
        LAImask = LAImask.flatten()
        
        LSTmask = LSTmask[~np.isnan(LSTmask)]
        SWNRmask = SWNRmask[~np.isnan(SWNRmask)]
        LWDRmask = LWDRmask[~np.isnan(LWDRmask)]
        TCCmask = TCCmask[~np.isnan(TCCmask)]
        CWVmask = CWVmask[~np.isnan(CWVmask)]
        LATmask = LATmask[~np.isnan(LATmask)]
        BBEmask = BBEmask[~np.isnan(BBEmask)]
        DOYmask = DOYmask[~np.isnan(DOYmask)]
        LAImask = LAImask[~np.isnan(LAImask)]
        
        step = 1500  # 1500,4400
        random1 = random.randint(0,step)
        
        LSTmask = LSTmask[random1::step+1]
        SWNRmask = SWNRmask[random1::step+1]
        LWDRmask = LWDRmask[random1::step+1]
        TCCmask = TCCmask[random1::step+1]
        CWVmask = CWVmask[random1::step+1]
        LATmask = LATmask[random1::step+1]
        BBEmask = BBEmask[random1::step+1]
        DOYmask = DOYmask[random1::step+1]
        LAImask = LAImask[random1::step+1]
        
        LSTs = np.append(LSTs,LSTmask)
        SWNRs = np.append(SWNRs,SWNRmask)
        LWDRs = np.append(LWDRs,LWDRmask)
        TCCs = np.append(TCCs,TCCmask)
        CWVs = np.append(CWVs,CWVmask)
        LATs = np.append(LATs,LATmask)
        BBEs = np.append(BBEs,BBEmask)
        DOYs = np.append(DOYs,DOYmask)
        LAIs = np.append(LAIs,LAImask)
        
SWNRs = np.where(SWNRs<0,0,SWNRs)

# 保存训练数据集
temp = np.vstack((SWNRs,LWDRs,TCCs,CWVs,LATs,DOYs,BBEs,LAIs,LSTs)).T  #SWNRs,LWDRs,TCCs,CWVs,LATs,DOYs,BBEs,LAIs,LSTs
np.savetxt('F:/PythonProject/3PolesLST/Land'+year+'.txt', temp, fmt='%14.6E', delimiter='\t', 
           header='SWNR\tLWDR\tTCC\tCWV\tLat\tDOY\tBBE\tLAI\tLST', comments='')
   

