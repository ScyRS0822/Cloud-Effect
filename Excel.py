# -*- coding: utf-8 -*-

import numpy as np
import cv2
from osgeo import gdal,osr
import pandas as pd

def getArea(lat):
    lat = np.abs(np.radians(lat))
    Lon = 0.25*2*np.pi*6371.4/360
    Lat = 0.25*2*np.pi*6371.4*np.cos(lat)/360
    area = Lat*Lon
    return area

# 记录CRE在Globe/Arctic/Antarctic/Tibet地区的时间变化趋势
data = {
    'Year\Month': ['1981','1991','2001','2011','2021'],
    '01': [0, -10, -20, -5, 0],
    '02': [5, -5, -15, -2, 0],
    '03': [10, 0, -10, 3, 0],
    '04': [15, 5, -5, 8, 0],
    '05': [20, 10, 0, 15, 0],
    '06': [25, 15, 5, 20, 0],
    '07': [30, 20, 10, 25, 0],
    '08': [25, 15, 5, 20, 0],
    '09': [20, 10, 0, 15, 0],
    '10': [15, 5, -5, 8, 0],
    '11': [10, 0, -10, 3, 0],
    '12': [5, -5, -15, -2, 0]
}

mask = cv2.imread(r"F:\PythonProject\3PolesLST\data\Mask\Tibet.tif",-1)
Area = cv2.imread('F:/PythonProject/3PolesLST/data/Mask/Area.tif',-1)

for year in ['1981','1991','2001','2011','2021']:
    for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        
        img = cv2.imread('F:/PythonProject/3PolesLST/result/'+year+'/CRE'+month+'.tif',-1)
        
        Globe = np.sum(img*Area/np.sum(Area))
        Arctic = np.sum(img[:95,:]*Area[:95,:]/np.sum(Area[:95,:]))
        Antarctic = np.sum(img[626:,:]*Area[626:,:]/np.sum(Area[626:,:]))
        Tibet = np.nansum(img*Area*mask/np.nansum(Area*mask))
        
        data[month][data['Year\Month'].index(year)] = np.round(Tibet,2)

df = pd.DataFrame(data)
df.to_excel('F:/PythonProject/3PolesLST/result/Tibet.xlsx', index=False)






