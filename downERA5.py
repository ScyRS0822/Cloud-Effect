# -*- coding: utf-8 -*-

import os
import cdsapi
import calendar

c = cdsapi.Client()
year = '2019'

for month in ['01','02','03','04','05','06','07','08','09','10','11','12']:
    days = calendar.monthrange(int(year), int(month))[1]

    for day in range(1,days+1):
        file = 'H:/ERA5/'+year+'LWNR/'+year+month+str(day).zfill(2)+'.nc'
        if os.path.exists(file):
            continue
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
            'mean_surface_net_long_wave_radiation_flux', 
            'mean_surface_net_long_wave_radiation_flux_clear_sky', 
                ],
                'year': year,
                'month': month,
                'day': str(day).zfill(2),# str(day).zfill(2),
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'format': 'netcdf',
            },
            'H:/ERA5/'+year+'LWNR/'+year+month+str(day).zfill(2)+'.nc')
        
        


