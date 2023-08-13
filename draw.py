# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal,osr

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(data):
    features = ['SWNR', 'LWDR', 'TCC', 'CWV', 'Lat', 'DoY', 'BBE', 'LAI']  #, 'BBE', 'LAI'
    colors = ['yellowgreen', 'black', 'darkorange', 'deepskyblue', 'lightcoral', 'gray', 'darkseagreen', 'gold']  #, 'darkseagreen', 'gold'

    fig, ax = plt.subplots()
    ax.bar(features, data, width=0.6, color=colors)

    ax.set_ylabel('Feature importance (%)')
    ax.set_xlabel('Features')
    
    plt.savefig('Feature.jpg', dpi=500, bbox_inches = 'tight')
    plt.show()

RFland = 100*np.array([4.64823656e-02, 8.49902151e-02, 6.47699272e-03, 8.23251775e-01, 2.48105927e-02, 3.53028981e-03, 9.99878405e-03, 4.58985069e-04])
LGBMland = [998,1963,1102,1345,1921,2672,885,319]
LGBMland = 100*np.array(LGBMland/np.sum(LGBMland))
CBland = np.array([16.04414604, 37.69324715, 8.25951327, 15.37406098, 9.28738493, 5.15881855, 7.7061659, 0.47666318])

RFsea = 100*np.array([0.00314113, 0.04410228, 0.00909561, 0.12418594, 0.80299403, 0.01648101])
LGBMsea = [847, 1744, 1247, 1892, 2015, 3460]
LGBMsea = 100*np.array(LGBMsea/np.sum(LGBMsea))
CBsea = np.array([3.24588173, 28.02090654, 8.82750393, 10.35380113, 38.74150246, 10.81040421])

MLland = (RFland + LGBMland + CBland)/3
MLsea = (RFsea + LGBMsea + CBsea)/3

plot_feature_importance(MLland)