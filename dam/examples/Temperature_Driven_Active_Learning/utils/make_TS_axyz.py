import os
import numpy as np

import sys
sys.path.append("../../../../../") # Change this according to location of Dam package
import dam as Dam

#train_id = 2;
Na=424

Dam.mlp_data.make_path('MLPData/')

t=Dam.mlp_data.MLPDataSet('DFT_Train/')
t.fetch_icp2k(Na,0,'MLPData/TSi.xyz')
#t.fetch_MLPdata(Na,1,'Train_wovdw.xyz')
del t

train_id = int(os.getcwd().split('/')[-1][2:])
if(train_id>0):
    os.system('cat ../TS'+str(train_id-1)+'/MLPData/TS.xyz MLPData/TSi.xyz > MLPData/TS.xyz')
else:
    os.system('mv MLPData/TSi.xyz  MLPData/TS.xyz')
