import os
import numpy as np
import random

import sys
sys.path.append("../../../../") # Change this according to location of Dam package
import dam as Dam

#To read MD configurations from SNAP of previous TS and select specific configurations to make new TS
train_id=1;
n_mdrun = 50; #from past train_id
element_list=['C','H','O','Zn']
faxyz = 'Itrain.xyz'

#------------------------------------------
dd=np.load('ts.npz',allow_pickle=True)

if(train_id>0):
    os.system('rm -rf TS'+str(train_id))
    os.system('mkdir TS'+str(train_id))
    #os.system('cp TS'+str(train_id-1)+'/Itrain.xyz TS'+str(train_id+1)+'/')
    os.system('cp TS'+str(train_id-1)+'/nf.npz TS'+str(train_id)+'/')

for i in range(n_mdrun):
    ft = 'TS'+str(train_id-1)+'/mdrun/run'+str(i)+'/'
    #Converting lammpstrj file of mdrun in previous training set to axyz file
    #print(ft)
    Dam.traj_tools.lammptraj2axyz(flmp=ft+'dump.lammpstrj', faxyz=ft+'dump.xyz', fcst=0, lcst=int(dd['ts'][train_id-1][2]/dd['ts'][train_id-1][3]), element_list=element_list)

    #Selecting training configurations from mdrun of previous training set
    ts=Dam.mlp_data.MLSetAnalysis(ft+'dump.xyz', special_indexes=[])
    ts.append_TS_analysis_sym(fnpz='TS'+str(train_id)+'/nf.npz', ftrain_append='TS'+str(train_id)+'/'+faxyz, nconfig_append=0)#0 means 1

#Creating DFT input files
#faxyz = 'Itrain.xyz'
path = 'TS'+str(train_id)+'/DFT_Train/'
Dam.mlp_data.make_path(path)
Dam.mlp_data.axyz2cp2kinp(faxyz = 'TS'+str(train_id)+'/'+faxyz, run_dir=path, cp2k_inp_dir='../cp2k_inp/', rm_at=0)

os.system('cp TS'+str(train_id-1)+'/make_TS_axyz.py TS'+str(train_id)+'/')
os.system('mkdir TS'+str(train_id)+'/snapfit')
os.system('cp TS'+str(train_id-1)+'/snapfit/snapfit.py TS'+str(train_id)+'/snapfit/')
