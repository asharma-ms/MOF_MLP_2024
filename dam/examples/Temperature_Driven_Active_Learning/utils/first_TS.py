import os
import numpy as np
import random

import sys
sys.path.append("../../../../") # Change this according to location of Dam package
import dam as Dam

n_config = 200;  #Number of random configurations to generate 
element_list=['C','H','O','Zn'] # Element list of MOF, order must match with input .xyz file
faxyz = 'Itrain.xyz'

#------------------------------------------
os.system('rm -rf TS0')
os.system('mkdir TS0')

#Creating new randomly displaced configurations
Dam.mlp_data.random_displaced_configurations_eq(fin='../Structure/IRMOF1_sorted.xyz', fout = 'TS0/Itrain_rand.xyz', dr = 0.12, Nconfigs=n_config, config_type='npt')

os.system('mv cell.tcl TS0/')

db=0.05     # Angstrom
da=5.0      # Degrees
dda=10.0     # Degrees
dscell=0.05 # Angstrom
dacell=0.07   # 0.035 Radian ~ 2 Degree


ts=Dam.mlp_data.MLSetAnalysis('TS0/Itrain_rand.xyz', special_indexes=[])
ts.analysis_sym_small_axyz(db,da,dda,dscell,dacell, fnpz='TS0/nf.npz', cell_type='cubic')
del ts

os.system('mv Train_AL.xyz TS0/'+faxyz)
os.system('mv Test_AL.xyz TS0/')

#Creating DFT input files
path = 'TS0/DFT_Train/'
Dam.mlp_data.make_path(path)
Dam.mlp_data.axyz2cp2kinp(faxyz = 'TS0/'+faxyz, run_dir=path, cp2k_inp_dir='../cp2k_inp/', rm_at=0)

os.system('cp ../utils/make_TS_axyz.py '+'TS0/')
os.system('mkdir TS0/snapfit')
os.system('cp ../utils/snapfit.py TS0/snapfit/')

