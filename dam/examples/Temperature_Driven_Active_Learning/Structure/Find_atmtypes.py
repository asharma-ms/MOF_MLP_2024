import os
import numpy as np
#from matplotlib import pyplot as plt
#import time

import sys
sys.path.append("../../../../")
import dam as Dam


fposcar='IRMOF-1.vasp'      #Input file name
fxyz = 'IRMOF1.xyz'               #xyz output file name
fxyz_align = 'IRMOF1_align.xyz'   #xyz output file name with aligned atoms
fxyz_sorted_at = 'IRMOF1_sorted_at.xyz' #Sorted file in xyz format with atom types assigned
fxyz_sorted = 'IRMOF1_sorted.xyz' #Sorted file in xyz format without atom types assigned, just elements

Dam.traj_tools.poscar2axyz(fposcar,fxyz)
Dam.traj_tools.align_cellnData(fxyz,fxyz_align,'C')
Dam.traj_tools.axyz2poscar(fxyz_align,'align.vasp') # Just to visualize the aligned cell

special_indexes=[]
ts=Dam.mlp_data.MLSetAnalysis(fxyz_align, special_indexes)
ts.get_atmtyp(fxyz_sorted_at)
del ts

#To remove atom types and just keep elements in a file in axyz format
Dam.traj_tools.remove_at_axyz(fxyz_sorted_at,fxyz_sorted)

###To add atomtypes in a trajectory file in axyz format 
#f = open('IRMOF1_sorted_at.xyz','r')
#x = f.readlines()
#f.close();
#atm_tp=[]
#for i in x[2:]:
#    atm_tp.append(i.split()[0])
#Dam.traj_tools.add_at_axyz(fin_axyz=fin_axyz_woat, fout_axyz=fout_axyz_wat, atm_tp=atm_tp)

###To remove atomtypes from a trajectory file in axyz format
#Dam.traj_tools.remove_at_axyz(fin_axyz_wat,fout_axyz_woat)

