import os
import numpy as np
import random

import sys
sys.path.append("../../../../") # Change this according to location of Dam package
import dam as Dam

#To make input files for MD simulations

ts=0;#starts from zero, *change it according to step
nparallel = 50; # Number of MD simulations to run
#--------------------------------------------------
dd=np.load('ts.npz',allow_pickle=True)
#id,dt,nstep,Temp,Press
dt  = dd['ts'][ts][1]; # Timestep in ps
nmd = dd['ts'][ts][2]; # Number of MD steps to perorm
npr = dd['ts'][ts][3]; # Print trajectory after this number of steps
T   = dd['ts'][ts][4]; # Temperature in kelvin
P   = dd['ts'][ts][5]; # Pressure in bar

mdf = '../mdrun_files'  # Folder containing MD files

#Creating new randomly displaced configurations
md_ini = 'md_ini.xyz'
Dam.mlp_data.random_displaced_configurations(fin='TS0/Itrain.xyz', fout = md_ini, dr = 0.1, Nconfigs=nparallel, config_type='npt')
#------------------------------------------------------------

train_id = ts;

snf = 'TS'+str(train_id)+'/snapfit'  # Folder contining snapcoeff and snappara files

p = 'TS'+str(train_id)+'/mdrun'
os.system('rm -rf '+p)
Dam.mlp_data.make_path(p)

for i in range(nparallel):
    p1 = p+'/run'+str(i)+'/'
    Dam.mlp_data.make_path(p1)
    os.system('cp '+mdf+'/runmd.py '+p1)
    os.system('cp '+mdf+'/in_md.snap '+p1)

    os.system("sed -i 's/300/"+str(T)+"/g' "+p1+"in_md.snap")
    x = int(random.uniform(0,50000))
    os.system("sed -i 's/624/"+str(x)+"/g' "+p1+"in_md.snap")
    os.system("sed -i 's/0.0005/"+str(dt)+"/g' "+p1+"in_md.snap")
    os.system("sed -i 's/run 500/run "+str(int(nmd))+"/g' "+p1+"runmd.py")
    os.system("sed -i 's/Nxxx/ "+str(int(npr))+"/g' "+p1+"in_md.snap")
    os.system('cp '+snf+'/snapparam '+p1)
    os.system('cp '+snf+'/snapcoeff '+p1)

    #Dam.traj_tools.axyz2lmpdata(faxyz = 'Itrain.xyz', frno=i, fdata=p1+'data.snap', ele=['C','H','O','Zn'], masses=[12.011, 1.008, 15.999, 65.409], tp='el')
    Dam.traj_tools.axyz2lmpdata(faxyz = md_ini, frno=i, fdata=p1+'data.snap', ele=['C','H','O','Zn'], masses=[12.011, 1.008, 15.999, 65.409], tp='at')


f = open('run_md.sh','w')
f.write('bf=TS'+str(ts)+'\n')
f.write('for i in `seq 0 '+str(nparallel-1)+'` \n')
f.write('do\n cd $bf/mdrun/run$i/ \n nohup python runmd.py & \n cd ../../../ \n sleep 100s\ndone\n')
f.close()


