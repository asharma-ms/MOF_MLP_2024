### In this folder
### cp2k_inp folder contains input files for DFT caldulation using CP2K 
### mdrun_files folder contains files for MD simulation
### utils folder contains few python script useful for the training set construction process as detailed below
### Training_Set_examplerun folder contain an example run of following steps
###===============================================================================================================

### 0. Activate appropriate conda environment
conda activate Dam ### Activate conda environment for Dam package

#mkdir Initial_structure
cd Initial_structure
### 1. Download or create cif file of a MOF (IRMOF-1.cif) 
### 2. Convert cif file to POSCAR file (IRMOF-1.vasp)
### 3. Run following script to generage file to assign atom types to atoms and sort the order of atoms

#rm *.xyz align.vasp order_change.txt
python Find_atmtypes.py

### Above code will generate sevaral files: align.vasp (note change in order of atoms),  IRMOF1_align.xyz,  IRMOF1_sorted.xyz,  IRMOF1_sorted_at.xyz,  IRMOF1.xyz  order_change.txt (details of order changed)

cd ..

# 4. Make directory Training_Set, which is going to have all data about training set
mkdir Training_Set

#6. Copy following files which will be used further for creation of MLP training set
cp utils/first_TS.py Training_Set/
cp utils/training_step_data.py Training_Set/
cp utils/newMD_TS.py Training_Set/
cp utils/refresh_TS.py Training_Set/
#-----------------------------------------------------------------------------------------------
# Step0
#------------------------------------------------------------------------------------------------
# A. To make initial training set (TS0), creat 500 random configurations and select 
cd Training_Set/
python first_TS.py #Change n_config, element_list (according to MOF), and db, da, dda, dscell, dacell

cd TS0

# B. Above code has created CP2K input structures of all training set configurations in folder TS0/DFT_Train/. Now put required CP2K input files (at each TS0/DFT_Train/snap) and run DFT calculations for each configuration.

# C. Once all DFT simulations are finished, run following code to fetch energy, forces, and virial stress values of all configurations and collect all data in axyz format (which is main input to Dam package) using following
cp ../../utils/make_TS_axyz.py .    
python make_TS_axyz.py # Change number of atoms accordingly

# D. Fit initial version of SNAP (i.e., MLP0) over this training data
cd snapfit
cp ../../../utils/snapfit.py .
python snapfit.py 
cd ../../            # Going back to directory Training_Set

# E. Now run 50 MD simulations in parallel using trained SNAP (i.e., MLP0) 
python training_step_data.py # Change here to modify time-stepsize, number of MD steps, temperature, pressure 
python newMD_TS.py    #Make sure ts=0

# F. Run MD simulations (it is at 100 K, change training_step_data.py to modify it for Step0)
sh run_md.sh  #Change sleep time according to your number of simultaneous runs in your hardware

#Now MD simultions with MLP0 have been finished. Now proceed towards Step 1

#-----------------------------------------------------------------------------------------------
# Step1
#------------------------------------------------------------------------------------------------
# A. Change train_id to 1 (i.e. step id) in refresh_TS.py file and run
python refresh_TS.py # Check train_id (it should match with step), number of MD simulations in previous steps, and element_list
cd TS1

# B. Above code has selected new training set configurations and created corresponding CP2K input in folder TS1/DFT_Train/. Now put required CP2K input files (at each TS0/DFT_Train/snap) and run DFT calculations for each configuration.

# C. Once all DFT simulations are finished, run following code to fetch energy, forces, and virial stress values of all configurations and collect all data in axyz format (which is main input to Dam package) using following
cp ../../utils/make_TS_axyz.py .
python make_TS_axyz.py # Change number of atoms accordingly

# D. Fit next version of SNAP (i.e., MLP1) over this training data (containing training configurations from all previous steps)
cd snapfit
cp ../../../utils/snapfit.py .
python snapfit.py
cd ../../            # Going back to directory Training_Set

# E. Now run 50 MD simulations in parallel using trained SNAP (i.e., MLP1)
python training_step_data.py # Change here to modify time-stepsize, number of MD steps, temperature, pressure
python newMD_TS.py # Change ts=1

# F. Run MD simulations (it is at 200 K, change training_step_data.py to modify it for Step0)
sh run_md.sh  #Change sleep time according to your number of simultaneous runs in your hardware

#Now MD simultions with MLP1 have been finished. Now proceed towards Step 1

#-----------------------------------------------------------------------------------------------
# Step2
#------------------------------------------------------------------------------------------------
#*** Follow steps same as Step 1 and increase temperature during MD simulations***
#-----------------------------------------------------------------------------------------------
# Step3
#------------------------------------------------------------------------------------------------
#*** Follow steps same as Step 2 and increase temperature during MD simulations***
#
#
#

