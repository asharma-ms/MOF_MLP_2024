
# 0. Activate appropriate conda environment
conda activate Dam

# 1. Make sure that training data set such as Train_D672_at.xyz is in same folder
# 2. If necessary, change values in Hyperparmeter_optimize.py file and run following
python snapfit.py

# To check columns of optimization_run_efs.txt file, check get_error() and opt_para() function of SnapEQ class in aSNAP.py file
