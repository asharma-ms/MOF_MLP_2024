#rm -rf *.png *.txt data* snapparam snapcoeff

# 0. Activate appropriate conda environment
conda activate Dam

# 1. Make sure that training data set such as Train_D672_at.xyz is in same folder
# 2. If necessary, change values in snapfit.py file and run following
python snapfit.py
