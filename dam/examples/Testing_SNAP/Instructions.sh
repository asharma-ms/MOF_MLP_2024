#rm  *.png *.txt log.lmp snap.eq data.snap

# 0. Activate appropriate conda environment
conda activate Dam

# 1. Make sure that SNAP parameter files snapcoeff  and snapparam and are in the same folder
# 2. Make sure that test data set such as Test_at.xyz is in the same folder
# 3. If necessary, change values in snaptest.py file and run following
python snaptest.py
