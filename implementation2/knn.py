import os

# Run this script to complete part 1_3.

for k in range(1, 52, 2):
    # print("k = %d" % k)
    cmd = "python q1.py knn_train.csv knn_test.csv %d" % (k)
    os.system(cmd)