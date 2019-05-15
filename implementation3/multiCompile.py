import os

defaultDropOut = 0.2
defaultMomentum = 0.5
defaultWeightDecay = 0.
fileName = "GraphPic"
counter = 1

# cmd = "python q3.py %.1f %.1f %.1f %s" % (defaultDropOut, defaultMomentum, defaultWeightDecay, (fileName + str(counter)))
# print("Running: " + cmd)
# os.system(cmd)

for dropOut in [0., 0.2, 0.4, 0.6, 0.8]:
    break
    # print("NUMT = %d" % threads)
    cmd = "python q3.py %.1f %.1f %.1f %s" % (dropOut, defaultMomentum, defaultWeightDecay, (fileName + str(counter)))
    counter += 1
    os.system(cmd)

for momentum in [0, 0.5, 1., 1.5, 2.]:
    if(momentum == 0 or momentum == 0.5):
        continue
    # print("NUMNODES = %d" % nodes)
    cmd = "python q3.py %.1f %.1f %.1f %s" % (defaultDropOut, momentum, defaultWeightDecay, (fileName + str(counter)))
    counter += 1
    os.system(cmd)

for weightDecay in [0, 0.5, 1., 1.5, 2.]:
    cmd = "python q3.py %.1f %.1f %.1f %s" % (defaultDropOut, defaultMomentum, weightDecay, (fileName + str(counter)))
    counter += 1
    os.system(cmd)

#     cmd = "prog"
#     os.system(cmd)
#     print()