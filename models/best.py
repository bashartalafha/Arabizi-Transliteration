import os
rootdir = '/home/ub05user/Arabizi/logs'

max_accuracy = 0
at_file=""
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if "result" in file:
            reader = open( os.path.join(subdir, file)).readlines()
            acc = float(reader[0].split()[1])
            if acc > max_accuracy:
                max_accuracy = acc
                at_file = os.path.join(subdir, file)

print ("Max acuracy", max_accuracy,"\nAt file",at_file)
