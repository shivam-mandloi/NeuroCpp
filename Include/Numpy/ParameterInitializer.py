import numpy as np
import os
import sys


if(len(sys.argv) != 4 and len(sys.argv) != 3):
    print(len(sys.argv))
    raise ValueError("Usage: python ParameterInitializer.py RowSize ColSize fileName.txt / python ParameterInitializer.py SizeOfVector fileName.txt")

m = int(sys.argv[1])
check = False
if(len(sys.argv) == 4):
    check = True
    n = int(sys.argv[2])
    fileName = sys.argv[3]
else:
    fileName = sys.argv[2]

if check:
    arr = np.random.randn(m, n)
else:
    arr = np.random.randn(m)

np.savetxt(fileName, arr)