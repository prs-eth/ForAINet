import sys
sys.path.append("../build/src")
import libcp
import numpy as np
obs = np.array([[10.,20.],[0.,1.],[0.,0. ]], dtype = 'float32')
source = np.array([0, 1], dtype = 'uint32')
target = np.array([1, 2], dtype = 'uint32')
edge_weight = np.array([[1., 1.]], dtype = 'float32')
[comp, in_com] = libcp.cutpursuit(obs, source, target, edge_weight, 1)
# sol = libcp.cutpursuit_reg(obs, source, target, edge_weight, 1)
