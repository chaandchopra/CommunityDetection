import numpy as np
import networkx as nx
from eigenvector import SecLargestEV
from eigenvector import LargestEV

def cut_edge(c1, c2, data):
    res1 = 0
    for i in c1:
        for j in c2:
            if(data.has_edge(str(i+1), str(j+1))):
                res1 = res1+ 1           
    return res1

def spectral(ssEV):
    node_index = sorted(range(len(ssEV)), key=lambda k: ssEV[k])
    class1_index = node_index[:16]
    class2_index = node_index[16:]
    cuts = cut_edge(class1_index, class2_index, data)
    class3_index = node_index[:18]
    class4_index = node_index[18:]
    cuts1 = cut_edge(class3_index, class4_index, data)
    if(cuts > cuts1):
        return class3_index, class4_index, 18
    else:
        return class1_index, class2_index, 16

def modularity(lEV):
    negcount, poscount = 0, 0
    neg, pos = [], []
    for i in range(0, len(lEV)):
        if lEV[i] < 0:
            negcount= negcount + 1 
            neg.append(i)
        else:
            poscount = poscount + 1
            pos.append(i)
    return neg,pos, negcount, poscount

data = nx.read_edgelist('data.txt')
lapData = nx.laplacian_matrix(data)
modData = nx.modularity_matrix(data)

degreeList = data.degree()
sorted_list = sorted(degreeList, key= lambda x: x[1])
max_deg = sorted_list[-1][1]

imat = np.identity(lapData.shape[0])
eval_mat = ((2*max_deg)*imat) - lapData
L = SecLargestEV(eval_mat)

print("Spectral bisection algorithm (based on second smallest eigen value of L):")
comm1, comm2, size = spectral(L)
comm1.sort()
comm2.sort()
print("size", str(size), "Community 1:", str(comm1), "\nsize", str(34-size) ,"Community 2: ", str(comm2))

B = LargestEV(modData)
commu1, commu2, size1, size2 = modularity(B)
print("Using  Modularity maximization algorithm (based on largest eigen value of B):")
commu1.sort()
commu2.sort()
print("size", str(size1),"  Community 1 :", str(commu1))
print("size", str(size2),"  Community 2 :", str(commu2))
