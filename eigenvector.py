import numpy as np
import math

def vectorEval(vec):
    sqsum = 0
    for x in np.nditer(vec):
        sqsum = sqsum + (x*x)
    return math.sqrt(sqsum)

def LargestEV(mat):
    init = np.ones((mat.shape[0], 1))
    nex = init
    prev = init
    nexVal = 0
    prevVal = 1
    count = 0
    while(nexVal != prevVal and count < 100):
        count = count + 1
        curr = mat.dot(prev)
        maxi = vectorEval(curr)
        nex = curr/maxi
        prevVal = round(vectorEval(prev), 5)
        nexVal = round(vectorEval(nex), 5)
        prev = nex
    return nex

def SecLargestEV(mat):
    vec = LargestEV(mat)
    x = np.ones((mat.shape[0], 1))
    matT = np.transpose(mat)
    vecT = np.transpose(vec)
    e1 = vecT.dot(x)
    e2 = vec.dot(e1)
    y = x - e2
    count = 0
    while(count < 100):
        y = matT.dot(y)
        y = y/vectorEval(y)
        count = count + 1
    return y