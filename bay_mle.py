import math,copy
import numpy as np
import pandas as pd
#import cv2, scipy
#for simplicity I'll assume 0-1 loss
# D = { Xi's }
# Xi = {xi1,xi2,...,xin} (used in code as "X")
#-------------------------------------------------------------------------------

mus = []
training_freq = {}
training_data = []
test_data = []
sigs = []
dets = 1
priori = []

#-------------------------------------------------------------------------------

def cal_priori():
    global training_freq
    global training_data
    global test_data
    global m0,s0
    global sigs
    global dets
    global priori
    global mus
    dic = training_freq.copy()
    for i in dic.keys():
        dic[i] = float(dic.get(i))/float(len(training_data))
    priori = dic.copy()
    return dic

#-------------------------------------------------------------------------------

def mean_mat():
    global training_freq
    global training_data
    global test_data
    global m0,s0
    global sigs
    global dets
    global priori
    global mus
    dic = {}
    for y,x in training_data:
        if y not in dic:
            dic[y] = []
        dic[y].append(x)
    for j in dic.keys():
        dic[j] = np.mean(dic.get(j),axis=0)
    mus = dic
    return dic

#-------------------------------------------------------------------------------

def mulvar_nor(X,y):
    global training_freq
    global training_data
    global test_data
    global m0,s0
    global sigs
    global dets
    global priori
    global mus
    c = math.log(math.pow(2*math.pi,-0.5*len(X)))+math.log(math.pow(np.real(dets),-0.5))
    xut = np.array([X-mus.get(y)])
    sig_i = np.linalg.inv(sigs)
    pd = np.matmul(xut , sig_i)
    xu = np.array([X-mus.get(y)]).T
    e = (-0.5)*np.linalg.det(np.matmul(pd,xu))
    return c+e

#-------------------------------------------------------------------------------

def train_rn():
    global training_freq
    global training_data
    global test_data
    global m0,s0
    global sigs
    global dets
    global priori
    global mus
    priori = cal_priori()
    mus = mean_mat()

#-------------------------------------------------------------------------------

def bayes_c(X):
    global training_freq
    global training_data
    global test_data
    global m0,s0
    global sigs
    global dets
    global priori
    global mus
    clas = ""
    max = 0
    dik={}
    for i in training_freq.keys():
        dik[i] = mulvar_nor(X,i)+math.log(priori.get(i))
    pm = -999999
    for v in dik.keys():
        if dik.get(v)>pm:
            pm = dik.get(v)
            clas = v
    #print str(te)+ " " + str(mulvar_nor(X,j)*priori.get(j))
    return clas

#-------------------------------------------------------------------------------

def test_rn():
    fp = 0
    cp = 0
    fn = 0
    cn = 0
    p = 0
    f = 0
    t = 0
    for v,u in test_data:
        t+=1
        k = str(bayes_c(u))
        if k==str(v):
            p+=1
        else:
            f+=1
    print "accuracy = " + str(float(p)/float(t))

#-------------------------------------------------------------------------------

if __name__ == '__main__':
    df = pd.read_csv("Medical_data.csv")
    y_train = np.array(df['Health'].values.tolist())
    X_train = df[['TEST1','TEST2','TEST3']].values.tolist()
    df = pd.read_csv("test_medical.csv")
    y_test = np.array(df['Health'].values.tolist())
    X_test = df[['TEST1','TEST2','TEST3']].values.tolist()
    sigs = np.cov(X_train,rowvar=0)
    dets = np.linalg.det(sigs)
    training_data = [x for x in zip(y_train,X_train)]
    test_data = [x for x in zip(y_test,X_test)]
    unique, counts = np.unique(y_train, return_counts=True)
    training_freq = dict(zip(unique,counts))
    train_rn()
    test_rn()

#-------------------------------------------------------------------------------
