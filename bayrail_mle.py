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
priori_d = {}

#-------------------------------------------------------------------------------

def cal_priori_d():
    global dets
    global training_freq
    global training_data
    global priori
    global sigs
    global mus
    global priori_d
    dicd = {}
    for y,x,x1 in training_data:
        if str(x1) not in dicd.keys():
            #print x1
            dicd[str(x1)] = 0
        if y==1:
            dicd[str(x1)]+=1
    for a in dicd.keys():
        dicd[a] = dicd.get(a)/float(len(training_data))
    priori_d = dicd.copy()
    return dicd


#-------------------------------------------------------------------------------

def cal_priori():
    global dets
    global training_freq
    global training_data
    global priori
    global sigs
    global mus
    global priori_d
    dic = training_freq.copy()
    for i in dic.keys():
        dic[i] = float(dic.get(i))/float(len(training_data))
    priori = dic.copy()
    return dic

#-------------------------------------------------------------------------------

def mean_mat():
    global dets
    global training_freq
    global training_data
    global priori
    global sigs
    global mus
    global priori_d
    dic = {}
    for y,x,x1 in training_data:
        if y not in dic:
            dic[y] = []
        dic[y].append(x)
    for j in dic.keys():
        dic[j] = np.mean(dic.get(j),axis=0)
    mus = dic.copy()
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

def mul_exp(X,y):
    return 0

#-------------------------------------------------------------------------------

def train_rn():
    global dets
    global training_freq
    global training_data
    global priori
    global sigs
    global mus
    global priori_d
    cal_priori()
    cal_priori_d()
    mean_mat()

#-------------------------------------------------------------------------------

def bayes_c(X,X1):
    global dets
    global training_freq
    global training_data
    global priori
    global sigs
    global mus
    global priori_d
    clas = 1
    max = 0
    dik ={}
    for i in training_freq.keys():
        if i==1:
            post_d = priori_d.get(X1)
        else:
            post_d = 1-priori_d.get(X1)
        dik[i] = mulvar_nor(X,i)*priori.get(i)*post_d
    if dik.get(0)>dik.get(1):
        return 0
    return 1

#-------------------------------------------------------------------------------

def test_rn():
    fp = 0
    cp = 0
    fn = 0
    cn = 0
    p = 0
    f = 0
    t = 0
    for v,u,c in test_data:
        t+=1
        if str(bayes_c(u,str(c)))==str(v):
            p+=1
        else:
            f+=1
    print "accuracy = " + str(float(p)/float(t))

#-------------------------------------------------------------------------------

if __name__ == '__main__':

#-------------------------------------------------------------------------------

    #df = pd.read_csv("Medical_data.csv")
    #y_train = np.array(df['Health'].values.tolist())
    #X_train = df[['TEST1','TEST2','TEST3']].values.tolist()
    df = pd.read_csv("railwayBookingList.csv")
    y_train = np.array(df['boarded'].values.tolist())
    X_train = df[['budget', 'memberCount', 'age']].values.tolist()
    sigs = np.cov(X_train,rowvar=0)
    dets = np.linalg.det(sigs)
    X_train1 = df[['preferredClass', 'sex']].values.tolist()
    training_data = [x for x in zip(y_train[:900],X_train[:900],X_train1[:900])]
    #print "length of training_data: "+str(len(training_data))
    test_data = [x for x in zip(y_train[900:-2],X_train[900:-2],X_train1[900:-2])]
    #print "length of test_data: "+str(len(test_data))
    unique, counts = np.unique(y_train, return_counts=True)
    training_freq = dict(zip(unique,counts))
    train_rn()
    test_rn()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
