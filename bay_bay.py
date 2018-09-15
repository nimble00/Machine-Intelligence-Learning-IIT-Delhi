import math,copy
import numpy as np
import pandas as pd
from utils import mnist_reader
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
m0 = 1
s0 = 1
n = 1

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
    global n
    dic = {}
    for y,x in training_data:
        if y not in dic:
            dic[y] = []
        dic[y].append(x)
    for j in dic.keys():
        dic[j] = np.mean(dic.get(j),axis=0)
    ns0 = np.array(n*s0)
    sigs = sigs.astype(float)
    ns0 = ns0.astype(float)
    for va in dic.keys():
        mn0 = np.matmul(np.matmul(n*s0,np.linalg.inv(np.array(ns0+sigs))),dic.get(va))
        mn1 = np.matmul(np.matmul(sigs,np.linalg.inv(np.array(ns0+sigs))),m0)
        mn = np.array(mn0+mn1)
        dic[va] = mn
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
    #print dets
    #c = math.log(math.pow(2*math.pi,-0.5*len(X)))+math.log(math.pow(np.real(dets),-0.5))
    #math.log(2*math.pi) = 1.8378770664093453
    c = -0.5*len(X)*1.8378770664093453+ -0.5*abs(dets)
    xut = np.array([X-mus.get(y)])
    sig_i = np.linalg.inv(sigs)
    pd = np.matmul(xut , sig_i)
    xu = np.array([X-mus.get(y)]).T
    e = (-0.5)*np.linalg.det(np.matmul(pd,xu))
    return c+e

#-------------------------------------------------------------------------------

def mul_exp(X,y):
    return lam.get(y)*math.exp(-1*np.matmul(lam.get(y),X))

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
        if t%100==0:
            print (t)
        k = str(bayes_c(u))
        if k==str(v):
            p+=1
        else:
            f+=1
    print "accuracy = " + str(float(p)/float(t))

#-------------------------------------------------------------------------------

def pca(X):
    a = []
    for i in X:
        M = np.mean(i)
        i = i-M
        b = i.reshape(28,28)
        c = np.cov(b)
        values,vectors = np.linalg.eig(c)
        eig_pair = [(np.abs(values[i]),vectors[:,i]) for i in range(len(values))]
        eig_pair.sort(key=lambda x:x[0])
        eig_pair.reverse()
        ab = []
        for j in range():
            ab = np.concatenate((ab,eig_pair[j][1]),axis = None)
        a.append(ab)
    return a

#-------------------------------------------------------------------------------
if __name__ == '__main__':

#-------------------------------------------------------------------------------

    X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
    #X_train = np.array(pca(X_train))
    X_train = np.array(X_train).real.astype(float)
    sigs = np.cov(X_train,rowvar=0)
    dets = np.linalg.slogdet(sigs)[-1]
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    #X_test = pca(X_test)
    X_test = np.array(X_test).real.astype(float)
    training_data = [x for x in zip(y_train,X_train[:-len(X_train)/10])]
    val_data = [x for x in zip(y_train,X_train[-len(X_train)/10:])]
    m0 = np.mean(X_train[-len(X_train)/10:],axis=0)
    s0 = np.cov(X_train[-len(X_train)/10:],rowvar=0)
    n=len(X_train)-len(X_train)/10
    sn = np.matmul(np.matmul(sigs,np.linalg.inv(np.array(n*s0+sigs))),s0)
    sigs = np.array(sn+sigs)
    test_data = [x for x in zip(y_test,X_test[:-len(X_test)/10])]
    unique, counts = np.unique(y_train, return_counts=True)
    training_freq = dict(zip(unique,counts))
    train_rn()
    test_rn()

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
