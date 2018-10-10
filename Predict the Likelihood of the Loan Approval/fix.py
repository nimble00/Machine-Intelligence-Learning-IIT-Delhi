import pandas as pd
import numpy as np
import random
import math
import copy

#-------------------------------------------------------------------------------

training_freq = {}
training_data = []
test_data = []
s1f = []
s0f = []
maxo = []

#-------------------------------------------------------------------------------

def readr():
    global training_freq
    global training_data
    global test_data
    df = pd.read_csv("train.csv")
    df.fillna("000", inplace=True)
    xd = []
    xb = []
    xi = []
    y = []
    for a in df.columns:
        if "d" in str(a):
            xd.append(a)
        if "l" in str(a):
            xi.append(a)
        if "b" in  str(a):
            xb.append(a)
    xlf = []
    for fe in df[xi].values.tolist():
        fc = fe[:]
        for f in range(len(fc)):
            if fc[f]!=0:
                fc[f] = 1
        xlf.append(fc)
    xbf = []
    for fb in df[xb].values.tolist():
        fk = fb[:]
        for g in range(len(fk)):
            if fk[g]>0:
                fk[g] = 100
        xbf.append(fk)
    xdf = []
    for fd in df[xd].values.tolist():
        fk = fd[:]
        if fk[1]>500:
            fk[1] = 780.0
        elif fk[1]==-1:
            fk[1]=-1
        else:
            fk[1] = 780.0
        xdf.append(fk)
    X_train = [x for x in zip(xdf,xbf,xlf)]
    y_train = np.array(df['target'].values.tolist())
    training_data = [x for x in zip(y_train,X_train)]
    #print "length of training_data: "+str(len(training_data))
    unique, counts = np.unique(y_train, return_counts=True)
    training_freq = dict(zip(unique,counts))
    print training_freq
    return

#-------------------------------------------------------------------------------

def separated():
    global training_data
    global s1f
    global s0f
    global maxo
    sep = {}
    for d in training_data:
		a = d[0]
		if a not in sep:
			sep[a] = []
		sep[a].append(d[1][0] + d[1][1] + d[1][2])
    zer = sep.get(0)
    one = sep.get(1)
    lol = []
    for i in range(len(zer[0])):
        lst = [item[i] for item in zer]
        lol.append(lst)
    lll = []
    for i in range(len(one[0])):
        lt = [item[i] for item in one]
        lll.append(lt)
    s0f = []
    for r in range(len(lol)):
        unique, counts = np.unique(lol[r], return_counts=True)
        f_freq = dict(zip(unique,counts))
        s0f.append(f_freq)
    s1f = []
    for a in range(len(lll)):
        unique, counts = np.unique(lll[a], return_counts=True)
        f_freq = dict(zip(unique,counts))
        s1f.append(f_freq)
    for c in s0f:
        ad = 0
        if '000' in c:
            ad = c.get("000")
            c.pop("000")
        ke = max(c, key=c.get)
        c[ke] = c.get(ke) + ad
        maxo.append(ke)
    return

#-------------------------------------------------------------------------------

def fz(xti):
    global s0f
    p=1
    for fa in range(len(xti)):
        if xti[fa] not in s0f[fa].keys():
            return 1
        if fa>10:
            if fa<39:
                p=p*3.0
            else:
                p=p*6.0
        p = p * (s0f[fa].get(xti[fa])/float(len(training_data)))
    return p

#-------------------------------------------------------------------------------

def fw(xtu):
    global s1f
    p=1
    for fa in range(len(xtu)):
        if xtu[fa] not in s1f[fa].keys():
            return 1
        if fa>10:
            if fa<39:
                p=p*3.0
            else:
                p=p*6.0
        p = p * (s1f[fa].get(xtu[fa])/float(len(training_data)))
    return p

#-------------------------------------------------------------------------------

def cal_prob(xin):
    global training_freq
    p1 = training_freq.get(1)/float(training_freq.get(1) + training_freq.get(0))
    p0 = 1-p1
    s = fw(xin)*float(p1)
    q = fz(xin)*float(p0)
    return s*p1/float(s+q)
    #return fw(xin)/float(s+q)

#-------------------------------------------------------------------------------

def test():
    i = 0
    global test_data
    global maxo
    df1 = pd.read_csv("test.csv")
    df1.fillna("000", inplace=True)
    #print(df1.isnull().values.any())
    xd = []
    xb = []
    xi = []
    for a in df1.columns:
        if "d" in str(a):
            xd.append(a)
        if "l" in str(a):
            xi.append(a)
        if "b" in  str(a):
            xb.append(a)
    xlf = []
    for fe in df1[xi].values.tolist():
        fc = fe[:]
        for f in range(len(fc)):
            if fc[f]!=0:
                fc[f] = 1
        xlf.append(fc)
    xbf = []
    for fb in df1[xb].values.tolist():
        fk = fb[:]
        for g in range(len(fk)):
            if fk[g]>0:
                fk[g] = 100
        xbf.append(fk)
    xdf = []
    for fd in df1[xd].values.tolist():
        fk = fd[:]
        if fk[1]>500:
            fk[1] = 780.0
        elif fk[1]==-1:
            fk[1]=-1
        else:
            fk[1] = 780.0
        xdf.append(fk)
    X_test = xdf[:]
    for va in range(len(xlf)):
        X_test[va] = xdf[va] + xbf[va] + xlf[va]
        for a in range(len(X_test[va])):
            if X_test[va][a]=="000":
                X_test[va][a] = maxo[a]
    #print "length of training_data: "+str(len(training_data))
    apl = df1['APP_ID_C'].values.tolist()
    fi = 0
    data = []
    for cx in X_test:
        a = []
        a.append(apl[fi])
        fi +=1
        tp = cal_prob(cx)
        a.append(tp)
        data.append(a)
        if fi%100==0:
            print fi
    dff = pd.DataFrame(data,columns=['APP_ID_C','target'])
    dff.to_csv("sub.csv", index=False, encoding='utf-8')
    return

#-------------------------------------------------------------------------------

def main():
    readr()
    separated()
    test()
main()

#-------------------------------------------------------------------------------
