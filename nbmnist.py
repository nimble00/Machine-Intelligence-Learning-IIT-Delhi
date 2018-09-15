import mnist_reader
import os
import pandas as pd
import numpy as np
import csv
import collections
import math
#-------------------------------------------------------------------------------
def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	leng = len(dataset)
	dataset = dataset[1:leng]
	return dataset

#-------------------------------------------------------------------------------

def mean(arr):
	arr = arr.astype(np.float)
	return np.sum(arr)/arr.size

#-------------------------------------------------------------------------------

def standev(arr):
	arr = arr.astype(np.float)
	mea = mean(arr)
	squaresum = sum(np.power(arr-mea,2))/float(arr.size)
	return math.sqrt(squaresum)

#-------------------------------------------------------------------------------

def trainprob(dicte,X_train):
	#unique, counts = np.unique(arr, return_counts=True)
	#counts = np.float_(counts)/float(arr.size)
	#return (unique,counts)
	a =[]
	lent = len(X_train)
	for j in dicte:
		a.append(float(len(dicte[j]))/float(lent))
	return a

#-------------------------------------------------------------------------------

def probdata(x,mean,stdev):
	if stdev==0:
		return 1
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#-------------------------------------------------------------------------------

def dsepbyclass(X_train,y_train):
	dicte = {}
	for i in range(len(X_train)):
		if(y_train[i] not in dicte):
			dicte[y_train[i]] = []
		dicte[y_train[i]].append(X_train[i])
	return dicte

#-------------------------------------------------------------------------------

def meanvar(dataset):
	dicte = {}
	for i in dataset:
		a = zip(*dataset[i])
		dicte[i] = []
		for j in a:
			p = np.array(j)
			dicte[i].append([mean(p),standev(p)])
	return dicte

#-------------------------------------------------------------------------------

def predictclass(mdict,data,probmatrix):
	dicter = predictprob(mdict,data)
	bestprob,bestclass = -1,None
	t =0
	for clas,prob in dicter.iteritems():
		if bestprob<prob*probmatrix[t] :
			bestprob = prob*probmatrix[t]
			bestclass = clas
		t+=1

	return bestclass

#-------------------------------------------------------------------------------

def predictprob(mdict,data):
	dicter = {}

	for classvalue,classdata in mdict.iteritems():
		prob = 1
		t=0
		for j in classdata:
			prob *= probdata(float(data[t]),j[0],j[1])
			t +=1
		dicter[classvalue] = prob
	return dicter

#-------------------------------------------------------------------------------

def answerdata(file1,file2):
	X_train, y_train = mnist_reader.load_mnist(os.getcwd(), kind='train')
	X_test, y_test = mnist_reader.load_mnist(os.getcwd(), kind='t10k')
	#dataset = loadCsv(file1)
	dic = dsepbyclass(X_train,y_train)
	mdict = meanvar(dic)
	counts = trainprob(dic,X_train)
	accuracy =0
	for j in range(len(X_test)):
		a =  predictclass(mdict,X_test[j],counts)
		if(a==y_train[j]):
			accuracy += 1
			print "correct"
		else:
			print "incorrect"
	print (float(accuracy)/float(len(X_test)))*100

#-------------------------------------------------------------------------------

answerdata("Medical_data.csv","Medical_data.csv")
