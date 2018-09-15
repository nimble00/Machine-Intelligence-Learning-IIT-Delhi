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
	dataset = np.array(dataset[1:leng])
	dataset1 = dataset[:,[1,2,3,6]]
	dataset2 = dataset[:,[1,4,5]]
	return dataset1,dataset2

#-------------------------------------------------------------------------------

def mean(arr):
	arr = arr.astype(np.float)
	return np.sum(arr)/arr.size

#-------------------------------------------------------------------------------

def standev(arr):
	arr = arr.astype(np.float)
	mea = mean(arr);
	squaresum = sum(np.power(arr-mea,2))/float(arr.size)
	return math.sqrt(squaresum)

#-------------------------------------------------------------------------------

def trainprob(arr):
	arr = np.array(arr)
	unique, counts = np.unique(arr, return_counts=True)
	counts = np.float_(counts)/float(arr.size)
	return (unique,counts)

#-------------------------------------------------------------------------------

def probdata(x,mean,stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#-------------------------------------------------------------------------------

def dsepbyclass(dataset):
	dicte = {}
	for i in dataset:
		if(i[0] not in dicte):
			dicte[i[0]] = []
		dicte[i[0]].append(i[1:4])
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

def newprob(dicter):
	dic ={}
	for j in dicter:
		a = zip(*dicter[j])
		unique1,counts1 = np.unique(a[0], return_counts=True)
		unique2,counts2 = np.unique(a[1], return_counts=True)
		b = {}
		c =0
		d= 0
		for k in counts1:
			c +=k
		for l in counts2:
			d +=l
		for qw in range(len(unique1)):
			b[unique1[qw]] = float(counts1[qw])/float(c)
		for qa in range(len(unique2)):
			b[unique2[qa]] = float(counts2[qa])/float(c)
		dic[j] = b
	if len(dic['0']) != len(dic['1']):
		for j in dic['0']:
			if not j in dic['1']:
				dic['1'][j] =0
		for j in dic['1']:
			if not j in dic['0']:
				dic['0'][j] =0
	return dic

#-------------------------------------------------------------------------------

def predictclass(mdict,data,data1,probmatrix,ndict):
	dicter = predictprob(mdict,data,probmatrix)
	bestprob,bestclass = -1,None
	#print dicter
	for clas,prob in dicter.iteritems():

		ab = ndict[clas]

		p = prob*ab[data1[1]]*ab[data1[2]]
		if bestprob< p:
			bestprob = p
			bestclass = clas
	return bestclass

#-------------------------------------------------------------------------------

def predictprob(mdict,data,cd):
	dicter = {}
	re = 0
	for classvalue,classdata in mdict.iteritems():
		prob = 1
		t=1
		for j in classdata:
			prob *= probdata(float(data[t]),j[0],j[1])
			t +=1
		dicter[classvalue] = prob*cd[re]
		re += 1
	return dicter

#-------------------------------------------------------------------------------

def answerdata(file1,file2):
	dataset1,dataset2 = loadCsv(file1)
	dic = dsepbyclass(dataset1)
	dic1 = dsepbyclass(dataset2)
	mdict = meanvar(dic)
	ndict = newprob(dic1)
	dataset3 ,dataset4 = loadCsv(file2)
	#print trainprob
	unique ,counts = trainprob(dataset1[:,1])
	accuracy =0
	for s in range(len(dataset3)):
		j = dataset3[s]
		k = dataset4[s]
		a =  predictclass(mdict,j,k,counts,ndict)
		if(a==j[0]):
			accuracy += 1
	print (float(accuracy)/float(len(dataset1)))*100

#-------------------------------------------------------------------------------

answerdata("railwayBookingList.csv","railwayBookingList.csv")
'''dataset1,dataset2 = loadCsv("railwayBookingList.csv")
print dataset1
print dataset2
print trainprob(dataset1)
print trainprob(dataset2)'''

#-------------------------------------------------------------------------------
