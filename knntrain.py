
import pandas as pd
import numpy as np
import csv
import collections
import math

def knnclassifier(file1,file2,k):
	dataset1 = np.array(loadCsv(file1))
	dataset2 = np.array(loadCsv(file2))
	dataset11 = dataset1[:,[1,2,3,6]]
	dataset12 = dataset1[:,[1,4,5]]
	dataset21 = dataset2[:,[1,2,3,6]]
	dataset22 = dataset2[:,[1,4,5]]

	unique,counts  = np.unique(dataset1[:,0],return_counts=True)
	accuracy =0
	for i in range(len(dataset2)):
		#print i
		clas = find(dataset21[i],dataset22[i],dataset11,dataset22,k)
		#print "---"
		#print clas
		#print"-----"
		if clas == dataset21[i][0]:
			accuracy +=1
	print (float(accuracy)/float(len(dataset2)))*100




def find(data,data1,file,file1,k):
	a = []
	for j in range(len(file)):
		dis = finddis(data,data1,file[j],file1[j])
		a.append([file[j][0],dis])
	#print len(a)
	b = findclass(a,k)
	return b

def findclass(a,k):
	m = {}
	b =[]
	n = len(a)
	for i in range(k):
		p =float("inf")
		c = ''
		for j in range(i,n-1):
			if a[j][1] <p:
				p = a[j][1]
				c = a[j][0]
		b.append([c,p])
	#print b
	for d in range(k-1):
		if b[d][0] not in m:
			m[b[d][0]] =1
		else:
			m[b[d][0]] +=1
	maxclass = None
	maxvalue = -1
	for clas,value in m.iteritems():
		if maxvalue<value:
			maxvalue = value
			maxclass = clas
	return maxclass

def finddis(data1,data3,data2,data4):
	a = np.array(data1[1:3])
	a = a.astype(np.float)
	b = np.array(data2[1:3])
	b = b.astype(np.float)
	c = math.sqrt(np.sum((a-b)**2))
	if data3[1] != data4[1]:
		c+=2
	if data3[2] != data4[2]:
		c+=2
	return c

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	leng = len(dataset)
	dataset = dataset[1:leng]
	return dataset
knnclassifier("railwayBookingList.csv","railwayBookingList.csv",25)
