import numpy as np

def loadDataSet(filename, num_attr):
	dataMat = []; labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split(',')		# Get tokens in line
		labelMat.append(int(lineArr[0]))		# Get class id
		dataMatRow = [1.0]				# This entry of dataMatRow corresponds to constant parameter in linear function
		for i in range(1,num_attr + 1):
			dataMatRow.append(float(lineArr[i]))	# Get attributes
		dataMat.append(dataMatRow)
	return dataMat,labelMat

def sigmoid(inX):
	return 1.0/(1 + np.exp(-inX))

# Multi-variables version of sigmoid function
# Number of categories: k
# Number of entries of sample: n
# Category: v
def ksigmoid(x, v, weights):
	n,k = np.shape(weights)	
	temp = np.copy(weights)
	temp_v = np.copy(weights[:,v])
	for i in range(k):
		temp[:,i] -= temp_v
	return 1.0 / float(np.sum(np.exp(temp.transpose() * x)))

# Generate sigmoid matrix of data with weights
def sigmoidArray(dataMat, weights):
	n, m = np.shape(dataMat)
	dump, k = np.shape(weights)
	sigMat = []
	for i in range(m):
		sigMat_row = []
		for j in range(k):
			sigMat_row.append(ksigmoid(dataMat[:,i],j,weights))
		sigMat.append(sigMat_row)
	return sigMat

# Kronecker delta
def kronecker(i,j):
	if i == j:
		return 1
	else:
		return 0

def kroneckerVec(x, y):
	delta = []
	n = len(x)
	for i in range(n):
		delta.append(kronecker(x[i],y[i]))
	return delta

# Create Kronecker delta matrix between vector x and y
def kroneckerArray(x, y):
	m = len(x)
	k = len(y)
	delta = []
	for i in range(m):	# For every sample label
		delta_row = []
		for j in range(k):	# Categories from 1 to k
			delta_row.append(kronecker(x[i],y[j]))
		delta.append(delta_row)
	return delta

# Number of traning samples: m
# Number of entries of sample: n
# Step width: alpha
# Iteration times: maxCycles
def gradAscent(dataMatIn,classLabels):
	dataMatrix = np.mat(dataMatIn)
	labelMat = np.mat(classLabels).transpose()
	m,n = np.shape(dataMatrix)
	alpha = 0.001
	maxCycles = 500
	weights = np.ones((n,1))
	for k in range(maxCycles):
		h = sigmoid(dataMatrix * weights)
		error = (labelMat - h)
		weights = weights + alpha * dataMatrix.transpose() * error
	return weights

# Multi-variables version of gradAscent function
def kgradAscent(data, labels):
	dataMat = np.mat(data).transpose()
	#labelMat = mat(labels)
	n, m = np.shape(dataMat)
	k = np.max(labels)
	alpha = 1.0
	beta = 0.99
	maxCycles = 500
	weights = np.ones((n,k))


	# Create delta matrix between labels and categories
	delta = kroneckerArray(labels, range(1, k + 1))
	deltaMat = np.mat(delta)

	for i in range(maxCycles):
		sigArray = sigmoidArray(dataMat,weights)
		sigMat = np.mat(sigArray)
		errorMat = deltaMat - sigMat
		# Shrink step width 
		alpha *= beta
		weights += alpha * dataMat * errorMat

	return weights
	
def test(dataMatIn,classLabels,weights):
	dataMat = np.mat(dataMatIn)
	labelMat = np.mat(classLabels)
	m,n = np.shape(dataMat)
	s = sigmoid(dataMat * weights)
	classified = []
	for i in range(len(classLabels)):
		if s[i] < 0.5:
			classified.append(0)
		else:
			classified.append(1)
	classifiedMat = np.mat(classified)
	errorVec = abs(labelMat - classifiedMat)
	correctRate =  1.0 - float(np.sum(errorVec)) / len(classLabels)
	print 'The correct rate of the classifier is: ' + str(correctRate)
	return correctRate,classified

# Multi-variables version of test function
def ktest(data, labels, weights):
	dataMat = np.mat(data).transpose()
	#labelMat = np.mat(labels)
	dump, k = np.shape(weights)
	n, m = np.shape(dataMat)
	sigArray = sigmoidArray(dataMat, weights)
	classified = []
	for i in range(m):
		# Choose the category in which sample i has largest propability to be there
		classified.append(sigArray[i].index(max(sigArray[i])) + 1)
	deltaVec = kroneckerVec(classified, labels)
	correctRate = float(sum(deltaVec)) / m
	print 'Correct rate of classifier: ' + str(correctRate)
	return correctRate, classified
	








