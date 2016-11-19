def loadDataSet(filename, num_attr):
	dataMat = []; labelMat = []
	fr = open(filename)
	for line in fr.readlines():
		lineArr = line.strip().split(',')		# Get tokens in line
		labelMat.append(int(lineArr[0]))		# Get class id
		dataMatRow = []				# This entry of dataMatRow corresponds to constant parameter in linear function
		for i in range(1,num_attr + 1):
			dataMatRow.append(float(lineArr[i]))	# Get attributes
		dataMat.append(dataMatRow)
	return dataMat,labelMat

def binaryParser(filename, num_attr, num_class):		# Seperate multiple classes data set into subset containing only 2 classes
	dataMat,labelMat = loadDataSet(filename, num_attr)	# Read in raw data
	# Seperate data set into subsets that only contain 1 class
	dataSubMat = []; labelSubMat = []
	cut_index = []
	for i in range(1,num_class + 1):
		cut_index.append(labelMat.index(i))
	cut_index.append(len(labelMat))
	for i in range(len(cut_index) - 1):
		labelSubMat.append(labelMat[cut_index[i]:cut_index[i+1]])
		dataSubMat.append(dataMat[cut_index[i]:cut_index[i+1]])
	# Combine 2 classes into one data set
	comb_order = []
	for i in range(1,num_class + 1):
		for j in range(i + 1, num_class + 1):
			comb_order.append([i,j])
	newDataMat = []; newLabelMat = []
	for i in range(len(comb_order)):
		newLabelMat.append(labelSubMat[comb_order[i][0] - 1]+labelSubMat[comb_order[i][1] - 1])
		newDataMat.append(dataSubMat[comb_order[i][0] - 1]+dataSubMat[comb_order[i][1] - 1])
	return comb_order,newDataMat,newLabelMat

def makePair(filename,num_attr,num_class):
	comb_order,dataMat,labelMat = binaryParser(filename,num_attr,num_class)
	fw = []
	for i in range(len(comb_order)):
		# Normalize labels
		mid = labelMat[i].index(max(labelMat[i]))
		labelMat[i][0:mid] = [0] * mid
		labelMat[i][mid:len(labelMat[i])] = [1] * (len(labelMat[i]) - mid)
		# Write in data
		fw = open(str(comb_order[i][0]) + '-' + str(comb_order[i][1]) + '.data','w')
		for j in range(len(labelMat[i])):
			fw.write(str(labelMat[i][j]) + ',' + str(dataMat[i][j])[1:-1] + '\n')
		fw.close()

def makeTrainTest(filename,num_attr,ratio):	# Seperate data set into train set (ratio * 100 %) and test one ((1 - ratio) * 100 %)
	dataMat,labelMat = loadDataSet(filename,num_attr)
	mid = labelMat.index(max(labelMat))
	trainSize = [int(mid * ratio), int((len(labelMat) - mid) * ratio)]
	trainData = dataMat[0 : trainSize[0]] + dataMat[mid : mid + trainSize[1]]
	testData = dataMat[trainSize[0] : mid] + dataMat[mid + trainSize[1] :]
	trainLabel = labelMat[0 : trainSize[0]] + labelMat[mid : mid + trainSize[1]]
	testLabel = labelMat[trainSize[0] : mid] + labelMat[mid + trainSize[1] :]
	# Write in data
	fw = open('train-' + filename,'w')
	for i in range(len(trainLabel)):
		fw.write(str(trainLabel[i]) + ',' + str(trainData[i])[1:-1] + '\n')
	fw.close()
	fw = open('test-' + filename,'w')
	for i in range(len(testLabel)):
		fw.write(str(testLabel[i]) + ',' + str(testData[i])[1:-1] + '\n')
	fw.close()	

# Multi-variables version of makeTrainTest function
def kMakeTrainTest(filename, num_attr, ratio):
	data, labels = loadDataSet(filename, num_attr)
	# max label
	k = max(labels)
	# Positions of categories
	pos = []
	for i in range(1, k + 1):
		pos.append(labels.index(i))
	pos.append(len(labels) - 1)
	trainData = []; trainLabel = []; testData = []; testLabel = []
	for i in range(len(pos) - 1):
		cut = pos[i] + int((pos[i + 1] - pos[i]) * ratio)
		trainData += data[pos[i] : cut]
		testData += data[cut : pos[i + 1]]
		trainLabel += labels[pos[i] : cut]
		testLabel += labels[cut : pos[i + 1]]
	# Write in
	fw = open('train-set', 'w')
	for i in range(len(trainLabel)):
		fw.write(str(trainLabel[i]) + ',' + str(trainData[i])[1:-1] + '\n')
	fw.close()
	fw = open('test-set' ,'w')
	for i in range(len(testLabel)):
		fw.write(str(testLabel[i]) + ',' + str(testData[i])[1:-1] + '\n')
	fw.close()	

	











