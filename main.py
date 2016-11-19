import logRegres

trainData,trainLabels = logRegres.loadDataSet('data/train-set',13)
weights = logRegres.kgradAscent(trainData,trainLabels)
#print 'weights:\n'
#print weights
testData,testLabels = logRegres.loadDataSet('data/test-set',13)
print 'Original:'
print testLabels
correct,classified = logRegres.ktest(testData,testLabels,weights)
print 'Classified:'
print classified
