import numpy as np
import pandas as pd
import tensorflow
import tensorflow.estimator as estimator
import tensorflow as tf
from sklearn.model_selection import train_test_split as skSplit
import datetime
import sys

def getInputFunc(data, y, n_Epochs=None, shuffle=True):
    numExamples = len(yTrain)
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((dict(data), y))
        if shuffle:
            dataset = dataset.shuffle(numExamples)
        dataset = dataset.repeat(n_Epochs)
        dataset = dataset.batch(numExamples)
        return dataset
    return input_fn


def getTestInputFunc(data):
    numExamples = len(data)
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(dict(data))
        dataset = dataset.batch(numExamples)
        return dataset
    return input_fn



numSteps = 200
trainFname = 'feature103_Train.txt'
testFname = 'features103_Test.txt'
if len(sys.argv) > 1:
    if sys.argv[1] == 1:
        trainFname = 'featuresall_train.txt'
        testFname = 'featuresall_test.txt'
        numSteps = 10

tf.enable_eager_execution()

tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(123)

# Load dataset.
trainSet = pd.read_csv(trainFname,sep='\t')
testSet = pd.read_csv(testFname,sep='\t')
ids = testSet.pop('#defLine')

#print(trainSet.shape)

trainSet.pop('#defLine')
ySet = trainSet.pop('class')

xTrain, xTest, yTrain, yTest = skSplit(trainSet, ySet, test_size=.2, random_state=123)

fc = tf.feature_column

col_names = trainSet.columns
print(col_names)
featColumns = []
for feature_name in col_names:
  featColumns.append(fc.numeric_column(feature_name,
                                           dtype=tf.float32))



trainInputFunc = getInputFunc(xTrain, yTrain)
evalInput = getInputFunc(xTest, yTest, n_Epochs=1, shuffle=False)

testInput = getTestInputFunc(testSet)


n_batches = 1
est = estimator.BoostedTreesClassifier(featColumns,
                                          n_batches_per_layer=n_batches)
print("Starting Training")
est.train(trainInputFunc, steps=200)

print("Completed {} steps of training".format(numSteps))

results = est.evaluate(evalInput)
print('Accuracy : ', results['accuracy'])

preds = est.predict(testInput)
classPredictions = []
for pred in preds:
    classPredictions.append(pred['class_ids'][0])

timeStmp = datetime.datetime.today().strftime('%H_%M')
fname = 'output_{}.txt'.format(timeStmp)
with open(fname, 'w') as f:
    for i, rnaID in enumerate(ids):
        outputStr = "{}, {}\n".format(rnaID, classPredictions[i])
        f.write(outputStr)
