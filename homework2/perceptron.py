import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv
import operator
import processing

dataset = []
trainSet = []
testSet = []

def splitDataset(filename, split, trainingSet=[], testingSet=[]):
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in open(filename)]
        dataset = []
        for l in lines:
            string = l#lines[0]
            string = string.split(',')
            # print(string)
            if (string[2][-2] == '0'):
                lst =[float(string[0]), float(string[1]), 0.0]
            else:
                lst =[float(string[0]), float(string[1]), 1.0]

            if random.random() < split:
                trainingSet.append(lst)
            else:
                testingSet.append(lst)
            
            dataset.append(lst)

def train_weights(train, test, learnRate, epoch):
    weights = [0.0 for i in range(len(train[0]))]
    # print(weights)
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    trainErrorRate = []
    testErrorRate = []
    for e in range(epoch):
        sum_error = 0.0
        for row in train:
            pred = predict(row, weights)
            if row[-1] == pred and pred == 1:
                truePos += 1
            elif row[-1] == pred and pred == 0:
                trueNeg += 1
            elif row[-1] == 1 and pred == 0:
                falseNeg += 1
            elif row[-1] == 0 and pred == 1:
                falsePos += 1
            error = row[-1] - pred
            sum_error += error ** 2
            weights[0] = weights[0] + learnRate * error
            for i in range(len(row)-1):
                weights[i+1] = weights[i+1] + learnRate * error * row[i]
        
        # Get Test error rate after each epoch
        testMetrics = testPerceptron(test, weights)
        testErrorRate.append(getErrorRate(testMetrics))
        # Get training set error rate and store it
        trainErrorRate.append(getErrorRate([truePos, trueNeg, falsePos, falseNeg]))
        
        print('epoch=%d, learnRate=%.3f, error=%.3f' %(e, learnRate, sum_error))
        # print(weights)
    # print(testErrorRate)
    # print(trainErrorRate)
    # plotErrorRates(testErrorRate, trainErrorRate)
    metrics = [truePos, trueNeg, falsePos, falseNeg]
    return weights, metrics

def getErrorRate(metrics):
    return 1.0 - (metrics[0] + metrics[1])/sum(metrics)

# def accuracy(actual, predicted):
#     correct = 0
#     for i in range(len(actual)):
#         if actual[i] == predicted[i]:
#             correct += 1
#     return correct / float(len(actual)) * 100.0

def testPerceptron(test, weights):
    truePos = 0
    trueNeg = 0
    falseNeg = 0
    falsePos = 0

    predictions = []
    for row in test:
        pred = predict(row, weights)
        predictions.append(pred)
        if row[-1] == pred and pred == 1:
            truePos += 1
        elif row[-1] == pred and pred == 0:
            trueNeg += 1
        elif row[-1] == 1 and pred == 0:
            falseNeg += 1
        elif row[-1] == 0 and pred == 1:
            falsePos += 1
    return [truePos, trueNeg, falsePos, falseNeg]
             

def perceptron(train, test, learnRate, epoch):
    preds = list()
    # print("Training weights...")
    weights, metrics = train_weights(train, test, learnRate, epoch)
    # print(weights)
    # print("Testing")
    for row in test:
        pred = predict(row, weights)
        preds.append(pred)
    return preds, metrics

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

def accuracy(testSet, preds):
    correct = 0
    truePos = 0
    trueNeg = 0
    falsePos = 0
    falseNeg = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == preds[x] and preds[x] == 1:
            truePos += 1
        elif testSet[x][-1] == preds[x] and preds[x] == 0:
            trueNeg += 1
        elif testSet[x][-1] == 1 and preds[x] == 0:
            falseNeg += 1
        elif testSet[x][-1] == 0 and preds[x] == 1:
            falsePos += 1
    return [truePos, trueNeg, falsePos, falseNeg]

def plotErrorRates(testER, trainER):
    x = np.linspace(0, len(testER), num=len(testER))
    test = plt.scatter(x, testER)
    train = plt.scatter(x, trainER)
    plt.legend((test, train), ('Test Error Rate', 'Train Error Rate'))
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    plt.title('Error Rate for Test and Training Data')
    # plt.show()
    # print('Error Rate Graph Created')
    plt.savefig('errorRate.png')
    

# print("Predictions: " + repr(preds))
def main():
    # print('perceptron')
    filename = 'demo.txt'
    split = 2.0/3.0
    dataset = splitDataset("demo.txt", split, trainSet, testSet)

    n_folds = 3
    l = 0.1
    n_epoch = 500
    preds, trainMetrics = perceptron(trainSet, testSet, l, n_epoch)
    metrics = accuracy(testSet, preds)
    # print(trainMetrics)
    # metrics = [truePos, trueNeg, falsePos, falseNeg]
    # print(metrics)
    hitRate = (metrics[0] + metrics[1] ) / sum(metrics) * 100.0
    trainHitRate = (trainMetrics[0] + trainMetrics[1]) / sum(trainMetrics) * 100.0
    # print('Hit Rate = %.2f %%\t\tTrain Hit Rate = %.2f %%' % (hitRate, trainHitRate))

    sensitivity = metrics[0] / (metrics[0] + metrics[3]) * 100.0
    trainSensitivity = trainMetrics[0] / (trainMetrics[0] + trainMetrics[3]) * 100.0
    # print('Sensitivity = %.2f %%\t\tTrain Sensitivity = %.2f %%' % (sensitivity, trainSensitivity))
    
    specificity = metrics[1] / (metrics[1] + metrics[2]) * 100.0
    trainSpec = trainMetrics[1] / (trainMetrics[1] + trainMetrics[2]) * 100.0
    # print('Specificity = %.2f %%\t\tTrain Specificity = %.2f' % (specificity, trainSpec))
    
    ppv = metrics[0] / (metrics[0] + metrics[1]) * 100.0
    trainPpv = trainMetrics[0] / (trainMetrics[0] + trainMetrics[1]) * 100.0
    # print('PPV = %.2f %%\t\t\tTrain PPV: = %.2f %%' % (ppv, trainPpv))
    
    npv = metrics[1] / (metrics[1] + metrics[3]) * 100.0
    trainNpv = trainMetrics[1] / (trainMetrics[1] + trainMetrics[3]) * 100.0
    # print('NPV = %.2f %%\t\t\tTrain NPV = %.2f %%' % (npv, trainNpv))
    print('\nTrain Size: ' + str(len(trainSet)))
    print('Test Size: ' + str(len(testSet)))



    # Plot Metrics
    # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    # objects = ('Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV')
    # y_pos = np.arange(len(objects))
    # performance = [hitRate, sensitivity, specificity, ppv, npv]
 
    # plt.bar(y_pos, performance, align='center', alpha=0.5)
    # plt.xticks(y_pos, objects)
    # plt.ylabel('Usage')
    # plt.title('Metrics for Perceptron Training')

    objects2 = ('Hit Rate', 'Sensitivity', 'Specificity', 'PPV', 'NPV')
    y_pos2 = np.arange(len(objects2))
    performance2 = [trainHitRate, trainSensitivity, trainSpec, trainPpv, trainNpv]

    plt.bar(y_pos2, performance2, align='center', alpha=0.5)
    plt.xticks(y_pos2, objects2)
    plt.ylabel('Usage')
    plt.title('Metrics for Perceptron Test Set')
 
    plt.show()
    plt.savefig('perceptronMetricsTrain.png')

main()