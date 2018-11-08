import matplotlib.pyplot as plt
import numpy as np
import random
import math
import csv
import operator
import processing
# import perceptron


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

def euclideanDist(inst1, inst2, length):
    dist = 0
    for x in range(length):
        dist += pow( (inst1[x] - inst2[x] ), 2)
    return math.sqrt(dist)

def findNeighbors(trainingSet, testingInst, radius):
    dist = []
    length = len(testingInst)-1

    for x in range(len(trainingSet)):
        distance = euclideanDist(testingInst, trainingSet[x], length)
        if distance < radius:
            dist.append((trainingSet[x], distance))
    dist.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(len(dist)):
        neighbors.append(dist[x][0])
    return neighbors

def getResponse(neighbors):
    if neighbors == []:
        return 0.0
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(list(classVotes.items()), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

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
    #     if testSet[x][-1] is preds[x]:
    #         correct += 1
    # return (correct / float(len(testSet))) * 100.0


def hitRate(metrics):
    return (metrics[0] + metrics[1])/sum(metrics) * 100.0
def sensitivity(metrics):
    return metrics[0] / (metrics[0] + metrics[3])
def specificity(metrics):
    return metrics[1] / (metrics[1] + metrics[2]) * 100.0
def ppv(metrics):
    return metrics[0] / (metrics[0] + metrics[1]) * 100.0
def npv(metrics):
    return metrics[1] / (metrics[1] + metrics[3]) * 100.0

def main():
    trainSet = []
    testSet = []
    split = 0.80
    hRList = []
    sensList = []
    specList = []
    npvList = []
    ppvList = []
    splitDataset('demo.txt', split, trainSet, testSet)
    print('Train: ' + repr(len(trainSet)))
    print('Test: ' + repr(len(testSet)))
    
    # This number has to be large enough where every test value will have at least one
    # neighbor in the range, otherwise I get funky errors that I'm not dealing with at the moment
    radius = np.linspace(1, 10, num=10)
    # preds = []
    for r in radius:
        preds = []
        for x in range(len(testSet)):
            neighbors = findNeighbors(trainSet, testSet[x], r)
            # print(neighbors)
            if neighbors == []:
                results = 0.0
            else:
                results = getResponse(neighbors)
            # print(results)
            preds.append(results)
            # print("\n\n")
        metrics = accuracy(testSet, preds)
        hRList.append(hitRate(metrics))
        sensList.append(sensitivity(metrics))
        specList.append(specificity(metrics))
        ppvList.append(ppv(metrics))
        npvList.append(npv(metrics))
        # createBoundary(testSet, trainSet, r)

    # print(hRList, sensList, specList, ppvList, npvList)

    meanlist = [np.mean(hRList), np.mean(sensList), np.mean(specList), np.mean(ppvList), np.mean(npvList)]
    stdevlist = [np.std(hRList), np.std(sensList), np.std(specList), np.std(ppvList), np.std(npvList)]
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[0]]
    radius = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    hRList.append(meanlist[0])
    radius.append(11)
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[0]]
    plt.errorbar(radius, hRList, e, fmt='.k')
    plt.title('Hit Rate versus Radius')
    plt.show()

    sensList.append(meanlist[1])
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[1]]
    plt.errorbar(radius, sensList, e, fmt='.k')
    plt.title('Sensitivity versus Radius')
    plt.show()

    specList.append(meanlist[2])
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[2]]
    plt.errorbar(radius, specList, e, fmt='.k')
    plt.title('Specificity versus Radius')
    plt.show()

    ppvList.append(meanlist[3])
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[3]]
    plt.errorbar(radius, ppvList, e, fmt='.k')
    plt.title('PPV versus Radius')
    plt.show()

    npvList.append(meanlist[4])
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stdevlist[4]]
    plt.errorbar(radius, npvList, e, fmt='.k')
    plt.title('NPV versus Radius')
    plt.show()

    # x = [1, 2, 3, 4, 5]
    # y = meanlist
    # e = stdevlist
    # plt.errorbar(x, y, yerr=e, fmt='o')
    # plt.title('Mean and One S.D. for Hit Rate, Sensitivity, Specificity, PPV, and NPV')
    # plt.show()

    print(meanlist)
    print(stdevlist)

def createBoundary(testingSet, trainingSet, radius):
    x = max([t[0] for t in testingSet])
    y = max([t[1] for t in testingSet])
    # print(x, y)
    wifeStress = [t[0] for t in testingSet if t[-1] == 1.0]
    husbandStress = [t[1] for t in testingSet if t[-1] == 1.0]
    wifeNot = [t[0] for t in testingSet if t[-1] == 0.0]
    husbandNot = [t[1] for t in testingSet if t[-1] == 0.0]

    points=100
    xRange = np.linspace(0, x+1, num=points)
    yRange = np.linspace(0, y+1, num=points)
    decisionBoundary = []

    stressedX = []
    stressedY = []
    notStressedX = []
    notStressedY = []
    radius = 1.5
    for x in xRange:
        for y in yRange:
            testInst = [x, y, 0] # Have to add the zero, to keep the size the same [wife, husband, stressed/not]
            neighbors = findNeighbors(trainingSet, testInst, radius)
            # print(neighbors)
            results = getResponse(neighbors)

            if results == 1.0:
                stressedX.append(x)
                stressedY.append(y)
            else: #if results == 0.0:
                notStressedX.append(x)
                notStressedY.append(y)
            # print(results)
            decisionBoundary.append([x, y, results])
    # print(decisionBoundary)
    ds = plt.scatter(stressedX, stressedY, c='green')
    dn = plt.scatter(notStressedX, notStressedY, c='orange')
    s = plt.scatter(wifeStress, husbandStress, marker='x', c='blue', label='Stressed')
    n = plt.scatter(wifeNot, husbandNot, marker='x', c='red', label='Stressed')
    plt.legend((s, n, ds, dn), ('Stressed', 'Not Stressed', 'Stressed Boundary', 'Not Stressed Boundary'))
    plt.xlabel('Wifes Earnings')
    plt.ylabel('Husbands Earnings')
    plt.title('Decision Boundary of Stressed Classifier')
    plt.show()
    plt.savefig('graph.png')
    print(decisionBoundary)

main()