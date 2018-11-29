import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import expit
import random
import math
from datetime import datetime

# During training, calculate value of loss on the training set at the beginning and after
# every tenth epoch and save it. Training should continue until loss funciton on the training 
# set is sufficiently low. No confusion matrix. After training, each hidden neuron has become 
# tuned to a specific 28x28 (784 pixels) feature. Plot these with a 28x28 grayscale image.

# Report performance on the training set and the test set using the loss function. Plot these
# two values side by side. Plot the same error in the same way but for each digit. There will 
# be two bars for 0, two for 1, etc. Plot the time series of the overall training error 
# during training using data saved at every tenth epoch. Plot images for large number (if possible
# all) features. 

X = np.loadtxt('npImages.txt')
Y = np.loadtxt('npLabels.txt')


Xlength = 784       # Length of X Arrays
Ylength = 10        # Length of Y Arrays


XTrain = []
YTrain = []
YTrainValues = []

XTest = []
YTest = []
YTestValues = []
XTrainSize = 0
# Randomly selects 80 % of data to be used in training
for i in range(len(X)):
    if (random.randint(0, 100) < 80):
        XTrain.append(X[i])
        YTrain.append(Y[i])
        count = 0
        for i in Y[i]:
            if i == 1:
                YTrainValues.append(count)
            count += 1
        XTrainSize += 1
    else:
        XTest.append(X[i])
        YTest.append(Y[i])
        count = 0
        for i in Y[i]:
            if i == 1:
                YTestValues.append(count)
            count += 1

#  y = np.asarray(self.hitRate)
XTrain = np.asarray(XTrain, dtype=np.float128)
YTrain = np.asarray(YTrain, dtype=np.float128)
XTest = np.asarray(XTest, dtype=np.float128)
YTest = np.asarray(YTest, dtype=np.float128)

print("Dataset Loaded!")

def shuffleData(i, o, num=1):
    # We will do this one by one for this problem
    count = 0
    X = []
    Y = []
    if len(i) != len(o):
        return -1
    while count < num:
        ranNum = random.randint(0, len(XTrain)-1)
        X.append(XTrain[ranNum])
        Y.append(YTrain[ranNum])
        count += 1
    count = 0
    for i in Y[0]:
        if i == 1:
            break
        else:
            count += 1    
    return np.asarray(X), np.asarray(Y), count

class Neural_Network(object):
    def __init__(self, inputSize, hiddenSize, outputSize, learningRate, epochs=500):#hiddenSize2, outputSize, learningRate):
        #parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        # self.hiddenSize2 = hiddenSize2
        self.outputSize = outputSize
        self.learningRate = learningRate
        
        self.rhoHat = 0.0                         # Average Activation: This is just initally, has to be changed after a full pass through
        self.rho = 0                            # That inital inputs activation of the hidden neuron

        self.alpha = 0.5                        # Used with momentum 
        self.alpha1 = 0
        self.alpha2 = 0

        self.beta = 2                           # Used w/ sparseness
        self.lambdaValue = 0.0005               # Used for regularization - weight decay
        self.epochs = 0
        self.avgActivation = np.zeros(shape=(1,150))
        # Get and save error rate (1 - hit rate) every ten epochs, then graph it later
        self.hitRate = []

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) * np.sqrt(1/(self.inputSize+self.hiddenSize)) # weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) * np.sqrt(1/(self.hiddenSize+self.outputSize))

    def forward(self, X):
        #forward propagation through our network
        self.z1 = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.a1 = self.sigmoid(self.z1) # activation function
        # self.avgActivation += self.a1

        self.z2 = np.dot(self.a1, self.W2)
        o = self.sigmoid(self.z2) # final activation function
        return o 

    def activation(self, X):
        NN.forward(X)
        rho = self.a1
        return rho

    def KLdivergence(self, X):
        try:
            rho = sum(sum((self.activation(X))))
            rho = float(rho)
        except:
            rho = sum(self.activation(X))
            rho = float(rho)
        
        try:
            a = rho * math.log(rho/self.rhoHat, 10)
        except:
            a = self.rhoHat
        try:
            b = (1 - rho) * math.log((1-rho)/(1-self.rhoHat), 10)
        except:
            b = a
        return a + b

    def weightDecay(self, X):
        return sum(sum(np.square(self.W1))) + sum(sum(np.square(self.W2)))

    def sigmoid(self, s):
        # try:
        s = np.array(s, dtype=np.float128)
        return 1.0 / (1.0 + np.exp(-s))
        # except:
        #     return 1.0 / (1.0 + expit(-s))

    def sigmoidPrime(self, s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.a1) # applying derivative of sigmoid to z2 error

        rhoFunc = (1-self.activation(X) / (1 - self.rhoHat)) - (self.activation(X)/self.rhoHat)

        # self.z2_delta += self.beta * rhoFunc 
        
        dW1 = self.learningRate * X.T.dot(self.z2_delta) #- self.lambdaValue * self.W1
        dW2 = self.learningRate * self.a1.T.dot(self.o_delta) #- self.lambdaValue * self.W2

        # Here is where the momentum changes, ie weights only change after the first epoch
        if epochs > 1: # Adjust weights w/ momentum 
            self.W1 += dW1 + (self.alpha * self.alpha1) 
            self.W2 += dW2 + (self.alpha * self.alpha2)

        self.alpha1 = dW1
        self.alpha2 = dW2

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        # self.accuracy(y, o)

    def calculateError(self, X, Y):
        # Error function is difference squared, summed over all of the data set
        loss = np.square(X-Y)
        loss = np.sum(loss) * (1.0/2.0)

        loss += self.beta * self.KLdivergence(X)

        # loss += self.lambdaValue/2 * self.weightDecay(X)
        return loss
        # Finish this

    def identifyDigit (self, o):
        npOutput = np.zeros(shape=len(o))
        index = 0
        for i in o:
            maxVal = np.amax(i)
            count = 0
            for a in i:
                if a == maxVal:
                    npOutput[index] = count

                count += 1
            index += 1
        return npOutput

    def accuracy(self, Y, Yhat):
        # Gets the accuracy of the output, compares Yhat to Y, returns a number >= 1.00
        Y = self.identifyDigit(Y)
        Yhat = self.identifyDigit(Yhat)
        acc = self.compareArray(Y, Yhat)

        if self.epochs % 10 == 0:
            self.hitRate.append(1-self.compareArray(Y, Yhat))
            print(self.hitRate)
        return acc
        
    def compareArray(self, arr1, arr2):
        # Returns accuracy for one epoch
        # This can also be used to determine the hit rate 
        length = len(arr1)
        count = 0
        overallAcc = 0
        for i in range(length):
            if arr1[i] == arr2[i]:
                count += 1
        
        return count/length

    def saveNN(self):
        np.savetxt('weights1AE.txt', self.W1)
        np.savetxt('weights2AE.txt', self.W2)

    def loadNN(self):
        try:
            self.W1 = np.load('weights1AE.npy')
            self.W2 = np.load('weights2AE.npy')
        except:
            print("none found")

    def test(self, X, y):
        o = self.forward(X)
        return o

    def initializeRhoHat(self, XTrain):
        z = np.dot(XTrain, self.W1)
        a = self.sigmoid(z)
        rhohat = float(sum(sum(a))/len(XTrain))
        if self.rhoHat < .15 and self.rhoHat > 0.05:
            self.rhoHat = self.rhoHat
        else:
            self.rhoHat = rhohat

    def loss(self, X, Y):
        loss = np.square(X-Y)
        loss = np.sum(loss) * (1.0/2.0)

        sumSet = 0
        for i in range(len(X)):
            sumSet += self.KLdivergence(X[i])
        loss += self.beta * sumSet

        # loss += self.lambdaValue/2 * self.weightDecay(X)
        return abs(loss)

def graphErrorRate(hitRate, epochs):
    x = np.linspace(0, epochs, num=len(hitRate))
    y = np.asarray(hitRate)
    plt.figure(2)
    plt.plot(x, y, 'o', color='black')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error Rate')
    plt.title('Error Rate Over Time')
    plt.savefig('AEErrorRate.png')

def graphRhoHat(rhoHat, epochs):
    x = np.linspace(0, epochs, num=len(rhoHat))
    # y = np.asarray(rhoHat)
    y = rhoHat
    print(y)
    plt.figure(3)
    plt.plot(x, y, 'o', color='black')
    plt.xlabel('Epoch Number')
    plt.ylabel('RhoHat')
    plt.title('Rho Hat Over Time')
    plt.savefig('RhoHat.png')

def plotErrorPerNumber(training, test):
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y1 = training
    y2 = test
    ax = plt.subplot(111)
    ax.bar(x-0.1, y1, width=0.2, color='b', align='center')
    ax.bar(x+0.1, y2, width=0.2, color='g', align='center')
    plt.xlim(-0.5, 9.5)
    plt.gca().legend(('Training','Test'))
    plt.xlabel('Label of MNIST Data')
    plt.ylabel('Average Error')
    plt.title('Error for Training and Testing Set Per Number')
    plt.figure(1)
    plt.savefig('AutoEncoderError.png')
    plt.show()


inputs = 784
hidden1 = 150
output = 784
learningRate = 0.05
epochs = 1000

NN = Neural_Network(inputs, hidden1, output, learningRate, epochs)
NN.initializeRhoHat(XTrain)

# Average loss is a 2 x 10 array, the first ten hold the cumulative loss of that index's digit
# the second row holds the number of times a training value was equal to that number.
# At the end, a cell by cell division will give the average loss of the autoencoder
averageLoss = np.zeros(shape=(2,10), dtype=float)

# result = input('Do you want to load previous data? (y/n)')
# if result == 'y':
#     NN.loadNN()
# else:
#     print("COOL")
lossOverTime = np.zeros(shape=(epochs//10))
rhoHatOverTime = np.zeros(shape=(epochs//10))

# Training and testing stuff
while NN.epochs < epochs: # and percentChange > 0.01:
    startTime = datetime.now()
    NN.initializeRhoHat(XTrain)
    print("\nEpoch #" + str(NN.epochs))
    print("Loss: \t\t" + str(NN.loss(XTrain, NN.forward(XTrain))))
    print("RhoHat: \t%f" % NN.rhoHat)
    # for i in range(len(XTrain)):
    #     Xmini = []
    #     Ymini = []
    #     Xmini.append(XTrain[i])
    #     Ymini.append(YTrain[i])
    #     Xmini = np.asarray(Xmini)
    #     Ymini = np.asarray(Ymini)
    #     Y = YTrainValues[i]
    #     # print("Epoch Number: " + str(NN.epochs))
    #     # Calculate, store, and print loss on most recent training point
    #     currentLoss = NN.calculateError(Xmini, NN.forward(Xmini))
    #     averageLoss[0][Y] += currentLoss
    #     averageLoss[1][Y] += 1
    #     # print(averageLoss)
    #     # print(NN.calculateError(Xmini, NN.forward(Xmini)))
    #     # print("Loss: \t" + str(currentLoss)) 
    #     NN.train(Xmini, Xmini)
    NN.train(XTrain, XTrain)

    if NN.epochs % 10 == 0:
        lossOverTime[NN.epochs//10] = float(NN.loss(XTrain, NN.forward(XTrain)))
        rhoHatOverTime[NN.epochs//10] = float(NN.rhoHat)
        print("Loss list: " + str(lossOverTime))
        print("rhoHat List: " + str(rhoHatOverTime))
        # print("Epoch Number: " + str(NN.epochs))
        # print("Loss: \t\t" + str(currentLoss)) 
    
     
    print("Epoch %s took %s seconds.\n" % (NN.epochs, datetime.now()-startTime))
    NN.epochs += 1



testingLoss = np.zeros(shape=(2,10), dtype=float)
# Run and record the testing data
for i in range(len(XTest)):
    y = 0
    for j in YTest[i]:
        if j == 1:
            break
        else:
            y += 1
            # NN.calculateError(Xmini, NN.forward(Xmini))
    loss = NN.calculateError(XTest[i], NN.forward(XTest[i])) #np.mean(np.square(XTest[i] - NN.forward(XTest[i])))
    testingLoss[0][y] += loss
    testingLoss[1][y] += 1

testLoss = np.zeros(shape=(10), dtype=float)
trainLoss = np.zeros(shape=(10), dtype=float)
for i in range(10):
    testLoss[i] = testingLoss[0][i]/testingLoss[1][i]
    trainLoss[i] = averageLoss[0][i]/averageLoss[1][i]

# plotErrorPerNumber(trainLoss, testLoss)
graphErrorRate(lossOverTime, epochs)
graphRhoHat(rhoHatOverTime, epochs)


# Option to save the data
# result = input('Do you want to save the data? (y/n)')
# if result == 'y':
#     NN.saveNN()
# else:
#     print("Goodbye")