import numpy as np
import matplotlib.pyplot as plt
import random

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
XTest = []
YTest = []
# Randomly selects 80 % of data to be used in training
for i in range(len(X)):
    if (random.randint(0, 100) < 80):
        XTrain.append(X[i])
        YTrain.append(Y[i])
    else:
        XTest.append(X[i])
        YTest.append(Y[i])


#  y = np.asarray(self.hitRate)
XTrain = np.asarray(XTrain)
YTrain = np.asarray(YTrain)
XTest = np.asarray(XTest)
YTest = np.asarray(YTest)

# print()

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
        # print(ranNum)
        # print(X.shape)
        # print(XTrain[0].shape)
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

        self.alpha = 0.5                        # Used with momentum 
        self.alpha1 = 0
        self.alpha2 = 0

        # self.confMatrix = np.zeros((10,10))
        self.epochs = 0
        # Get and save error rate (1 - hit rate) every ten epochs, then graph it later
        self.hitRate = []

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) * np.sqrt(1/(self.inputSize+self.hiddenSize)) # weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) * np.sqrt(1/(self.hiddenSize+self.outputSize))

    def forward(self, X):
        #forward propagation through our network
        self.z1 = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.a1 = self.sigmoid(self.z1) # activation function

        self.z2 = np.dot(self.a1, self.W2)
        o = self.sigmoid(self.z2) # final activation function
        # self.identifyDigit(o)
        return o 

    def sigmoid(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.a1) # applying derivative of sigmoid to z2 error
        
        dW1 = self.learningRate * X.T.dot(self.z2_delta)
        dW2 = self.learningRate * self.a1.T.dot(self.o_delta)

        # Here is where the momentum changes, ie weights only change after the first epoch
        if epochs > 1: # Adjust weights w/ momentum 
            self.W1 += dW1 + self.alpha * self.alpha1
            self.W2 += dW2 + self.alpha * self.alpha2 # adjusting second set (hidden --> output) weights

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
        # print("Accuracy: ", str(acc))
        # overallAcc = 0
        # for i in range(10):
        #     overallAcc += self.confMatrix[i][i]
        # print("Overall Accuracy: " + str(overallAcc/sum(sum(self.confMatrix))))
        if self.epochs % 10 == 0:
            self.hitRate.append(1-self.compareArray(Y, Yhat))
            print(self.hitRate)
        # Creates Confusion Matrix
        # self.confusionMatrix(Y, Yhat)
        # print("Confusion Matrix: \n", str(self.confMatrix))
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
        # save W1, W2, confusionMatrix
        np.savetxt('weights1AE.txt', self.W1)
        np.savetxt('weights2AE.txt', self.W2)
        # np.save('ConfusionMatrix', self.confMatrix)

    def loadNN(self):
        try:
            self.W1 = np.load('weights1AE.npy')
            self.W2 = np.load('weights2AE.npy')
            # self.confMatrix = np.load('ConfusionMatrix.npy')
        except:
            print("none found")

    def test(self, X, y):
        o = self.forward(X)
        return o
        # acc = self.accuracy(y, o)
        # self.hitRate.pop()
        # print("Testing Accuracy: " + str(acc))

def graphErrorRate(hitRate, epochs):
    x = np.linspace(0, epochs, num=len(hitRate))
    y = np.asarray(hitRate)
    plt.figure(2)
    plt.plot(x, y, 'o', color='black')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error Rate')
    plt.title('Error Rate Over Time')
    plt.savefig('AEErrorRate.png')

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
learningRate = 0.01
epochs = 1000

NN = Neural_Network(inputs, hidden1, output, learningRate, epochs)

percentChange = 1
prevLoss = 1

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

while NN.epochs < epochs: # and percentChange > 0.01:
    # Create new shuffled data points
    Xmini, Ymini, Y = shuffleData(XTrain, YTrain)
    print("Epoch Number: " + str(NN.epochs))

    # Calculate, store, and print loss on most recent training point
    currentLoss = NN.calculateError(Xmini, NN.forward(Xmini)) #np.mean(np.square(Xmini - NN.forward(Xmini)))
    averageLoss[0][Y] += currentLoss
    averageLoss[1][Y] += 1
    # print(NN.calculateError(Xmini, NN.forward(Xmini)))
    print("Loss: \t" + str(currentLoss)) 

    # Save Loss every tenth epoch
    if NN.epochs % 10 == 0:
        lossOverTime[NN.epochs//10] = currentLoss
        print("Loss list: " + str(lossOverTime))

    # Train the data using X as both input and desired output
    NN.train(XTrain, XTrain)

    # Calculate percent change for stopping conditions
    # percentChange = abs(1 - (currentLoss/prevLoss) * 100)
    # print("Percent Change: " + str(percentChange))
    # currentLoss = prevLoss

    # Incremement epochs and continue on
    NN.epochs += 1
    print("\n")

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

plotErrorPerNumber(trainLoss, testLoss)
graphErrorRate(lossOverTime, epochs)


# Option to save the data
result = input('Do you want to save the data? (y/n)')
if result == 'y':
    NN.saveNN()
else:
    print("Goodbye")