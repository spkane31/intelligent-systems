import numpy as np
import matplotlib.pyplot as plt
import random
# # X = (hours sleeping, hours studying), y = score on test
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([92, 5], [86, 5], [89, 8]), dtype=float)

# # scale units
# X = X/np.amax(X, axis=0) # maximum of X array
# y = y/100 # max test score is 100

X = np.loadtxt('npImages.txt')
Y = np.loadtxt('npLabels.txt')

XTrain = []
YTrain = []
XTest = []
YTest = []
for i in range(len(X)):
    if (random.randint(0, 100) < 80):
        XTrain.append(X[i])
        YTrain.append(Y[i])
    else:
        XTest.append(X[i])
        YTest.append(Y[i])


#  y = np.asarray(self.hitRate)
XTrain = np.asarray(XTrain)
YTrain = XTrain
XTest = np.asarray(XTest)
YTest = YTrain



class Neural_Network(object):
    def __init__(self, inputSize, hiddenSize, outputSize, learningRate, epochs=500):#hiddenSize2, outputSize, learningRate):
        #parameters
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        # self.hiddenSize2 = hiddenSize2
        self.outputSize = outputSize
        self.learningRate = learningRate
        self.confMatrix = np.zeros((10,10))
        self.epochs = 0
        # Get and save error rate (1 - hit rate) every ten epochs, then graph it later
        self.hitRate = []

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
    def forward(self, X):
        #forward propagation through our network
        self.z1 = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        self.a1 = self.sigmoid(self.z1) # activation function

        self.z2 = np.dot(self.a1, self.W2)
        o = self.sigmoid(self.z2) # final activation function
        self.identifyDigit(o)
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

        self.W1 += self.learningRate * X.T.dot(self.z2_delta)
        self.W2 += self.learningRate * self.a1.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
        # self.accuracy(y, o)

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
        np.save('weights1AE', self.W1)
        np.save('weights2AE', self.W2)
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
        # acc = self.accuracy(y, o)
        # self.hitRate.pop()
        # print("Testing Accuracy: " + str(acc))


inputs = 784
hidden1 = 50
output = 784
learningRate = 0.01
epochs = 10

NN = Neural_Network(inputs, hidden1, output, learningRate, epochs)

# result = input('Do you want to load previous data? (y/n)')
# if result == 'y':
#     NN.loadNN()
# else:
#     print("COOL")

for i in range(epochs): 
    print("Epoch Number: " + str(NN.epochs))
    Loss = np.square(YTrain - NN.forward(XTrain))
    Loss = np.sum(Loss)/2
    print("Loss: \t" + str(Loss)) # mean sum squared loss
    NN.train(XTrain, YTrain)
    NN.epochs += 1
    print("\n")

# Run the testing data
NN.test(XTest, YTest)

# Option to save the data
# result = input('Do you want to save the data? (y/n)')
# if result == 'y':
#     NN.saveNN()
# else:
#     print("Goodbye")