import numpy as np
import matplotlib.pyplot as plt
import random
import csv

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

print("Dataset Loaded!")

def shuffleData(i, o, num):
    count = 0
    X = []#np.array((num, Xlength))
    Y = []#np.array((num, Ylength))
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
    return np.asarray(X), np.asarray(Y)



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

        self.confMatrix = np.zeros((10,10))
        self.epochs = 0
        # Get and save error rate (1 - hit rate) every ten epochs, then graph it later
        self.hitRate = []

        #weights
        # Weights are set with xavier's initialization: hackernoon acticle
        
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) * np.sqrt(1/(self.inputSize+self.hiddenSize)) # weight matrix from input to hidden layer
       
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) * np.sqrt(1/(self.hiddenSize+self.outputSize))
    
    # def setWeights(self):


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
        self.accuracy(y, o)

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
        print("Accuracy: ", str(acc))
        overallAcc = 0
        for i in range(10):
            overallAcc += self.confMatrix[i][i]
        print("Overall Accuracy: " + str(overallAcc/sum(sum(self.confMatrix))))
        if self.epochs % 10 == 0:
            self.hitRate.append(1-self.compareArray(Y, Yhat))
        # Creates Confusion Matrix
        self.confusionMatrix(Y, Yhat)
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

    def confusionMatrix(self, Y, Yhat):
        # Updates confusion matrix
        # Link to take numpy array and make it a table in tex document
        # https://tex.stackexchange.com/questions/54990/convert-numpy-array-into-tabular
        for i in range(len(Y)):
            x = int(Y[i])
            y = int(Yhat[i])
            self.confMatrix[y][x] += 1
            
    def graphErrorRate(self):
        x = np.linspace(0, self.epochs, num=len(self.hitRate))
        y = np.asarray(self.hitRate)
        plt.plot(x, y, 'o', color='black')
        plt.xlabel('Epoch Number')
        plt.ylabel('Error Rate')
        plt.title('Error Rate Over Time')
        plt.savefig('report/NNErrorRate.png')
    
    def saveNN(self):
        # save W1, W2, confusionMatrix
        np.save('weights1', self.W1)
        np.save('weights2', self.W2)
        np.save('ConfusionMatrix', self.confMatrix)

    def loadNN(self):
        try:
            self.W1 = np.load('weights1.npy')
            self.W2 = np.load('weights2.npy')
            self.confMatrix = np.load('ConfusionMatrix.npy')
        except:
            print("none found")

    def test(self, X, y):
        o = self.forward(X)
        acc = self.accuracy(y, o)
        self.hitRate.pop()
        print("Testing Accuracy: " + str(acc))
        return acc


inputs = 784
hidden1 = 150
output = 10
learningRate = 0.01
epochs = 800
miniBatchSize = 50

NN = Neural_Network(inputs, hidden1, output, learningRate, epochs)

percentChange = 1
prevLoss = 1
# Option to load previous data
result = input('Do you want to load previous data? (y/n)')
if result == 'y':
    NN.loadNN()

# Actual training of data
# for i in range(epochs): 
# i = 0
while percentChange > 0.01 and NN.epochs < epochs:
    Xmini, Ymini = shuffleData(XTrain, YTrain, miniBatchSize)
    print("Epoch Number: " + str(NN.epochs))
    currentLoss = np.mean(np.square(Ymini - NN.forward(Xmini)))
    # currentLoss = np.mean(np.square(YTrain - NN.forward(XTrain)))
    print("Loss: \t" + str(currentLoss)) # mean sum squared loss
    # NN.train(XTrain, YTrain)
    NN.train(Xmini, Ymini)
        
    percentChange = abs(1 - (currentLoss/prevLoss) * 100)
    print("Percent Change: " + str(percentChange))
    currentLoss = prevLoss

    print("\n")
    NN.epochs += 1


# Run the test data
testResults = NN.test(XTest, YTest)

# Create graph of the Error Rate
NN.graphErrorRate()

# Print Confusion Matrix
CM = NN.confMatrix/sum(sum(NN.confMatrix))
CM1 = np.around(CM, 5)
# print(CM1)
print(NN.confMatrix)
np.savetxt("report/confusionmatrix.csv", NN.confMatrix)

# Option to save the data
result = input('Do you want to save the data? (y/n)')
if result == 'y':
    NN.saveNN()
else:
print("Goodbye")