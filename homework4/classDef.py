import numpy as np
# https://github.com/jldbc/numpy_neural_net/blob/master/four_layer_network.py
# X = (hours sleeping, hours studying), y = score on test
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([92], [86], [89]), dtype=float)

# # scale units
# X = X/np.amax(X, axis=0) # maximum of X array
# y = y/100 # max test score is 100

X = np.loadtxt('npImages.txt', dtype=float)
Y = np.loadtxt('npLabels.txt', dtype=int)
# print("Length of output: ", str(len(y)))

class Neural_Network(object):
    def __init__(self, inputSize = 784, hiddenSize1 = 100, hiddenSize2 = 50, outputSize = 10, learningRate = 0.1):
        #parameters
        self.inputSize = inputSize              # 784 neurons
        self.outputSize = outputSize            # 100 neurons
        self.hiddenSize = hiddenSize1           # 50 neurons
        self.hiddenSize2 = hiddenSize2          # 10 neurons
        self.l = learningRate

        #weights
        # Weights from input -> Hidden Layer 1
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)      # 784 x 100 matrix
        # weights from Hidden Layer 1 -> Hidden Layer 2
        self.W2 = np.random.randn(self.hiddenSize, self.hiddenSize2)    # 100 x 50 matrix
        # Weights from Hidden Layer 2 -> Output
        self.W3 = np.random.randn(self.hiddenSize2, self.outputSize)    # 50 x 10 matrix
        # print(self.W1.shape)
        # print(self.W2.shape)
        # print(self.W3.shape)


    def forward(self, X): # Think this is all good
        #forward propagation through network

        # input -> hidden layer 1
        self.z1 = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
        # hidden layer 1 value
        # self.z2 = self.activation(self.z) # activation function
        self.a1 = self.activation(self.z1)

        # hidden layer 1 -> hidden layer 2
        self.z2 = np.dot(self.a1, self.W2) # dot product of layer1 and second set of weights
        # hidden layer 2 value
        # self.z4 = self.activation(self.z3) # activation function
        self.a2 = self.activation(self.z2)

        # hidden layer 2 -> output
        self.z3 = np.dot(self.a2, self.W3) # dot product of layer2 and third set of weights
        # output value
        # o = self.activation(self.z5)
        self.output = self.activation(self.z3)
        # print(type(o))
        return self.findResult(self.output)

    def activation(self, s):
        # activation function 
        return 1/(1+np.exp(-s))

    def activationPrime(self, s):
        #derivative of activation
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        # create np array for y that has 1 in index that is correct, 0 elsewhere
        npY = np.zeros(shape=10)
        npY[y] = 1
        delta3 = o
        delta3[range(X.shape[0]), y] -= 1
        print(delta3)
        self.dW3 = (self.z2.T).dot(delta3)
        # print(self.dW3)
        




        # # find error in output
        # self.o_error = npY - o # error in output, yhat - y
        # self.o_delta = self.o_error*self.activationPrime(o) # applying derivative of activation to error
        # # print(self.W1)
        # self.dW3 = (self.z5).dot(self.o_delta)
        # # self.W3 += self.z4.T.dot(self.o_delta) 
        # print("SHAPE: ", str(self.dW3.shape))
        # print(self.dW3)
        # print("MADE IT")
        # # LEFT OFF HERE, W1 DOES NOT ADJUST HOW IT NEEDS TO

        # # find error of HL2
        # self.z4_error = self.o_delta.dot(self.W2.T) # z4 error: how much hidden layer 2 contributes to output error
        # self.z4_delta = self.z4_error * self.activationPrime(self.z4) # apply derivative of activation function
        # self.W2 += self.z2.T.dot(self.z4_delta)
        # print("MADE IT")

        # # find error of HL1
        # self.z2_error = self.z4_delta.dot(self.W1.T) # z2 error: how much our hidden layer weights contributed to output error
        # self.z2_delta = self.z2_error*self.activationPrime(self.z2) # applying derivative of activation to z2 error
        # print(self.z2_delta)
        # # print(X.T)
        # self.W1 += X.T.dot(self.z2_delta)
        # print("MADE IT")

        # self.W1 += X.T.dot(self.z2_delta) # adjusting input -> HL1 weights
        # self.W2 += self.z4.T.dot(self.z4_delta) # adjust HL1 -> HL2 weights
        # self.W3 += self.z2.T.dot(self.o_delta) # adjust HL2 -> output weights
    
    def train (self, x, y):
        # X is 784 x 1 input array
        # y is the digit that X should be identified as
        o = self.forward(x)
        print(o)
        self.backward(x, y, o)

  
    # Creates a numpy array with 1 where the index of the predicted ouput is, -1 elsewhere
    def findResult(self, output):
        maxVal = float(np.amax(output))
        npResult = np.zeros(shape=len(output))
        i = 0
        for o in output:
            if o == maxVal:
                npResult[i] = 1
            else:
                npResult[i] = -1
            i += 1
        return npResult

NN = Neural_Network()
# Loss = []
for x in X:#i in range(10): # trains the NN 1,000 times
    for y in Y:
        # NN.forward(x)
#   print("Input: \t" + str(X))
#   print("Actual Output: \t" + str(y))
        print("Predicted Output: \t" + str(NN.forward(x)))
    # Loss.append(np.mean(np.square(y-NN.forward(X))))
    # print("Loss: \t" + str(np.mean(np.square(y - NN.forward(X))))) # mean sum squared loss
#   print("Weights1: \t" +str(NN.W1))
        print("\n")
        NN.train(x, y)
# print(Loss)