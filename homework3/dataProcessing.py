import numpy as np

images = np.zeros(shape=(5000,10))

def processData():
    count = 0
    numProcessed = 0
    images = []
    file = open("MNISTnumImages5000.txt", "r")
    for line in file:
        if count > 10:
            break
        test = line
        count += 1


        arr = []
        nums = "0123456789."
        temp = ""
        count = 0
        for i in range(len(test)):
            if count > 0:
                count -= 1
            elif test[i] in nums:
                while test[i] in nums:
                    temp += test[i]
                    i += 1
                    count += 1
            try:
                if temp != '':
                    arr.append(float(temp))
                    temp = ''
            except:
                temp = ''
            if test[i] not in nums:
                temp = ''
            i = i + count

        images.append(arr)

        # comment this out to get the whole dataset
        # numProcessed += 1
        # if numProcessed > 200:
        #     break
        
    npImages = np.array(images)
    np.savetxt('npImages.txt', npImages, fmt='%2.5f')

def processLabels():
    # Takes a list of actual MNIST values and returns a numpy array
    # of shape (n, 10) with a 1 where the actual value is. n is the
    # number of MNIST values inputted.

    count = 0
    labels = [] #np.zeros(shape=(5000))
    file = open("MNISTnumLabels5000.txt", "r")
    for line in file:
        # print(len(line))
        labels.append(int(line))
        # labels[count] = int(line)
        count += 1
        
        # Get rid of this for whole dataset
        # if count > 200:
        #     break

    npLabels = np.zeros(shape=(len(labels), 10))
    count = 0
    for l in labels:
        npLabels[count][int(l)] = 1
        count += 1
    np.savetxt('npLabels.txt', npLabels, fmt='%d')

processLabels()


processData()
