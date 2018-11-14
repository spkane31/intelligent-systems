import numpy as np
import matplotlib.pyplot as plt

W1 = np.loadtxt('weights1AE.txt')

temp = np.empty((150, 784))
for i in range(150):
    temp[i] = W1[:,i]

grayscale = np.empty((150,28,28))

for j in range(150):
    for i in range(784):
        grayscale[j][i // 28][i % 28] = temp[j][i]

fig, axs = plt.subplots(10, 15, sharex=True, sharey=True)

for i in range(10):
    for j in range(15):
        axs[i, j].imshow(grayscale[i*2 + j], cmap="gray")
plt.savefig('features.png')
plt.show()