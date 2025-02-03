import matplotlib.pyplot as plt
import numpy as np
import csv

// Load data
with open('sphere.txt') as file:
    data = np.loadtxt('sphere.txt')

m, n = data.shape

// Plot data
x = range(0, m, n)
fig, ax = plt.subplots(1, figsize=(8,8))
g = ax.imshow(data[0:n], cmap='gray')
plt.savefig('sphere.png')