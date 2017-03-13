import numpy as np
import glob

labels = []
samples_a = np.empty((0,32), float)
samples_b = np.empty((0,32), float)
samples_H = np.empty((0,32), float)
samples_S = np.empty((0,32), float)
samples_f = []
samples_HS = np.empty((0,64), float)

for filename in glob.glob('*.npz'):
    if filename != 'coordinates.npz':
        foo = np.load(filename)
        labels = np.append(labels, foo['labels'])
        samples_a = np.append(samples_a, foo['samples_a'], axis = 0)
        samples_b = np.append(samples_b, foo['samples_b'], axis = 0)
        samples_H = np.append(samples_H, foo['samples_H'], axis = 0)
        samples_S = np.append(samples_S, foo['samples_S'], axis = 0)
        samples_f = np.append(samples_f, foo['samples_f'])
        samples_HS = np.append(samples_HS, foo['samples_HS'], axis = 0)

print labels
print samples_a
print samples_b
print samples_H
print samples_S
print samples_f
print samples_HS
