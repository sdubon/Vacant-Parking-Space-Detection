import cv2
import numpy as np
import glob

class StatModel(object):
    def load(self, fn):
        self.model.load(fn) 
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5, kernel = cv2.ml.SVM_RBF):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(kernel)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

    #def train_auto(self, samples, responses):
    #    params = dict( kernel_type = cv2.ml.SVM_RBF, svm_type = cv2.ml.SVM_C_SVC )
    #    self.model.trainAuto(samples,responses,None,None,params)

# recovering all samples
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
        labels = np.float32(np.append(labels, foo['labels']))
        samples_a = np.float32(np.append(samples_a, foo['samples_a'], axis = 0))
        samples_b = np.float32(np.append(samples_b, foo['samples_b'], axis = 0))
        samples_H = np.float32(np.append(samples_H, foo['samples_H'], axis = 0))
        samples_S = np.float32(np.append(samples_S, foo['samples_S'], axis = 0))
        samples_f = np.float32(np.append(samples_f, foo['samples_f']))
        samples_HS = np.float32(np.append(samples_HS, foo['samples_HS'], axis = 0))
    
labels=labels.astype(int)

if __name__ == '__main__':
    print('training SVM...')
    model_a = SVM(C=2.67, gamma=5.383, kernel = cv2.ml.SVM_LINEAR)
    model_a.train(samples_a, labels)
    print('saving SVM as "channel_a_svm.dat"...')
    model_a.save('channel_a_svm.dat')
    model_b = SVM(C=2.67, gamma=5.383, kernel = cv2.ml.SVM_LINEAR)
    model_b.train(samples_b, labels)
    print('saving SVM as "channel_b_svm.dat"...')
    model_b.save('channel_b_svm.dat')
    model_H = SVM(C=2.67, gamma=5.383, kernel = cv2.ml.SVM_LINEAR)
    model_H.train(samples_H, labels)
    print('saving SVM as "channel_H_svm.dat"...')
    model_H.save('channel_H_svm.dat')
    model_S = SVM(C=2.67, gamma=5.383, kernel = cv2.ml.SVM_LINEAR)
    model_S.train(samples_S, labels)
    print('saving SVM as "channel_S_svm.dat"...')
    model_S.save('channel_S_svm.dat')
    model_f = SVM(C=2.67, gamma=5.383, kernel = cv2.ml.SVM_LINEAR)
    model_f.train(samples_f, labels)
    print('saving SVM as "fast_svm.dat"...')
    model_f.save('fast_svm.dat')

    model_HS = SVM(C=1, gamma=5, kernel = cv2.ml.SVM_LINEAR)
    model_HS.train(samples_HS, labels)
    print('saving SVM as "channel_HS_svm.dat"...')
    model_HS.save('channel_HS_svm.dat')

    cv2.waitKey(0)
