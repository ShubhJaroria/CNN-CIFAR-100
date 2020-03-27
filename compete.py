from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.utils import to_categorical
import sys


#opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)
trainfile = sys.argv[1]#"/content/drive/My Drive/Google HPC/train.csv"
testfile = sys.argv[2]#"/content/drive/My Drive/Google HPC/test.csv"
outputfile = sys.argv[3]#"/content/drive/My Drive/Google HPC/predictions.txt"

#print("loading ur data daddy")
psd_trainX = np.loadtxt(trainfile,delimiter=" ")
np.random.shuffle(psd_trainX)
n_train = psd_trainX.shape[0]
m = psd_trainX.shape[1]
train_x = psd_trainX[:,0:m-2]
train_y = psd_trainX[:,m-1]
train_y = train_y.astype(int)
n_features = train_x.shape[1]
train_y = train_y.reshape((n_train,1))
#print(train_y[0:10])
#print(train_y.shape)


psd_testX = np.loadtxt(testfile,delimiter=" ")
n_test = psd_testX.shape[0]
m = psd_testX.shape[1]
test_x = psd_testX[:,0:m-2]
test_y = psd_testX[:,m-1]
test_y = test_y.astype(int)
n_features = test_x.shape[1]
test_y = test_y.reshape((n_test,1))
#Preprocessing
train_x = train_x.reshape((train_x.shape[0], 3, -1))
test_x = test_x.reshape((test_x.shape[0], 3, -1))
#val_x = val_x.reshape((test_x.shape[0], 3, -1))

train_x = train_x/255.0
test_x = test_x/255.0
#val_x = val_x/255.0

train_x = np.concatenate([train_x[:, i, None].reshape(train_x.shape[0], -1, 1) for i in range(3)], axis=2)
test_x = np.concatenate([test_x[:, i, None].reshape(test_x.shape[0], -1, 1) for i in range(3)], axis=2)
#val_x = np.concatenate([val_x[:, i, None].reshape(val_x.shape[0], -1, 1) for i in range(3)], axis=2)

train_x = train_x.reshape(-1,32,32,3)
test_x = test_x.reshape(-1,32,32,3)


#print(train_x.shape,test_x.shape)

train_x = train_x.reshape(-1, 32, 32, 3)
test_x = test_x.reshape(-1, 32, 32, 3)
#val_x = val_x.reshape(-1, 32, 32, 3)


train_y = to_categorical(train_y)
#print(train_y.shape)
#test_y = to_categorical(test_y)
batch_size = 128
maxepoches = 1
learning_rate = 0.1
lr_decay = 1e-6
lr_drop = 20

model = Sequential()
weight_decay = 0.0005

model.add(Conv2D(64, (3, 3), padding='same',
                 input_shape=(32,32,3),kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))


#training parameters
def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))
reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_x)

#optimization details
sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


# training process in a for loop with learning rate drop every 25 epoches.

historytemp = model.fit_generator(datagen.flow(train_x, train_y,
                                 batch_size=batch_size),
                    steps_per_epoch=train_x.shape[0] // batch_size,
                    epochs=maxepoches,
                    callbacks=[reduce_lr],verbose=2)        


def convert(train_y):
    target = np.zeros((len(train_y),1), dtype = int)
    for i in range(len(train_y)):
        target[i][0] = int(np.argmax(train_y[i]))
    return target


predictions = model.predict(test_x)
#test_label_ids = test_y
#accuracy=accuracy_score(predictions , test_label_ids)
#print(accuracy)
Y_hat = convert(predictions)
np.savetxt(outputfile,Y_hat,delimiter = '\n',fmt="%s")

