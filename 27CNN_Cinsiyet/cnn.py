#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# ilkleme
classifier = Sequential()

# Adım 1 - Convolution
classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Adım 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 2. convolution katmanı
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adım 3 - Flattening
classifier.add(Flatten())

# Adım 4 - YSA
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# CNN ve resimler

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=len(training_set),
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=len(test_set))

# Evaluation
scores = classifier.evaluate_generator(test_set, steps=len(test_set))
print("Accuracy: %.2f%%" % (scores[1]*100))



