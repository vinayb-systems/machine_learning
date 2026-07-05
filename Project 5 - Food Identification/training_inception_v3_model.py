# Generated from: training_inception_v3_model.ipynb
# Converted at: 2026-07-05T17:37:57.395Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from google.colab import drive
drive.mount('/content/drive')

!nvidia-smi

# !cd /content/drive/My\ Drive/project2 && wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
# > /dev/null 2>&1
rm -r food-101

!tar -xvf /content/drive/My\ Drive/project2/food-101.tar.gz > /dev/null 2>&1

%%capture
import os
from collections import defaultdict
import shutil, sys 

if not os.path.isdir('/content/food-101/test') or not os.path.isdir('/content/food-101/train'):

    def copytree(src, dst, symlinks = False, ignore = None):
        if not os.path.exists(dst):
            os.makedirs(dst)
            shutil.copystat(src, dst)
        lst = os.listdir(src)
        if ignore:
            excl = ignore(src, lst)
            lst = [x for x in lst if x not in excl]
        for item in lst:
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if symlinks and os.path.islink(s):
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.readlink(s), d)
                try:
                    st = os.lstat(s)
                    mode = stat.S_IMODE(st.st_mode)
                    os.lchmod(d, mode)
                except:
                    pass # lchmod not available
            elif os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                shutil.copy2(s, d)

    def generate_dir_file_map(path):
        dir_files = defaultdict(list)
        with open(path, 'r') as txt:
            files = [l.strip() for l in txt.readlines()]
            for f in files:
                dir_name, id = f.split('/')
                dir_files[dir_name].append(id + '.jpg')
        return dir_files

    train_dir_files = generate_dir_file_map('/content/food-101/meta/train.txt')
    test_dir_files = generate_dir_file_map('/content/food-101/meta/test.txt')


    def ignore_train(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = train_dir_files[subdir]
        return to_ignore

    def ignore_test(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = test_dir_files[subdir]
        return to_ignore

    copytree('/content/food-101/images', '/content/food-101/test', ignore=ignore_train)
    copytree('/content/food-101/images', '/content/food-101/train', ignore=ignore_test)
    
else:
    print('Train/Test files already copied into separate folders.')

# !rm -r /content/drive/My\ Drive/project2/food-101/test

os.path.isdir('/content/food-101/train')

os.path.isdir('/content/food-101/test')

from __future__ import print_function
from __future__ import division

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import argparse


def setup_generator(train_path, test_path, batch_size, dimentions):
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        test_path, # this is the target directory
        target_size=dimentions,
        batch_size=batch_size)

    return train_generator, validation_generator

def load_image(img_path, dimentions, rescale=1. / 255):
    img = image.load_img(img_path, target_size=dimentions)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x *= rescale # rescale the same as when trained

    return x

# def create_model(num_classes, dropout, shape):
#     base_model = ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_tensor=Input(
#             shape=shape))
# #     for l in base_model.layers[:-10]:
# #         l.trainable = False
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dropout(dropout)(x)
#     predictions = Dense(num_classes, activation='softmax')(x)

#     model_final = Model(inputs=base_model.input, outputs=predictions)
#     model_final.summary()
# #     model_final.load_weights("/content/drive/My Drive/project2/weights-resnet50.epoch-06-val_loss-1.14.hdf5")
#     return model_final

# def train_model(model_final, train_generator, validation_generator, callbacks):
#     model_final.compile(
#         loss='categorical_crossentropy',
#         optimizer='adam',
#         metrics=['accuracy'])

#     model_final.fit_generator(train_generator, validation_data=validation_generator,
#                               epochs=10, callbacks=callbacks,
#                               steps_per_epoch=train_generator.samples//32,
#                               validation_steps=validation_generator.samples//32)

shape = (224, 224, 3)
X_train, X_test = setup_generator('/content/food-101/train', '/content/food-101/test', 32, shape[:2])


X_train.num_classes

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
x = base_model.output
x = AveragePooling2D()(x)

x = Dropout(.5)(x)
x = Flatten()(x)
predictions = Dense(X_train.num_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

opt = SGD(lr=.1, momentum=.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='/content/drive/My Drive/project2/lan2_v/model4_lan2.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('/content/drive/My Drive/project2/lan2_v/model4.log')

def schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return .0002
    elif epoch < 15:
        return 0.00002
    else:
        return .0000005
lr_scheduler = LearningRateScheduler(schedule)
model.summary()



model = load_model(filepath='/content/drive/My Drive/project2/model4.18-0.81.hdf5')

model.fit_generator(X_train, validation_data=X_test,
                              epochs=50,
                              steps_per_epoch=X_train.samples//32,
                              validation_steps=X_test.samples//32,
                               callbacks=[lr_scheduler, csv_logger, checkpointer])