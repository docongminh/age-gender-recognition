import keras
import pathlib
import random
import argparse
from collections import Counter
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from data_generator import DataGenerator
from inceptionv4 import inceptionv4
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import tensorflow as tf
from keras import backend as K
from collections import Counter
from tqdm import tqdm


K.tensorflow_backend._get_available_gpus()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def visual_trainning(H, number_epoch):
    # plot visualize loss, accuracy of traning set và validation set
    fig = plt.figure()
    numOfEpoch = 10
    plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
    plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
    plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
    plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
    plt.title('Accuracy and Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss|Accuracy')
    plt.legend()


def get_list_path(path):
    """
    :param path:  path to datasets ( data )
    :return: list of path to image with format: data/{age_label}/{gender_label/img.*}
    """
    data_root = pathlib.Path(path)
    all_image_paths = data_root.glob('*/*/*')
    all_path_images = [str(path_img) for path_img in all_image_paths]
    random.shuffle(all_path_images)
    return all_path_images


def remove_path_incorrect(list_paths):
    """ remove the path incorect or cannot read image"""
    print("removing inccorect path ...")
    list_path_tmp = []
    for path in tqdm(list_paths):
        try:
            img = cv2.imread(path)
            if img is None: continue
        except:
            continue

        list_path_tmp.append(path)

    print("Done...!")
    return list_path_tmp


if __name__ == '__main__':

    list_path_train = pickle.load(open("../../X_train.pkl", "rb"))
    list_path_test = pickle.load(open("../../X_test.pkl", "rb"))

    #  filtet path to remove path incorrect
    list_paths_train = remove_path_incorrect(list_path_train)
    list_paths_test = remove_path_incorrect(list_path_test)

    model = inceptionv4(input_shape=(160, 160, 3), dropout_keep=0.8)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=opt, loss={'age_output': 'categorical_crossentropy',
                                       'gender_output': 'categorical_crossentropy'},
                  loss_weights={'age_output': 2.,
                                'gender_output': 1.},
                  metrics={'age_output': 'accuracy',
                           'gender_output': 'accuracy'})

    # Parameters
    params = {'batch_size': 64,
              'n_channels': 1,
              # 'n_classes': 2,
              'shuffle': True}
    train_generator = DataGenerator(list_paths_train, **params)
    test_generator = DataGenerator(list_paths_test, **params)
    checkpoint = ModelCheckpoint('checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                                save_best_only=True,
                                save_weights_only=False,
                                period=1)
    tf_board = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0,
                                           batch_size=64, write_graph=True, write_grads=True,
                                           write_images=True,
                                           update_freq='epoch')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', min_delta=0.001,
                                  min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=0,
                                               restore_best_weights=True)

    # fit
    print("train generator: ", train_generator)
    model.fit_generator(generator=train_generator,
                        validation_data=test_generator,
                        use_multiprocessing=False,
                        workers=8,
                        verbose=2,
                        callbacks=[reduce_lr, tf_board, checkpoint],
                        epochs=50)

    # save model
    model_json = model.to_json()
    with open("model_v4/inceptionv4.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_v4/inceptionv4.h5")
    print("Saved model to disk")
