import keras
from keras import Model, Input
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, Flatten
from keras.layers import MaxPooling2D, concatenate, AveragePooling2D, Dense
import numpy as np


def preprocess_input(x):
    """ preprocess input"""
    x = np.divide(x, 255)
    x = np.subtract(x, 0.5)
    x = np.multiply(x, 2.0)
    return x


def conv_block(x, nb_filter, nb_row, nb_col, padding="same", strides=(1, 1), use_bias=False):
    """
    :param x:
    :param nb_filter:
    :param nb_row:
    :param nb_col:
    :param padding:
    :param strides:
    :param use_bias:
    :return:  a value after convolution
    """
    x = Conv2D(nb_filter, (nb_row, nb_col), strides=strides, padding=padding, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = Activation("relu")(x)
    return x


def stem(input):
    """
    the stem of the pure Inception-v4 and inception resnet v2 net works, this is input part of those networks
    """
    # input shape 299*299*3
    x = conv_block(input, 32, 3, 3, strides=(2, 2), padding="same")
    x = conv_block(x, 32, 3, 3, padding="same")
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding="same")

    # concatenate, ouput's dimension: (73, 73, 160)
    x = concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding="same")

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, padding="same")

    # concatenate, output: (71, 71, 192)
    x = concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding="same")
    x2 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # concatenate , ouput: (35, 35, 384)
    x = concatenate([x1, x2], axis=-1)
    return x


def inception_A(input):

    """Achitecture of inception A block """

    a1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    a1 = conv_block(a1, 96, 1, 1)

    a2 = conv_block(input, 96, 1, 1)

    a3 = conv_block(input, 64, 1, 1)

    a4 = conv_block(input, 64, 1, 1)
    a4 = conv_block(a4, 96, 3, 3)
    a4 = conv_block(a4, 96, 3, 3)

    #     concatenate, output demension
    a = concatenate([a1, a2, a3, a4], axis=-1)
    return a


def inception_B(input):

    """ Achitecture of inception B block """

    b1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    b1 = conv_block(b1, 128, 1, 1)

    b2 = conv_block(input, 384, 1, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = conv_block(input, 192, 1, 1)
    b4 = conv_block(b4, 192, 1, 7)
    b4 = conv_block(b4, 224, 7, 1)
    b4 = conv_block(b4, 224, 1, 7)
    b4 = conv_block(b4, 256, 7, 1)

    #     concatenate
    b = concatenate([b1, b2, b3, b4], axis=-1)
    return b


def inception_C(input):

    """ Architecture of inception C block"""

    c1 = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(input)
    c1 = conv_block(c1, 256, 1, 1)

    c2 = conv_block(input, 256, 1, 1)

    c3 = conv_block(input, 384, 1, 1)
    c31 = conv_block(c3, 256, 1, 3)
    c32 = conv_block(c3, 256, 3, 1)

    c4 = conv_block(input, 384, 1, 1)
    c4 = conv_block(c4, 448, 1, 3)
    c4 = conv_block(c4, 512, 3, 1)
    c41 = conv_block(c4, 256, 3, 1)
    c42 = conv_block(c4, 256, 1, 3)

    # concatenate
    c = concatenate([c1, c2, c31, c32, c41, c42], axis=-1)
    return c


def reduction_A(input):

    """Architecture reduction 35*35 to 17*17 dimension"""
    ra1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    ra2 = conv_block(input, 384, 3, 3, strides=(2, 2), padding="same")

    ra3 = conv_block(input, 192, 1, 1)
    ra3 = conv_block(ra3, 224, 3, 3)
    ra3 = conv_block(ra3, 256, 3, 3, strides=(2, 2), padding="same")

    # caoncatenate
    ra = concatenate([ra1, ra2, ra3], axis=-1)
    return ra


def reduction_B(input):
    """ Architecture reduction 17*17 to 8*8 dimension  """
    rb1 = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(input)

    rb2 = conv_block(input, 192, 1, 1, strides=(1, 1))
    rb2 = conv_block(rb2, 192, 3, 3, strides=(2, 2), padding="same")
    print(rb2.shape)

    rb3 = conv_block(input, 256, 1, 1)
    rb3 = conv_block(rb3, 256, 1, 7)
    rb3 = conv_block(rb3, 320, 7, 1)
    rb3 = conv_block(rb3, 320, 1, 1, strides=(2, 2), padding="same")

    #concatenate
    rb = concatenate([rb1, rb2, rb3], axis=-1)
    return rb


def inception_base_v4(input):
    net = stem(input)

    #     inception a
    for i in range(4):
        net = inception_A(net)

    #     reduction a
    net = reduction_A(net)

    # inception b
    for i in range(7):
        net = inception_B(net)

    #     reduction b
    net = reduction_B(net)

    # inception c
    for i in range(3):
        net = inception_C(net)

    return net


def inceptionv4(input_shape, dropout_keep, include_top = 1):
    """
    :param input_shape:
    :param dropout_keep:
    :param weigth:
    :param include_top:
    :param nb_class_age:
    :param nb_class_gender:
    :return:
    """
    inputs = Input(input_shape)

    net_base = inception_base_v4(inputs)

    # final pooling and predict
    if include_top:
        net = AveragePooling2D((4, 4), padding="valid")(net_base)
        net = Dropout(1-dropout_keep)(net)
        net = Flatten()(net)

    out_gender = Dense(units=2, activation="softmax", name="gender_output")(net)

    # layer of age model
    out_age = Dense(units=512, activation="relu", name="age_dense")(net)
    out_age = Dropout(0.4)(out_age)
    out_age = Dense(units=512, activation="relu", name="age_dense2")(out_age)
    out_age = Dropout(0.2)(out_age)
    out_age = Dense(units=10, activation="softmax", name="age_output")(out_age)

    model = Model(inputs, outputs=[out_gender, out_age], name="inceptionv4")
    return model
