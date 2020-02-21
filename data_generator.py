import numpy as np
import keras
import cv2


class DataGenerator(keras.utils.Sequence):

    """generates data_agu for keras"""

    def __init__(self, list_IDs, batch_size, n_channels=1, shuffle=True):

        """ Initialization"""

        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):

        """ Denotes the number of batches per epoch """

        return int(np.floor(len(self.list_IDs)) / self.batch_size)

    def __getitem__(self, index):

        """generate one batch of data_agu"""

        # generate index of the batch data_agu
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """ generate data_agu containing batch_size samples"""

        X = []
        y_gender = []
        y_age = []
        #gen data_agu
        for path in list_IDs_temp:
            # store sample
            path = path.strip()
            img = cv2.imread(path)
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("====shape image: ",img.shape)
            image = cv2.resize(cvt_img, (160, 160))
            image = image.astype('float32')/255
            X.append(image)

            #    get label gender
            label_gender = path.split("/")[-2]
            y_gender.append(label_gender)

            # get label of age
            label_age = path.split("/")[-3]
            y_age.append(label_age)

        X = np.array(X)
        age_label = keras.utils.to_categorical(y_age, num_classes=10)
        gender_label = keras.utils.to_categorical(y_gender, num_classes=2)

        return X, [gender_label, age_label]
