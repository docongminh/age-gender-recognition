import os
import cv2
import sys
import dlib
from datetime import date
import time
import csv 
import numpy as np
import argparse
import keras
print(">>>Keras version: ", keras.__version__)
from keras.models import model_from_json
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb

FEMALE = 0
MALE = 1
label_age = {
        '0': '(1-12)',
        '1': '(13-18)',
        '2': '(19- 22)',
        '3': '(23-29)',
        '4': '(30-34)',
        '5': '(35-39)',
        '6': '(40-44)',
        '7': '(45-50)',
        '8': '(51-59)',
        '9': '(>60)'
}


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def gen_csv(path):
    row = ['gender', 'age', "time"]
    with open(path, "a", newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)


def save_csv(genders, ages, path):
    print('gender: {genders}, age: {ages}, path: {path}'.format(genders=genders, ages=ages, path=path))
    print(os.path.exists(path))
    with open(path, 'a', newline='') as writeFile:
        writer = csv.writer(writeFile)
        # print(writer)
        for i in range(len(genders)):
            try:
                age = str(label_age[str(ages[i])])
                sex = 'FEMALE' if genders[i] == FEMALE else 'MALE'
                t = time.strftime('%H:%M:%S')
                # print(age, sex)
                writer.writerow([sex, age, t])
            except Exception as e:
                with open('logs/error_csv.log', 'w') as el:
                    el.write(str(e))


def main(model):
    depth = 16
    k = 8
    # for face detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)

    # load model and weights
    img_size = 160

    # capture video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()
    while True:
        # get video frame
        ret, img = cap.read()
        if not ret:
            print("error: failed to capture image")
            return -1
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        for i, d in enumerate(detected):

            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - 0.4 * w), 0)
            yw1 = max(int(y1 - 0.4 * h), 0)
            xw2 = min(int(x2 + 0.4 * w), img_w - 1)
            yw2 = min(int(y2 + 0.4 * h), img_h - 1)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)

            faces[i, :, :, :] = fa.align(input_img, gray, detected[i])
            faces = faces.astype('float32')/255
            # faces[i,:,:,:] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
        #
        if len(detected) > 0:
            age_list = []
            gender_list = []
            print(">>> face shape: ", faces.shape)
            gender_arr, age_arr = model.predict(faces)
            # get gender
            for gen in gender_arr:
                gender_list.append(np.argmax(gen))
            # get age
            for age in age_arr:
                age_list.append(np.argmax(age))
            # print("estimates", ages, genders)
            time_ = (time.time()-start_time)  
            if time_ > sec:
                path_out = "output"+"/{}.csv".format(date.today())
                if not os.path.exists(path_out):
                    gen_csv(path_out)
                # print("=========================",gender_list, age_list)
                save_csv(gender_list, age_list, path_out)
                # print("dump....")
                start_time = time.time()

        # draw results
        for i, d in enumerate(detected):
            __age = str(age_list[i])
            draw_age = label_age[__age]
            label = "{}, {}".format(draw_age, "F" if gender_list[i] == 0 else "M")
            draw_label(img, (d.left(), d.top()), label)

        cv2.imshow("Output of predict", img)
        # cv2.imwrite("output/result{}.jpg".format(count), img)
        key = cv2.waitKey(1)

        if key == 27:
            break
    

def load_model(model_path):

    json_file = open(model_path + 'inceptionv4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights(model_path + "inceptionv4.h5")
    print("Loaded model from disk")

    return model


if __name__ == '__main__':

    if not os.path.exists('./output'):
        os.mkdir('./output')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    try:
        path_model = './model_v4/'
        model_v4 = load_model(path_model)
        # print(age, gender)
        sec = int(sys.argv[1]) if len(sys.argv) > 1 else 3
        main(model_v4)
    except Exception as err:
        with open('logs/error.log', 'w') as f:
            f.write(str(err))
            f.write("\n")
