import cv2
import numpy as np
import face_recognition
import os,pickle

path = "IMAGES"
names=[]
images=[]
encodes=[]
paths = os.listdir(path)

for i in paths:
    f = open(f'{path}/{i}', "rb")
    image = cv2.imread(f'{path}/{i}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # x = pickle.load(f)

    x = face_recognition.face_encodings(image)[0]

    encodes.append(x)
    names.append(i.split(".")[0])

l = len(encodes)

cap = cv2.VideoCapture(0)

dir = os.listdir(path)

while True:
    success, img = cap.read()
    imag = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facelocation = face_recognition.face_locations(imag)
    faceencoding = face_recognition.face_encodings(imag,facelocation)


    for faceenc, faceloc in zip(faceencoding, facelocation):

        matches = face_recognition.compare_faces(encodes, faceenc)
        facedistance = face_recognition.face_distance(encodes, faceenc)
        y1, x1, y2, x2 = faceloc[0], faceloc[1], faceloc[2], faceloc[3]



        matchindex=np.argmin(facedistance)

        print(matchindex)
        if matches[matchindex] > 4.5:
            name = names[matchindex].upper()
            print(name)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0,), 2)
        cv2.putText(img, names[matchindex], (x2 + 20, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1,
                    cv2.LINE_AA)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)

