import pymongo
import pickle
import cv2,face_recognition
import numpy as np

enc=[]

f = open("Encodings/encodings.pickle", "rb")

x = pickle.load(f)

f.close()

image = cv2.imread("IMAGES/UTKARSH RAI.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
encode = face_recognition.face_encodings(image)[0]

#encode = np.asarray(encode)
enc.append(encode)

com = face_recognition.compare_faces(x,encode)
dis = face_recognition.face_distance(x,encode)

print(x)