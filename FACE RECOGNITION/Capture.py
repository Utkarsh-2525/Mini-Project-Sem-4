import cv2,pickle
import os
import face_recognition
import sqlite3

conn = sqlite3.connect('test.db')
print "Opened database successfully";

path ="Encodings"
cap = cv2.VideoCapture(0)

name = input("Name : ")

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('\r'):
        a = frame
        break
del(cap)

a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

encode = face_recognition.face_encodings(a)


print(encode)
