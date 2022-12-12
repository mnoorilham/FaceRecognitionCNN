#Buat dataset baru menggunakan facecam
import os

import cv2

dataset_folder = "dataset/"

cap = cv2.VideoCapture(0)

my_name = "Muhammad_Noor_Ilham"
os.mkdir(dataset_folder + my_name)
num_sample = 160

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Capture Photo", frame)
        cv2.imwrite("dataset/%s/%s_%04d.jpg" % (my_name, my_name, i), cv2.resize(frame, (250, 250)))

        if cv2.waitKey(100) == ord('q') or i == num_sample:
            break
        i += 1
cap.release()
cv2.destroyAllWindows()
