import cv2,os
import numpy as np
from PIL import Image

import cv2,os
import numpy as np
from PIL import Image
import pickle


recognizer1 = cv2.face.LBPHFaceRecognizer_create()
recognizer2 = cv2.face.LBPHFaceRecognizer_create()
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataSet'

def get_images_and_labels(path):

     image_paths = [os.path.join(path, f) for f in os.listdir(path)]

     # images will contains face images
     images = []

     # labels will contains the label that is assigned to the image
     labels = []

     for image_path in image_paths:
         # Read the image and convert to grayscale
         image_pil = Image.open(image_path).convert('L')


         # Convert the image format into numpy array
         image = np.array(image_pil, 'uint8')

         # Get the label of the image
         nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))

         #nbr=int(''.join(str(ord(c)) for c in nbr))
         print (nbr)

         # Detect the face in the image
         faces = faceCascade.detectMultiScale(image)

         # If face is detected, append the face to images and the label to labels
         for (x, y, w, h) in faces:
             images.append(image[y: y + h, x: x + w])
             labels.append(nbr)

             cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
             cv2.waitKey(10)

     # return the images list and labels list
     return images, labels


images,labels = get_images_and_labels(path)
cv2.imshow('test', images[0])
cv2.waitKey(1)

recognizer1.train(np.asarray(images), np.array(labels))
recognizer2 = recognizer1
recognizer2.save('trainer/trainer.yml')

print('Training successfully done!!  ')

cv2.destroyAllWindows()



###########################################################################################
####################                P R E D E C T I O N              ######################
###########################################################################################

recognizer2.read('trainer/trainer.yml')

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    (width, height) = (130, 100)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]  # cropping the captured face
        face_resize = cv2.resize(roi_gray, (width, height))

        nbr_predicted, conf = recognizer2.predict(face_resize)
        result = recognizer1.predict(gray[y:y+h, x:x+w])


        print (nbr_predicted)

        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))
            display_string = str(confidence) + '% Confidence it is user'
        cv2.putText(im, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

        if confidence > 65:
            cv2.putText(im, "Unlocked id : "+nbr_predicted.__str__(), (100, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            cv2.imshow('im', im)


        else :
            cv2.putText(im, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('im', im)

    if faces is ():
        cv2.putText(im, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('im', im)

    if cv2.waitKey(1) == 13:
            break

#cap.release()
#cv2.destroyAllWindows()