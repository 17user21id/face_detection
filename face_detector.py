import cv2

#load some pre-trained data on face frontals from open-cv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#capture video from webcam
webcam = cv2.VideoCapture(0)

#looping forever over frame
while True:
    #Read the current frame
    check_sucessful_frame , frame = webcam.read()

    #converting to grayscale
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detecting faces
    face_cordinate = trained_face_data.detectMultiScale(grayscaled_image)

    #draw rectangle around the faces
    for (x, y, w, h) in face_cordinate:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) ,(0,255,255) , 2)

    #show frames
    cv2.imshow("face detection" , frame)
    key = cv2.waitKey(1)

    #exit screen when Q is pressed
    if key==81 or key==113:
        break

#release the video capture object
webcam.release()

