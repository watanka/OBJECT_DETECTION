import cv2

# 동영상을 읽어온다.
cap = cv2.VideoCapture('video/나로산책.mp4')

if cap.isOpened() == False :
    print("Error opening video stream or file")

while cap.isOpened() :
    ret, frame = cap.read()
    if ret == True :

        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q') :
            break

    # Break the loop
    else :
        break

# when everything is done, release the video capture object
cap.release()
cv2.destroyAllWindows()  