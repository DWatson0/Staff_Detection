import cv2
import os

vid = cv2.VideoCapture('./sample.mp4')

if not os.path.exists('frame_extraction'):
    os.makedirs('frame_extraction')

currentFrame = 0

while (True):
    ret, frame = vid.read()
    if not ret:
        break
    cv2.imwrite('./data/dataframe'+str(currentFrame)+".jpg",frame)
    currentFrame += 1

vid.release()
cv2.destroyAllWindows()
