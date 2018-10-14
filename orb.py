import cv2
img = cv2.imread('test.jpg', 1)
img = cv2.resize(img, None, fx=0.25, fy=0.25)

# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img, None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=1)
cv2.imshow('keypoints', img2)
cv2.waitKey(0)
