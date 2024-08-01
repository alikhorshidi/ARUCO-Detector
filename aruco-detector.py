import cv2 as cv2 #make sure you've installed OpenCV successfully 
 
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100) #you can change the type

parameters =  cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(dictionary, parameters)
 
def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
 
        ids = ids.flatten()
 
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
 
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
 
            cv2.line(image, topLeft, topRight, (255, 0, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 0, 255), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
 
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 145, 255), -1) #small dot in the center of marker
 
            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 145, 255), 2)  #show ID inside feed
            print("[Inference] ArUco marker ID: {}".format(markerID))  #show ID on terminal
 
    return image
 
cap = cv2.VideoCapture(2)  #change number to find your webcam feed. usually it should be 0 
 
while cap.isOpened():
 
    ret, img = cap.read()
 
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(img)
 
    print(markerCorners) 
 
    detected_markers = aruco_display(markerCorners, markerIds, rejectedCandidates, img)
 
    cv2.imshow("Markers", detected_markers)  #It's Show Time!!!
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  #press q to exit!
        break
 
cv2.destroyAllWindows()
cap.release()
