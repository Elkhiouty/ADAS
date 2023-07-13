import cv2
import preprocess
import laneDetection

def pipeline(frame):
    image = frame

    #Disabled, techinically each frame needs to be undistored before being processed.
    #objpoints, imgpoints = [] #Add them manually
    #frame = calibrateCamera.calibrate(objpoints, imgpoints, frame)

    frame, invM = preprocess.warp(frame)
    frame = preprocess.grayscale(frame)
    frame = preprocess.threshold(frame)
    frame, left_curverad, right_curverad = laneDetection.search_around_poly(frame)
    frame = cv2.warpPerspective(frame, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

    #Add curvature and distance from the center
    curvature = (left_curverad + right_curverad) / 2
    car_pos = image.shape[1] / 2
    center = (abs(car_pos - curvature)*(3.7/650))/10
    curvature = 'Radius of Curvature: ' + str(round(curvature, 2)) + 'm'
    center = str(round(center, 3)) + 'm away from center'
    frame = cv2.putText(frame, curvature, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, center, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


