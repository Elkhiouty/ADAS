import cv2
import numpy as np

def coordinates(h, line_parameters):
    slope, intercept = line_parameters
    y1 = int(h)
    y2 = int(y1*0.8)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return x1, y1, x2, y2

def bumb_coordinates(line_parameters, x1, x2):
    slope, intercept = line_parameters
    y1 = int(slope*x1+intercept)
    y2 = int(slope*x2 + intercept)
    return x1, y1, x2, y2

def process(img): 
    w = int(img.shape[1])
    h = int(img.shape[0])
    roi = img[int(0.6*h):h,0:int(w)]
    roi = cv2.GaussianBlur(roi,(5,5),0)
    edges = cv2.Canny(roi,25,150)
    mask = np.zeros_like(img)
    mask = mask[:,:,0]
    mask[int(0.6*h):h,0:int(w)] = edges
    lines = cv2.HoughLinesP(mask,1,np.pi/180,50,maxLineGap=150)
    draw = 1
    left_fit_avg = np.polyfit((int(0.2*w), int(0.4*w)), (h, int(0.8*h)), 1)
    right_fit_avg = np.polyfit((int(0.8*w), int(0.6*w)), (h, int(0.8*h)), 1)
    c="N"
    
    if lines is not None:
        left_fit = []
        right_fit = []
        line_fit = []
        
        slope, intercept = np.polyfit((int(0.2*w), int(0.4*w)), (h, int(0.8*h)), 1)
        left_fit.append((slope, intercept))
        slope, intercept = np.polyfit((int(0.8*w), int(0.6*w)), (h, int(0.8*h)), 1)
        right_fit.append((slope, intercept))
        for line in lines:
            x1,y1,x2,y2=line[0]
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if (-0.05 < slope < 0.05): # take only horizontal lines
                line_fit.append((slope, intercept))
            if (-0.45 < slope < 0.45): #horizental filter
                continue
            else :
                draw = 0
            if slope > 0 : 
                right_fit.append((slope, intercept))
            else :
                left_fit.append((slope, intercept))
        
        if(len(left_fit) == 1) and (len(right_fit) == 1) :
            c="N"
        elif (len(right_fit) == 1) :
            c ="R"
        elif (len(left_fit) == 1):
            c = "L"
        else :
            c = "C"
            
        left_fit_avg = np.average(left_fit, axis =0)
        right_fit_avg = np.average(right_fit, axis =0)
        l_x1,l_y1,l_x2,l_y2 = coordinates(h, left_fit_avg)
        r_x1,r_y1,r_x2,r_y2 = coordinates(h, right_fit_avg)
        
#=====================================Bumb=================================================
        # if len(line_fit):
        #     line_fit_avg = np.average(line_fit, axis =0)
        #     x1, y1, x2, y2 = bumb_coordinates(line_fit_avg, l_x2, r_x2)
        #     cv2.line(img,(x1, y1),(x2, y2),(0,0,250),5)
        #     cv2.putText(img, "Bump", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
#==========================================================================================
    if draw :
      cv2.line(img,(int(0.2*w), h),(int(0.4*w), int(0.8*h)),(210,250,250),5)
      cv2.line(img,(int(0.8*w), h),(int(0.6*w), int(0.8*h)),(210,250,250),5)  
    else :
        p = np.array([[r_x1,r_y1],[r_x2,r_y2],
                      [l_x2,l_y2],[l_x1,l_y1]])
        cv2.fillPoly(img, [p], (210,250,250))

    return img,left_fit_avg,right_fit_avg,c
