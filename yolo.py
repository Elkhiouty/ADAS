import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from keras.models import load_model
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression,scale_coords
from utils.general import  strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel
from lines import process
from tracker import EuclideanDistTracker
from Game import PressKey,ReleaseKey
import time
#from uart import send  #for communication with UART
from udp import send    #for communication with UDP

class yolo:
    
    def __init__(self, device='',view_img=True,update = True,no_trace=False):
        self.view_img = view_img
        self.update = update
        self.trace = not no_trace
        self.device,self.half = self.read_device(device)
        self.vehciles = {}
        self.focal_length = 100  #camera focal lenght
        self.car = 140           #car height
        self.person = 170        #person height
        self.van = 190           #van height
        self.truck = 350         #truck height
        self.model = load_model('sign_model_transfer.h5')   #sign model
        self.tracker = EuclideanDistTracker()    #tracker 
        self.classes ={                       #sign classes
                0: 'Turn Right',
                1: 'Turn Left',
                2: 'Bump',
                3: 'Stop',
                4: 'Speed Limit 20',
                5: 'Speed Limit 30',
                6: 'Speed Limit 50',
                7: 'Speed Limit 60',
                8: 'Speed Limit 70',
                9: 'Speed Limit 80',
                10:'other'
                }
        self.buttons ={           #game contrl buttons
            'w': 0x11,
            'a': 0x1E,
            's': 0x1F,
            'd': 0x20
            }


    def read_device(self,device):
        # Initialize
        set_logging()
        device = select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        return device,half


    def load_model(self,weights, img_size):
        with torch.no_grad():
            if self.update:  # update all models (to fix SourceChangeWarning)
                for weight in [weights]:
                    model = attempt_load(weight, map_location=self.device)  # load FP32 model
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(img_size, s=stride)  # check img_size

                    if self.trace:
                        model = TracedModel(model, self.device, self.img_size)

                    if self.half:
                        model.half()
                        
                    strip_optimizer(weight)
        return(model,stride,imgsz)
    
    
    def Dataloader(self,source ='inference/images',imgsz=800,stride=32):
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
        view_img = self.view_img
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
        return(dataset,webcam,view_img)
                    
    def sign(self,img):                       #  Sign Recognation
        data=[]
        image = cv2.resize(img, (30,30), interpolation = cv2.INTER_AREA)
        data.append(np.array(image))
        X_test=np.array(data)
        Y_pred = self.model.predict(X_test)
        i =10
        for y in Y_pred[0]:
            if round(y,2) >= 0.55:
                i = list(Y_pred[0]).index(y)
                break
        send(self.classes[i])
        return(self.classes[i])
    
    
    def compare(self,l,r,y,x,d):     # Determine if the car is in the same lane or not
        slope, intercept = l
        x1 = int((y-intercept)/slope)
        slope, intercept = r
        x2 = int((y-intercept)/slope)
        if (x1-100)<x<(x2+100) :
            return d
        else :
            return 0
                
    def keys(self,elem):         # For Sorting
        return elem[3]
    
    
    def run(self,model,dataset,webcam,view_img,conf_thres=0.55, iou_thres=0.45,
            save_conf=False,classes=None,agnostic_nms=True,augment=True,imgsz=800):
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
        skip = 0
        s_img=[]
        #signn = ""
        si=0
        ready = 0
        for path, img, im0s, vid_cap in dataset:
            if skip:
                skip = skip - 1
                continue
            else:
                skip = 5
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]

            # Inference
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=augment)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    im0 = im0s[i].copy()
                else:
                    im0 = im0s
                # =============== Lane and bumb detection =====================
                im0,l,r,c = process(im0)
                if c == "N":
                    #ReleaseKey(self.buttons['w'])
                    cv2.putText(im0, "No Lanes Detected", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1,cv2.LINE_AA)
                elif c == "R":
                    send('R')
                    #ReleaseKey(self.buttons['w'])
                    #ReleaseKey(self.buttons['a'])
                    PressKey(self.buttons['d'])
                    time.sleep(0.30)
                    ReleaseKey(self.buttons['d'])
                    cv2.putText(im0, "Go Right", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1,cv2.LINE_AA)
                elif c == "L":
                    send('L')
                    #ReleaseKey(self.buttons['w'])
                    #ReleaseKey(self.buttons['d'])
                    PressKey(self.buttons['a'])
                    time.sleep(0.30)
                    ReleaseKey(self.buttons['a'])
                    cv2.putText(im0, "Go Left", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1,cv2.LINE_AA)
                else:
                    send('C')
                    ReleaseKey(self.buttons['a'])
                    ReleaseKey(self.buttons['d'])
                    if ready:
                        PressKey(self.buttons['w'])
                    else :
                        ReleaseKey(self.buttons['w'])
                    cv2.putText(im0, "Center", (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1,cv2.LINE_AA)
                #==============================================================
                    
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    vehciles = []
                    for o in det[:,:] :
                        y = int(o[1])
                        x = int(o[0])
                        h = int(o[3])- int(o[1])
                        w = int(o[2])- int(o[0])
                        c = names[int(o[5])]
                        # Equation ==>            F x H = P x D 
                        if (c=="car"):
                            d=round(((self.car * self.focal_length)/(100*h)),2)
                            if d <= 25.0 :
                                vehciles.append([x,y,w,h,d])
                        elif (c=="truck"):
                            d=round(((self.truck * self.focal_length)/(100*h)),2)
                            if d <= 25.0 :
                                vehciles.append([x,y,w,h,d])
                        elif (c=="van"):
                            d=round(((self.van * self.focal_length)/(100*h)),2)
                            if round((d/100),2) <= 25.0 :
                                vehciles.append([x,y,w,h,d])
                        elif(c=="person"):
                            d=round(((self.person * self.focal_length)/(100*h)),2)
                            if round((d/100),2) <= 25.0 :
                                vehciles.append([x,y,w,h,d])
                        elif (c =="sign") :
                            s_img=im0[y:y+h,x:x+w]
                            sign_type = self.sign(s_img)
                            si=50
                    
                    vehciles = self.tracker.update(vehciles) #to get id for each car
                    
                    if ((len(vehciles)) > 3):                        
                        vehciles =sorted(vehciles,reverse = True ,key=(self.keys))[:4]
                    
                    if (len(vehciles)):
                        nearest = []
                        for veh in vehciles:
                            x,y,_,_,d,id = veh
                            cv2.putText(im0, f"D: {d} M", (x, y-15),cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2,cv2.LINE_AA)
                            cv2.putText(im0, "id:" + str(id), (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2,cv2.LINE_AA)
                            if l is not None:
                                if self.compare(l,r,y+h,x+0.5*w,d):    
                                    nearest.append((d,id))
                        if len(nearest):
                            n_d,n_id=min(nearest)
                            send(f'Object ID : {n_id},Distance :{n_d}')
                            # for Game control
                            if n_d > 4 :
                                PressKey(self.buttons['w'])
                                ready = 1
                            elif n_d < 4 :
                                ReleaseKey(self.buttons['w'])
                                ready = 0
                            cv2.putText(im0, "Nearest object : " + str(n_d) + " M", (0, 35), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1,cv2.LINE_AA)
                        else :
                            PressKey(self.buttons['w'])
                            ready = 1
                    else :
                        PressKey(self.buttons['w'])
                        ready = 1
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                else:
                    ready = 1

                if si:          #  Crop the sign image and show on the top left 
                    dim = (50, 50)
                    s_img=cv2.resize(s_img, dim, interpolation = cv2.INTER_AREA)
                    im0[20:20+s_img.shape[0],im0.shape[1]-s_img.shape[1]-20:im0.shape[1]-20] = s_img
                    #cv2.putText(im0, "sign: " + sign_type, (0, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                    si=si-1 
               
                # Stream results
                if view_img:
                    dim = (1280,800)
                    im0=cv2.resize(im0, dim, interpolation = cv2.INTER_AREA)
                    cv2.imshow("Control", im0)
                    cv2.waitKey(1)  # 1 millisecond
               
        cv2.destroyAllWindows()
    
    
