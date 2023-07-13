from yolo import yolo
import cv2

obj=yolo(no_trace=True,view_img =True)

model,stride,imgsz=obj.load_model(weights='weights.pt',img_size=800)

# source accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
dataset,webcam,view_img=obj.Dataloader(source = '5',                
                                     imgsz=imgsz,stride=stride) 

try :
    obj.run(model= model,dataset= dataset,webcam= webcam,view_img= view_img,
          conf_thres=0.55)
    
finally :
    cv2.destroyAllWindows()