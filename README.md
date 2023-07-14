# ADAS

#### This is my graduation project, we enhanced some ADAS systems using AI and image processing.

## Which systems did we enhance?

### 1- Adaptive cruise control (ACC)
#### 1-Set max speed automatically.
#### 2-Distance measurement without sensors.
### 2- Lane centering and keeping
#### Lane detection using slope only, which gives efficient and fast detection.

## How did we enhance this system?

### 1- Object Detection using YOLOv7
#### We trained the model on a custom dataset that contains seven classes
#### [car - truck/bus - van - person - bike - sign - plate]

### 2- Distance Mesaurement
#### We measured the distance using a single camera only depending on this equation

#### F * H = P * D

#### F ==> Focal Lenght
#### H ==> Vehicle Height
#### P ==> Count of Pixels
#### D ==> Distance

### 3- Sign Recognition
#### We trained the model on a German traffic signs dataset [Link](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign), then we froze the unneeded cells to have better accuracy on the most common traffic sign on the highways

### 4- Lane Detection
####  We managed to detect our lane by calculating the slopes on all the lines on the image and filtering out all unrelated lines

### 5- Sending the Information
####  We send this information through the UART protocol to the AVR or through UDP to the Raspberry Pi to take action

## Testing environment
####  We created a testing environment on an open-world  game to test our model, so we simulated our keyboard keys to control the vehicle in the game and see its behavior on the roads