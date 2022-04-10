import cv2
import matplotlib.pyplot as plt

config_file = 'AIcode.pbtxt'
frozen_model = 'inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = [] # empty List of python
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

#This part will reshape the img
model.setInputSize(320,320)
model.setInputScale(1.0/127.5) ## 255/2 = 127.5
model.setInputMean((127.5,127.5,127.5)) ##
model.setInputSwapRB(True) # to change the color automatic

# Change the img to the img you want to be verified
img = cv2.imread("imgDemo1.jpg")

ClassIndex, confidence, bbox = model.detect(img,confThreshold=0.3)

#This part will print the indexes found in the img
print(ClassIndex)

#This part will plot the boxes to the img
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    #cv2.rectangle(frame, (x ,y), (x+w, y+h), (255,0,0),2)
    #cv2.putText(img, text, (test_offset_x,text_offset_y), fontScale=font_scale, color=(0,0,0), thickness=1)
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img, classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale,color=(0,255,0), thickness=3)

#This step will print the img with the plotted identifier boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
