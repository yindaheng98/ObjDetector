import cv2
import numpy as np
from Detector import Detector
#cv2.imshow("img", cv2.imread(training_path%('pos',100),cv2.IMREAD_GRAYSCALE))
#training_path='../TrainImages/%s-%d.pgm'
#img=cv2.imread(training_path%('pos',100),cv2.IMREAD_GRAYSCALE)
#img_path='../c0.png'
#img_path='../c1.jpg'
img_path='../TestImages_Scale/test-2.pgm'
img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
print(img.shape)

from DetectorKernelExamples import SURF_Flann_detector as detector_kernel
scales=np.arange(0.1,1,0.1)
shape=(100,40)
step_size=10
detector=Detector(detector_kernel,scales,shape,step_size)
'''
for res in detector.detect_results_generator(img):
    print(res)
'''
boxes_dict=detector.detect_non_max(img,0.25,[-1])
for cls in boxes_dict:
    for place,shape,score in boxes_dict[cls][-3:]:
        cv2.rectangle(img,
                      place,
                      (int(place[0]+shape[0]),int(place[1]+shape[1])),
                      (0, 255, 0), 1)
        cv2.putText(img, "%f" % score, place, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
cv2.imshow("img", img)
print(boxes_dict)
