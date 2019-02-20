"""DetectorKernel类的使用方法"""
import cv2
from DetectorKernel import SURF_Flann_DetectorKernel,SIFT_Flann_DetectorKernel

train_img_paths=[]
train_imgs=[]
train_labels=[]

training_path='../TrainImages/%s-%d.pgm'
for i in range(60):
    train_img_path=training_path%('pos',i)
    train_img_paths.append(train_img_path)
    train_imgs.append(cv2.imread(train_img_path,cv2.IMREAD_GRAYSCALE))
    train_labels.append(1)
    train_img_path=training_path%('neg',i)
    train_img_paths.append(train_img_path)
    train_imgs.append(cv2.imread(train_img_path,cv2.IMREAD_GRAYSCALE))
    train_labels.append(-1)

"""使用例1"""
detector=SURF_Flann_DetectorKernel()
#推荐方法initAndTrain或initAndTrainFromFiles
detector.initAndTrainFromFiles(train_img_paths,train_labels,120,40)

img=cv2.imread(training_path%('pos',100),cv2.IMREAD_GRAYSCALE)
print(detector.detect(img))
print(detector.detectFile(training_path%('neg',100)))
print(detector.detectFile('../c0.png'))

"""使用例2"""
SURF_Flann_detector=SURF_Flann_DetectorKernel()
detector=SURF_Flann_detector
#读图像训练BOW和SVM
detector.initVocabulary(train_imgs,120)
detector.trainSVM(train_imgs,train_labels)

right_reject_rate=0
false_accept_rate=0
for i in range(100,200):
    #读图像方法
    img=cv2.imread(training_path%('pos',i),cv2.IMREAD_GRAYSCALE)
    cls,score=detector.detect(img)
    if cls!=1:
        right_reject_rate+=1
    #读文件方法
    cls,score=detector.detectFile(training_path%('neg',i))
    if cls==1:
        false_accept_rate+=1
print('SURF_Flann方法的错误拒绝率='+str(right_reject_rate)+'%')
print('SURF_Flann方法的错误接受率='+str(false_accept_rate)+'%')

"""使用例3"""
SIFT_Flann_detector=SIFT_Flann_DetectorKernel()
detector=SIFT_Flann_detector
#读文件训练BOW和SVM
detector.initVocabularyFromFiles(train_img_paths,120)
detector.trainSVMFromFiles(train_img_paths,train_labels)

right_reject_rate=0
false_accept_rate=0
for i in range(100,200):
    img=cv2.imread(training_path%('pos',i),cv2.IMREAD_GRAYSCALE)
    cls,score=detector.detect(img)
    if cls!=1:
        right_reject_rate+=1
    cls,score=detector.detectFile(training_path%('neg',i))
    if cls==1:
        false_accept_rate+=1
print('SIFT_Flann方法的错误拒绝率='+str(right_reject_rate)+'%')
print('SIFT_Flann方法的错误接受率='+str(false_accept_rate)+'%')
