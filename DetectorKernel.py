"""
用于单张图像判别的SVM判别器类及相关函数
使用方法1
    SURF_Flann_Detector或SIFT_Flann_Detector初始化
    ->initAndTrain或initAndTrainFromFiles初始化BOW词汇表和SVM
    ->detect或detectFile判断指定图像
使用方法2
    SURF_Flann_Detector或SIFT_Flann_Detector初始化
    ->initVocabulary或initVocabularyFromFiles初始化BOW词汇表
    ->trainSVM或trainSVMFromFiles训练SVM
    ->detect或detectFile判断指定图像
"""
import cv2
import numpy as np
class DetectorKernel:
    """
    SVM图像判别器类
    """
    def __init__(self,feature_alg,matcher_alg,svm):
        """
        构造判别器
        参数feature_alg:一个cv2特征提取算法类(SURF或SIFT等)
        参数matcher_alg:一个cv2匹配算法类(Flann等)
        """
        self.feature=feature_alg#特征提取算法
        self.bow_extractor=cv2.BOWImgDescriptorExtractor(feature_alg,matcher_alg)#BOW提取器
        self.svm=svm#SVM
        
    def initVocabulary(self,train_imgs,clusterNum):
        """
        初始化BOW词汇表
        参数train_imgs:训练用的图像list
        参数clusterNum:BOW单词数量
        """
        bow_trainer=cv2.BOWKMeansTrainer(clusterNum)#BOW训练器
        for img in train_imgs:
            _,features=self.feature.detectAndCompute(img,None)#提取特征
            bow_trainer.add(features)#加入bow训练器
        self.bow_extractor.setVocabulary(bow_trainer.cluster())#聚类训练，生成词汇表

    def initVocabularyFromFiles(self,train_img_paths,clusterNum):
        """
        从图片文件初始化BOW词汇表
        参数train_img_paths:训练用的图像文件路径list
        参数clusterNum:BOW单词数量
        """
        train_imgs=[]
        for img_path in train_img_paths:
            train_imgs.append(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
        self.initVocabulary(train_imgs,clusterNum)

    def trainSVM(self,train_imgs,train_labels):
        """
        训练SVM
        参数train_imgs:训练用的图像list
        参数train_labels:训练用的图像标签
        """
        train_data=[]
        for img in train_imgs:
            bow=self.bow_extractor.compute(img,self.feature.detect(img))#提取特征
            train_data.extend(bow)
        self.svm.train(np.array(train_data),cv2.ml.ROW_SAMPLE,np.array(train_labels))

    def trainSVMFromFiles(self,train_img_paths,train_labels):
        """
        从图片文件训练SVM
        参数train_img_paths:训练用的图像文件路径list
        参数train_labels:训练用的图像标签
        """
        train_imgs=[]
        for img_path in train_img_paths:
            train_imgs.append(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
        self.trainSVM(train_imgs,train_labels)

    def initAndTrain(self,train_imgs,train_labels,clusterNum,bow_init_num=0):
        """
        初始化BOW词汇表并且训练SVM
        参数train_imgs:训练用的图像list
        参数train_labels:训练用的图像标签
        参数clusterNum:BOW单词数量
        参数bow_init_num:选train_imgs中前几个用于构造BOW词汇表，默认为0全部使用
        """
        if bow_init_num<=0:
            bow_init_num=len(train_imgs)
        self.initVocabulary(train_imgs[0:bow_init_num],clusterNum)
        self.trainSVM(train_imgs,train_labels)

    def initAndTrainFromFiles(self,train_img_paths,train_labels,clusterNum,bow_init_num=0):
        """
        从图像文件中初始化BOW词汇表并且训练SVM
        参数train_img_paths:训练用的图像文件路径list
        参数train_labels:训练用的图像标签
        参数clusterNum:BOW单词数量
        参数bow_init_num:选train_imgs中前几个用于构造BOW词汇表，默认为0全部使用
        """
        train_imgs=[]
        for img_path in train_img_paths:
            train_imgs.append(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))
        self.initAndTrain(train_imgs,train_labels,clusterNum,bow_init_num=0)
        
    
    def detect(self,img):
        """
        判断一张图片的label
        参数img:要判断的图片
        返回值:(训练时输入的标签中的一个,评分)
        如果返回了(None,None)表示未能提取有效特征
        """
        bow=self.bow_extractor.compute(img,self.feature.detect(img))
        if bow is None:
            return None,None
        cls=self.svm.predict(bow)[1][0][0]
        score=self.svm.predict(
            bow,
            flags=cv2.ml.STAT_MODEL_RAW_OUTPUT | cv2.ml.STAT_MODEL_UPDATE_MODEL
            )[1][0][0]
        return cls,score

    def detectFile(self,img_path):
        """
        判断一张图片文件的label
        参数img_path:要判断的图片文件
        返回值:(训练时输入的标签中的一个,评分)
        """
        return self.detect(cv2.imread(img_path,cv2.IMREAD_GRAYSCALE))

def SURF_Flann_DetectorKernel():
    """基于SURF特征提取算法和Flann匹配算法的判别器"""
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)
    return DetectorKernel(
        cv2.xfeatures2d.SURF_create(),
        cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),{}),
        svm)

def SIFT_Flann_DetectorKernel():
    """基于SIFT特征提取算法和Flann匹配算法的判别器"""
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)
    return DetectorKernel(
        cv2.xfeatures2d.SIFT_create(),
        cv2.FlannBasedMatcher(dict(algorithm=1, trees=5),{}),
        svm)

'''
笔记1：
cv2.xfeatures2d.SURF_create()中的
detect(img)方法计算图像中的关键点位置，返回一keypoint列表
compute(img,keypoints)方法接受一张图和一keypoint列表返回每个关键点的描述符
detectAndCompute(img)等同于compute(img,detect(img))
并且同时返回keypoint列表和关键点的描述符

笔记2：
SURF/SIFT特征提取算法是基于局部特征的，算法对输入的图像尺寸没有要求
输入SVM中的参数是从图像中提取出的特征，和图像本身的尺寸无关
'''
