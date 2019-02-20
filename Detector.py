"""
图像金字塔+滑动窗口判别器
使用方法:先用DetectorKernel训练好一个DetectorKernel类
->指定:图像金字塔需要哪些缩放值,滑动窗口尺寸,滑动窗口步长
->初始化一个判别器Detector
->图像喂给判别器的detect_results方法，判别器吐出一个迭代器
->迭代器中的数据就是图像金字塔中所有滑动窗口的判别结果
"""
import cv2
import numpy as np
from non_maximum import non_max_suppression_fast
class Detector:
    """图像金字塔+滑动窗口判别器"""
    def __init__(self,detector_kernel,scales,shape,step_size):
        """
        参数detector_kernel:一个训练完成的DetectorKernel作为判别器
        参数scales:放大倍数列表，小于1就是缩小
        参数shape:滑动窗口长宽
        参数step_size:滑动窗口步长
        """
        self.detector_kernel=detector_kernel
        self.scales=scales
        self.shape=shape
        self.step_size=step_size

    """
    图像金字塔迭代器
    参数img:原图
    参数scales:放大倍数列表，小于1就是缩小
    返回值:(放大(缩小)倍数,放大(缩小)后的图片)
    """
    @staticmethod
    def pyramids(img,scales):
        for scale in scales:
            yield (scale,
                   cv2.resize(
                       img,(
                           int(img.shape[1] * scale),
                           int(img.shape[0] * scale)),
                       interpolation=cv2.INTER_AREA))

    """
    滑动窗口迭代器
    参数img:原图
    参数shape:滑动窗口长宽
    参数step_size:滑动窗口步长
    返回值:(滑动窗口位置,滑动窗口内的图像)
    """
    @staticmethod
    def sliding_windows(img,shape,step_size):
        for i in range(0,img.shape[0]-shape[1],step_size):
            for j in range(0,img.shape[1]-shape[0],step_size):
                yield ((j,i),img[i:i+shape[1],j:j+shape[0]])
            
    """
    图像金字塔+滑动窗口判别结果迭代器
    参数img:原图
    返回值:(滑动窗口在原图中的位置,滑动窗口在原图中的大小,滑动窗口内的图像判别结果)
    """
    def detect_results_generator(self,img):
        for scale,image in self.pyramids(img,self.scales):
            #滑动窗口在原图中的大小↓
            shape_slide=(int(self.shape[0]/scale),int(self.shape[1]/scale))
            for place,slide in Detector.sliding_windows(image,self.shape,self.step_size):
                place_slide=(int(place[0]/scale),int(place[1]/scale))
                cls,score=self.detector_kernel.detect(slide)
                if cls is None:
                    continue
                yield (place_slide,shape_slide,(cls,score))

    """
    带最大值抑制的图像金字塔+滑动窗口判别结果
    参数img:原图
    参数overlapThresh:最大值抑制重叠率阈值
    参数useless_labels:有哪些判断结果是没用的要在结果中去掉
    返回值:boxes_dict={滑动窗口识别物标签:[[滑动窗口位置,滑动窗口大小,滑动窗口结果评分],]}
    """
    def detect_non_max(self,img,overlapThresh,useless_labels=[]):
        boxes_dict={}
        for box_place,box_shape,(box_cls,box_score) in self.detect_results_generator(img):
            if box_cls in useless_labels:
                continue#去除无用项
            box=[box_place[0],
                 box_place[1],
                 box_place[0]+box_shape[0],
                 box_place[1]+box_shape[1],
                 abs(box_score)]
            if not box_cls in boxes_dict:
                boxes_dict[box_cls]=[]
            boxes_dict[box_cls].append(box)
        for key in boxes_dict:
            boxes=non_max_suppression_fast(np.array(boxes_dict[key]),overlapThresh)
            boxes_dict[key]=[]
            for x1,y1,x2,y2,score in boxes:
                boxes_dict[key].append([(int(x1),int(y1)),
                                        (int(x2-x1),int(y2-y1)),
                                        score])
        return boxes_dict
"""
笔记：
opencv读的图像img.shape=(宽,长)
但是opencv图像操作函数中的参数总是(长,宽)
"""
