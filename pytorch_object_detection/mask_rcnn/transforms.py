import random
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
        return image, target





class Resize(object):
    def __init__(self, nh, nw):
        self.nw = nw
        self.nh = nh 

    def __call__(self, image, target):
        """
        修改 box
        :param filename_jpg: 图片名
        :param box: 原box
        :param nw: 改变后的宽度
        :param nh: 改变后的高度
        :return:
        """
        c, ih, iw = image.shape        
        if ih > 256 or iw >512:
            # 对图像进行缩放并且进行长和宽的扭曲
            image = transforms.Resize((self.nh, self.nw))(image)
            target["masks"] = transforms.Resize((self.nh,self.nw),Image.NEAREST)(target["masks"])
            # 将box进行调整
            for boxx in target["boxes"]:
                boxx[0] = int(int(boxx[0]) * (self.nw / iw))
                boxx[1] = int(int(boxx[1]) * (self.nh / ih))
                boxx[2] = int(int(boxx[2]) * (self.nw / iw))
                boxx[3] = int(int(boxx[3]) * (self.nh / ih))

        return image, target