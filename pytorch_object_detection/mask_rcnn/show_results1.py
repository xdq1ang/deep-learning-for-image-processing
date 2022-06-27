from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision
import numpy as np
import cv2
import random
import os
from backbone import resnet50_fpn_backbone
from network_files import MaskRCNN
import torch
from my_dataset_voc import VOCInstances
import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 验证数据集
data_transform = {
    "val": transforms.Compose([transforms.ToTensor()])
}
val_dataset = VOCInstances("data/VOCdevkit", year="2012", txt_name="val.txt", transforms=data_transform["val"])
val_dataset_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=1,
                                                 collate_fn=val_dataset.collate_fn)

# 加载模型
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
backbone = resnet50_fpn_backbone()
model = MaskRCNN(backbone, num_classes=20 + 1)

# 载入你自己训练好的模型权重
weights_path = "save_weights/model_25.pth"
model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
model.eval()
# 标签
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
PASCAL_VOC_INSTANCE_CATEGORY_NAMES = [
    '__background__', "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_prediction(pred, threshold):  # 定义模型，并根据阈值过滤结果
    pred_score = list(pred[0]['scores'].detach().numpy())
    try: 
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    except IndexError as e:
        print("未检测到任何对象！")
        return [],[],[],[]

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [PASCAL_VOC_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    pred_score = pred_score[:pred_t + 1]
    return masks, pred_boxes, pred_class, pred_score


def random_colour_masks(image):
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    randcol = colours[random.randrange(0, 10)]
    r[image == 1] = randcol[0]
    g[image == 1] = randcol[1]
    b[image == 1] = randcol[2]
    coloured_mask = np.stack([r, g, b], axis=2)
    print("randcol", randcol)
    return coloured_mask, randcol


def instance_segmentation_api(save_root, val_dataset_loader, threshold=0.5, rect_th=3, text_size=0.5, text_th=1):  # 进行目标检测
    with torch.no_grad():
        for image, targets in tqdm(val_dataset_loader, desc="validation..."):
            outputs = model(image)
            img = np.array(ToPILImage()(image[0]))
            masks, boxes, pred_cls, pred_score = get_prediction(outputs, threshold)  # 调用模型
            for i in range(len(masks)):
                rgb_mask, randcol = random_colour_masks(masks[i])  # 使用随机颜色为模型的掩码区进行填充。
                img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                # 元组里面有小数，需要转化为整数 否则报错
                T1, T2 = boxes[i][0], boxes[i][1]
                x1 = int(T1[0])
                y1 = int(T1[1])
                x2 = int(T2[0])
                y2 = int(T2[1])
                cv2.rectangle(img, (x1, y1), (x2, y2), color=randcol, thickness=rect_th)
                # # putText各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色黑，字体粗细
                cv2.putText(img, pred_cls[i] + " " + str(pred_score[i]), (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, text_size,
                            randcol, thickness=text_th)
                
            
            save_name = os.path.join(save_root, str(len(os.listdir(save_root))+1)+".png")
            cv2.imwrite(save_name,img)



# 显示模型结果
if __name__ == "__main__":
    save_root = "show_results"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    instance_segmentation_api(save_root, val_dataset_loader)
