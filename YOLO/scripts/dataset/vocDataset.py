# basic imports
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from collections import namedtuple

# DL library imports
from torchvision import transforms


###################################
# FILE CONSTANTS
###################################
STD_IMG_TRANSFORM = [
                    transforms.Resize((448, 448)), 
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
                ]


# when using torch datasets we defined earlier, the output image
# is normalized. So we're defining an inverse transformation to 
# transform to normal RGB format
STD_INVERSE_TRANSFORM = transforms.Compose([
        transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])


# Constants for Standard PASCAL VOC Dataset color mapping
# reference : https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
Label = namedtuple( "Label", [ "name", "color"])
VOC_CLASSES = [ 
                Label("aeroplane", (0,0,0)),
                Label("bicycle", (128,0,0)),
                Label("bird", (0,128,0)),
                Label("boat", (128,128,0)),
                Label("bottle", (0,0,128)),
                Label("bus", (128,0,128)),
                Label("car", (0,128,128)),
                Label("cat", (128,128,128)),
                Label("chair", (64,0,0)),
                Label("cow", (192,0,0)),
                Label("diningtable", (64,128,0)),
                Label("dog", (192,128,0)),
                Label("horse", (64,0,128)),
                Label("motorbike", (192,0,128)),
                Label("person", (64,128,128)),
                Label("pottedplant", (192,128,128)),
                Label("sheep", (0,64,0)),
                Label("sofa", (128,64,0)),
                Label("train", (0,192,0)),
                Label("tvmonitor", (128,192,0))
            ]
VOC_COLOR_MAP = [c.color for c in VOC_CLASSES]
VOC_COLOR_MAP = np.array(VOC_COLOR_MAP)



#####################################
### transformComponse CLASS DEFINITION ##
#####################################

class transformComponse(object):
    def __init__(self, transforms:list):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes


#####################################
### vocDataset CLASS DEFINITION ##
#####################################

class vocDataset(torch.utils.data.Dataset):
    def __init__(self, annotationFile:str, imgDir:str, labelsDir:str, 
                 S:int, B:int, C:int, transform=None) -> None:
        super(vocDataset, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.imgDir = imgDir
        self.labelsDir = labelsDir
        self.transform = transform
        self.annotationData : pd.DataFrame = pd.read_csv(annotationFile)        

    def __getitem__(self, index):
        # get image data
        imgPath = os.path.join(self.imgDir, self.annotationData.iloc[index, 0])
        image = Image.open(imgPath)

        # read label data
        # parse label data from text file
        # creating list of target bounding boxes
        # convert to tensor for torch transformations
        boxes = []
        labelPath = os.path.join(self.labelsDir, self.annotationData.iloc[index, 1])
        with open(labelPath, 'r') as f:
            for lineData in f.readlines():
                classId, x, y, w, h = [float(x) if float(x) != int(float(x)) else int(x) \
                            for x in lineData.replace("\n", "").split()]
                boxes.append([classId, x, y, w, h])
        boxes = torch.Tensor(boxes)

        if(self.transform is not None):
            image, boxes = self.transform(image, boxes)

        # initialize label data
        labels = torch.zeros(self.S, self.S, self.B * 5 + self.C)
        for box in boxes:
            # all the values are in image scale, relative
            # to top left corner [0,1] ratio
            classId, xImg, yImg, wImg, hImg = box.tolist()
            classId = int(classId)

            # i refers to cell index along y (vertical) axis
            # j refers to cell index along x (horizontal) axis
            # xCell and yCell are relative to cell top left corner
            i,j = int(self.S * yImg), int(self.S * xImg)
            xCell = (self.S * xImg) - j   # column (x-axis)
            yCell = (self.S * yImg) - i   # row (y-axis)
            
            # Example : consider image consisting of 10x10 grid cells
            # let image size = 100x100 pixel -> each grid is 10x10
            # if a bbox is (20x20), then wImg = 20/100 = 0.2
            # wCell = 20/10 = 2. 
            wCell, hCell = wImg * self.S, hImg * self.S 

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object per cell!
            if(labels[i,j,20] == 0):
                labels[i,j, 20] = 1
                labels[i,j, classId] = 1
                boxCoordinates = torch.Tensor([xCell, yCell, wCell, hCell])
                labels[i,j, 21:25] = boxCoordinates

        return image, labels

    def __len__(self):
        return len(self.annotationData)