import torch
import operator
from collections import Counter
from metrics.iou import intersectionOverUnion


def meanAveragePrecision(predBoxes: list, targetBoxes:list, iouThreshold:float, 
                    boxFormat:str = 'corners', numClasses:int = 20) -> float:
    """ Function calculates mean average precision metric b/w
    predicted and ground truth boxes based on input IoU limit.
    Average precision is calculated for each class as area under
    Precision-Recall curve and is averaged over all classes.

    Args:
        predBoxes (list): list of tensors containing predicted bounding
        box data in format [train_idx, predicted_class, prob_score, x1,y1,x2,y2]
        targetBoxes (list): list of tensors containing ground truth bounding
        box data in format [train_idx, predicted_class, prob_score, x1,y1,x2,y2]
        iouThreshold (float): IoU limit b/w predicted and target bbox to 
        consider a True Positive
        boxFormat (str, optional):type of coordinate representation 
        (either `corners`  or `midpoint`). Defaults to 'corners'.
        numClasses (int, optional): number of classes in dataset. Defaults to 20.

    Returns:
        float: mAP metric
    """
    assert type(predBoxes) == list, 'predBoxes must be of type `list`'
    assert type(targetBoxes) == list, 'targetBoxes must be of type `list`'

    epsilon = 1e-6
    averagePrecision = []

    # iterate over each class
    for classId in range(numClasses):
        # filter by class
        detections = [box for box in predBoxes if box[1] == classId]
        groundTruths = [box for box in targetBoxes if box[1] == classId]

        # ensure there's atleast 1 bbox for the class
        if(len(groundTruths) == 0):
            continue

        # img1 has 3 boxes
        # img2 has 5 boxes
        # boxCounter = {0:3, 1:5}
        boxCounter = Counter([gt[0] for gt in groundTruths])

        for k,v in boxCounter.items():
            # boxCounter =  {0 : torch.Tensor([0,0,0]), 1 : torch.Tensor([0,0,0,0,0])}
            boxCounter[k] = torch.zeros(v)

        detections.sort(key = operator.itemgetter(2), reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        for detIdx, detData in enumerate(detections):
            # initialize best iou to the min value
            bestIou = 0.0

            # shortlist groundtruth boxes for particular image
            imgIdx = detData[0]
            imgGtLabels = [box for box in groundTruths if box[0] == imgIdx]

            for gtIndex, gtData in enumerate(imgGtLabels):
                iou = intersectionOverUnion(torch.Tensor(gtData[3:]), torch.Tensor(detData[3:]), 
                                         boxFormat=boxFormat)
                if(iou > bestIou):
                    bestIou = iou
                    bestGtIndex = gtIndex

            # prediction has overlap with Gt box 
            if(bestIou > iouThreshold):
                # Gt image is not yet assigned to any prediction yet
                if(boxCounter[imgIdx][bestGtIndex] == 0):
                    TP[detIdx] = 1
                    boxCounter[imgIdx][bestGtIndex] = 1
                else:
                    FP[detIdx] = 1
                
            # not enough overlay with Gt box
            else:
                FP[detIdx] = 1
                
        # calculate metric
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)    
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recall = TP_cumsum / (len(groundTruths) + epsilon)
        precision = torch.cat((torch.Tensor([1]), precision))
        recall = torch.cat((torch.Tensor([0]), recall))

        # AUC is calculated using numerical integrating
        # the Precision-Recall curve
        averagePrecision.append(torch.trapz(precision, recall))
    
    return sum(averagePrecision) / len(averagePrecision)


