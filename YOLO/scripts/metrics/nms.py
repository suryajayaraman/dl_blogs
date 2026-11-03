import torch
import operator
from metrics.iou import intersectionOverUnion


def nonMaximalSuppression(bboxes:list, iouThreshold:float, 
                          probThreshold:float = 0.1, boxFormat:str = 'corners') -> list:
    """ Function uses non-maximal suppression to remove
    multiple bouding box instances of same object. It removes
    boxes whose IoU is more than threshold with another box
    for the same class with higher probability

    Args:
        bboxes (list): input list of bounding boxes
        iouThreshold (float): threshold to decide whether 2 boxes 
        are from same object
        probThreshold (float): minimum threshold for probability
        below which bounding boxes are discarded.
        boxFormat (str, optional): type of coordinate representation 
        (either `corners`  or `midpoint`). Defaults to 'corners'.
    
    Returns:
        list: boxes filterted after non-maximal suppression 
    """
    assert type(bboxes) == list, 'bboxes must be of type `list`'

    # initialize output
    filteredBoxes = []

    # filter boxes with pr(obj) > threshold and sort
    # based on confidence in bboxes
    bboxes = [box for box in bboxes if box[1] > probThreshold]
    bboxes.sort(key = operator.itemgetter(1), reverse=True)

    while bboxes:
        chosenBox = bboxes.pop(0)
        filteredBoxes.append(chosenBox)
        # select boxes which have different class
        # or have Iou < threshold for same class
        bboxes = [box for box in bboxes 
                 if box[0] != chosenBox[0] or
                 intersectionOverUnion(
                    torch.Tensor(box[2:]),
                    torch.Tensor(chosenBox[2:]),
                    boxFormat=boxFormat) < iouThreshold]

    return filteredBoxes