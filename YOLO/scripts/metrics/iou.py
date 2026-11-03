import torch


def intersectionOverUnion(box1:torch.Tensor, box2:torch.Tensor, boxFormat:str = 'midpoint', 
    threshold:float=1e-6) -> torch.Tensor:
    """Function calculates Intersection Over union for the input
    bounding boxes based on input format (corners / midpoint)

    Args:
        box1 (torch.Tensor): (N,4) tensor containing coordinates of 1st bounding box
        box2 (torch.Tensor): (N,4) tensor containing coordinates of 2nd bounding box
        boxFormat (str, optional): type of coordinate representation 
        (either `corners`  or `midpoint`). Defaults to 'corners'.
        threshold (float, optional): zero division error threshold. Defaults to 1e-6.

    Returns:
        iou (torch.Tensor) : (N,1) tensor containing metric for corresponding boxes
    """
    if(boxFormat == 'corners'):
        # direct corner points (x1, y1, x2, y2) format
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]
        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]
        
    elif(boxFormat == 'midpoint'):
        # direct corner points (x_center, y_centre, width, height) format
        box1_x1 = box1[..., 0:1]  - (box1[..., 2:3] * 0.5)
        box1_y1 = box1[..., 1:2]  - (box1[..., 3:4] * 0.5)
        box1_x2 = box1[..., 0:1]  + (box1[..., 2:3] * 0.5)
        box1_y2 = box1[..., 1:2]  + (box1[..., 3:4] * 0.5)
        
        box2_x1 = box2[..., 0:1]  - (box2[..., 2:3] * 0.5)
        box2_y1 = box2[..., 1:2]  - (box2[..., 3:4] * 0.5)
        box2_x2 = box2[..., 0:1]  + (box2[..., 2:3] * 0.5)
        box2_y2 = box2[..., 1:2]  + (box2[..., 3:4] * 0.5)        
    
    # corner points of intersection
    xminIntersection = torch.max(box1_x1, box2_x1)
    yminIntersection = torch.max(box1_y1, box2_y1)
    xmaxIntersection = torch.min(box1_x2, box2_x2)
    ymaxIntersection = torch.min(box1_y2, box2_y2)
    
    # clamp(0) is for case when boxes dont intersect
    intersectionArea = torch.abs( (xmaxIntersection - xminIntersection).clamp(0) * 
                                  (ymaxIntersection - yminIntersection).clamp(0) )
    box1Area = torch.abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2Area = torch.abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))
    
    # iou = intersection / union, threshold is to avoid division by zero error
    iou = intersectionArea / (box1Area + box2Area - intersectionArea + threshold)
    return iou