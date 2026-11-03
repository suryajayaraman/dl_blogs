import os, sys
import torch
import torch.nn as nn

####################################################
###### Add scripts folder to sys.path for import ###
def addScriptsFolderToPath() -> None:
    scriptsFolder = os.path.join( os.path.dirname(os.getcwd()), 'scripts')
    sys.path.append(scriptsFolder)
addScriptsFolderToPath()
####################################################

from metrics.iou import intersectionOverUnion


class yoloV1Loss(nn.Module):
    def __init__(self, S:int, B:int, C:int) -> None:
        super(yoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.mseLoss = nn.MSELoss(reduction="sum")

        # constants
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5


    def forward(self, predictions:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        # (2, 1470) -> (2,7,7,30)
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)

        # predictions[..., 21:25] shape = (2,7,7,4)
        # target[..., 21:25] shape = (2,7,7,4)
        # iouBox1 shape = (2,1)
        # |||ly iouBox2 shape = (2,1)
        iouBox1 = intersectionOverUnion(predictions[..., 21:25], target[..., 21:25])
        iouBox2 = intersectionOverUnion(predictions[..., 26:30], target[..., 21:25])

        # ious shape = (2,2,1). each channel(1st dim) represents a bbox predictor
        # in this case, there are only 2 predictors per grid cell
        # hence only 2 channels, if we opted for 5 predictors, then 5 channels
        ious = torch.cat([iouBox1.unsqueeze(0), iouBox2.unsqueeze(0)], dim=0)

        # iouMaxes shape = (2,1) containing the max iou value 
        # for each example. bestBox contains the index of max value
        # bestBox shape = (2,1)
        _, bestBox = torch.max(ious, dim=0)

        # objExists shape = (2,7,7,1). identity function  
        # representing whether object exists in that grid cell 
        objExists = target[..., 20].unsqueeze(3)


        #############################
        ### BOX COORDINATES LOSS  ###
        #############################

        # boxPredictions shape = (2,7,7,4). boxPredictions contains the
        # coordinates of the predictor with max iou for the grid cell
        # if the object exists in target. boxTargets shape = (2,7,7,4)
        # contains the target coordinates if object exists
        boxPredictions = objExists * (bestBox * predictions[..., 26:30] + 
                                      (1-bestBox) * predictions[..., 21:25])
        boxTargets = objExists * target[..., 21:25]

        # square root of the width and height
        boxPredictions[..., 2:4] = torch.sign(boxPredictions[...,2:4]) * torch.sqrt(torch.abs(boxPredictions[...,2:4] + 1e-6))
        boxTargets[..., 2:4] = torch.sqrt(boxTargets[...,2:4])

        # reshaping boxPredictions and boxTargets from (2,7,7,4) to (2*7*7,4)
        # before sending to mse loss fn
        boxLoss = self.mseLoss(torch.flatten(boxPredictions, end_dim=-2), 
                            torch.flatten(boxTargets, end_dim=-2))

        #############################
        ###       OBJECT LOSS     ###
        #############################
        
        # objPredictions shape = (2,7,7,1). identify which predictor
        # is responsible to prediction and include that confidence score
        # in MSE loss calculation
        objPredictions = ((bestBox * predictions[..., 25:26]) + 
                          (1-bestBox) * predictions[..., 20:21])
        objloss = self.mseLoss(torch.flatten(objExists * objPredictions), 
                            torch.flatten(objExists * target[..., 20:21]))

        #############################
        ###    NO OBJECT LOSS     ###
        #############################

        # no object loss accounts for the case to minimize predictors
        # confidence score when no object exists actually in that grid cell
        # account for both the predictors when calculating the no object loss
        # noObjectLoss shape = (2,7,7,1)
        noObjLoss = self.mseLoss(
                    torch.flatten((1-objExists) * predictions[..., 20:21], start_dim=1),
                    torch.flatten((1-objExists) * target[..., 20:21], start_dim=1)
                )

        noObjLoss += self.mseLoss(
                    torch.flatten((1-objExists) * predictions[..., 25:26], start_dim=1),
                    torch.flatten((1-objExists) * target[..., 20:21], start_dim=1)
                )

        #############################
        ###  CLASSIFICATION LOSS  ###
        #############################

        # class predictions shape = (2,7,7,20) -> reshaped to (2*7*7,20)
        classificationLoss = self.mseLoss(
                    torch.flatten(objExists * predictions[..., :20], end_dim=-2),
                    torch.flatten(objExists * target[..., :20], end_dim=-2)
                )

        #############################
        ###       TOTAL LOSS      ###
        #############################
        
        loss = (self.lambda_coord * boxLoss) + \
                objloss + \
               (self.lambda_noobj * noObjLoss) + \
                classificationLoss

        return loss
