# basic imports
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# DL library imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# helper functions
from metrics.nms import nonMaximalSuppression
from dataset.vocDataset import VOC_COLOR_MAP, STD_INVERSE_TRANSFORM, VOC_CLASSES

# interactive plots
import plotly.graph_objects as go
from plotly.subplots import make_subplots


###################################
# FUNCTION TO EVALUATE MODEL ON DATALOADER
###################################

def evaluate_model(model:nn.Module, dataloader:DataLoader, criterion, num_classes:int, device:torch.device, metric_class=None,**kwargs):
    model.eval()
    metaData = {}
    metaData['total_loss'] = 0.0
#     t = tqdm(dataloader, desc='Val total loss', total=len(dataloader), leave=True)
    t = tqdm(dataloader, total=len(dataloader), leave=True)

    if(metric_class is not None):
        metric_object = metric_class(num_classes, **kwargs)
    else:
        metric_object = None
        metaData['metric'] = None

    with torch.no_grad():
        for inputs, labels in t:
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            metaData['total_loss'] += loss.item()
#             t.set_description(f"Val total loss {metaData['total_loss']:7.2f}", refresh=True)

            # update batch metric information
            if(metric_object is not None):
                metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())
            else:
                pass   
                    
    metaData['mean_loss'] = metaData['total_loss'] / len(dataloader)
    if(metric_object is not None):
        metaData['metric'] = metric_object.compute()
    return metaData


###################################
# FUNCTION TO TRAIN MODEL ON DATALOADER
###################################

def train_model(model:nn.Module, dataloader:DataLoader, criterion, optimizer, lr_scheduler, num_classes:int, 
                device:torch.device, metric_class=None,**kwargs):
    metaData = {}
    metaData['total_loss'] = 0.0
    if(metric_class is not None):
        metric_object = metric_class(num_classes, **kwargs)
    else:
        metric_object = None
        metaData['metric'] = None

    # Training
    model.train()
#     t = tqdm(dataloader, desc='Train total loss', total=len(dataloader), leave=True)
    t = tqdm(dataloader, total=len(dataloader), leave=True)
    for inputs, labels in t:
        inputs = inputs.to(device)
        labels = labels.to(device) 

        # Forward pass
        y_preds = model(inputs)
        loss = criterion(y_preds, labels)
        metaData['total_loss'] += loss.item()
#         t.set_description(f"Train total loss {metaData['total_loss']:7.2f}", refresh=True)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # update batch metric information
        if(metric_object is not None):
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())
        else:
            pass         

    metaData['mean_loss'] = metaData['total_loss'] / len(dataloader)
    if(metric_object is not None):
        metaData['metric'] = metric_object.compute()
    return metaData  


###################################
# FUNCTIONS TO CONVERT FROM CELL BOXES TO IMAGE BOXES
###################################

def convertCellboxes(predictions:torch.Tensor, S:int=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellBoxesToBoxes(out:torch.Tensor, S:int=7) -> list:
    convertedPred = convertCellboxes(out).reshape(out.shape[0], S * S, -1)
    convertedPred[..., 0] = convertedPred[..., 0].long()
    allBBoxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in convertedPred[ex_idx, bbox_idx, :]])
        allBBoxes.append(bboxes)

    return allBBoxes


def getBoundingBoxes(loader:DataLoader, model:nn.Module, iouThreshold:float, 
                    threshold:float, boxFormat:str="midpoint", device="cuda"):
    # initialize return variable
    allPredBoxes = []
    allTrueBoxes = []

    # unique index for each image
    trainIdx = 0    

    # make sure model is in eval before get bboxes
    model.eval()
    
    for _, (x, labels) in enumerate(loader):
        # make model predictions
        # x shape = labels shape = (N,S,S,B*5 +C)
        x = x.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            predictions = model(x)
        
        # trueBBoxes and bBoxes is in [[[class, prob, x, y, w, h],[]], ...] format
        # Both are list of length bs, each element containing bounding boxes
        # in that sample. A bounding box is represented by 6 element list
        trueBBoxes = cellBoxesToBoxes(labels)
        bBoxes = cellBoxesToBoxes(predictions)

        # iterate through each sample
        for idx in range(x.shape[0]):
            # run predicted boxes through NMS to get filtered boxes
            # and append to total prediction data
            nmsBoxes = nonMaximalSuppression(bBoxes[idx],
                iouThreshold=iouThreshold, probThreshold=threshold, boxFormat=boxFormat)
            for nmsBox in nmsBoxes:
                allPredBoxes.append([trainIdx] + nmsBox)

            # Filter only labels having any object 
            for box in trueBBoxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    allTrueBoxes.append([trainIdx] + box)

            trainIdx += 1

    model.train()
    return allPredBoxes, allTrueBoxes



def plotTrainingResults(df, model_name):
    fig, ax1 = plt.subplots(figsize=(10,4))
    ax1.set_ylabel('trainLoss', color='tab:red')
    ax1.plot(df['epoch'].values, df['trainLoss'].values, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    if(np.isinf(df['validationLoss']).values.sum() ==0):
        ax2 = ax1.twinx()  
        ax2.set_ylabel('validationLoss', color='tab:blue')
        ax2.plot(df['epoch'].values, df['validationLoss'].values, color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.grid(True)
    plt.suptitle(f'{model_name} Training, Validation Curves')
    plt.show()


###################################
# FUNCTIONS TO SAVE AND LOAD FROM CHECKPOINTS
###################################

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


###################################
# FUNCTION TO PLOT IMAGES AND BBOXES
###################################

def plotBBoxesInImage(image, boxes, ax, **kwargs):
    """Plots predicted bounding boxes on the image"""
    plotObj = []
    
    # plot rgb image
    im = np.array(image)
    height, width, _ = im.shape
    plotObj.append(ax.imshow(im))
    
    # Create a Rectangle potch
    for box in boxes:
        # box = [classId, confidence, x midpoint, y midpoint, width, height]
        upper_left_x = box[2] - box[4] / 2
        upper_left_y = box[3] - box[5] / 2
        classColor = VOC_COLOR_MAP[int(box[0])] / 255.0
        className = VOC_CLASSES[int(box[0])].name
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[4] * width, box[5] * height, edgecolor= classColor, facecolor="none", **kwargs)
        
        # Add the patch to the Axes and class name text
        plotObj.append(ax.add_patch(rect))
        ax.text(upper_left_x* width, (upper_left_y *height) - 10, className, color=classColor)

    ax.set_xticks([])
    ax.set_yticks([])
    return plotObj


def plotDatasetSamples(dataset:torch.utils.data.Dataset, numSamples:int):
    randomIndices = np.random.choice(len(dataset), numSamples).tolist()
    nRows = np.ceil(numSamples/3).astype(int)
    fig, axes = plt.subplots(nRows, 3, figsize=(3*3.5, nRows * 4))
    axes = axes.flatten()

    for i in range(len(randomIndices)):
        sampleImage, sampleBoxes = dataset[randomIndices[i]] 
        rgbImage = STD_INVERSE_TRANSFORM(sampleImage).permute(1, 2, 0).cpu().detach().numpy()
        bboxes = cellBoxesToBoxes(sampleBoxes.unsqueeze(0))
        bboxes = nonMaximalSuppression(bboxes[0], iouThreshold=0.5, probThreshold=0.4, boxFormat="midpoint")
        _ = plotBBoxesInImage(rgbImage, bboxes, axes[i], linewidth=3)
    
    for idx in range(numSamples, len(axes)):
        fig.delaxes(axes[idx])
    return    


def visualizeModelPredictions(model:nn.Module, dataset:torch.utils.data.Dataset, device:torch.device, numSamples:int):
    randomIndices = np.random.choice(len(dataset), numSamples).tolist()
    nRows = np.ceil(numSamples/3).astype(int)
    fig, axes = plt.subplots(nRows, 3, figsize=(3*3.5, nRows * 4))
    axes = axes.flatten()
    
    # set model to eval mode
    model.eval()
    model.to(device);
    
    for i in range(len(randomIndices)):
        # pass input through model to get predictions
        inputImage, _ = dataset[randomIndices[i]] 
        inputImage = inputImage.to(device);
        with torch.no_grad():
            modelPreds = model(inputImage.unsqueeze(0))

        # inverse transform to get rgb image
        rgbImage = STD_INVERSE_TRANSFORM(inputImage).permute(1, 2, 0).cpu().detach().numpy()

        # convert from cell to image coordinates
        bboxes = cellBoxesToBoxes(modelPreds.unsqueeze(0))
        bboxes = nonMaximalSuppression(bboxes[0], iouThreshold=0.5, probThreshold=0.4, boxFormat="midpoint")
        _ = plotBBoxesInImage(rgbImage, bboxes, axes[i], linewidth=3)
    
    for idx in range(numSamples, len(axes)):
        fig.delaxes(axes[idx])
    return          


###################################
# FUNCTION TO FIND LEARNING RATE
###################################

def lrFinder(model, optimizer, dataloader, criterion, device, initValue = 1e-8, finalValue=100.0, beta = 0.98, numBatches = 200):
    # make deep copies of model, optimizer 
    # to keep original unimpacted
    testModel = deepcopy(model)
    testModel.to(device)
    testOptimizer = deepcopy(optimizer)

    assert(numBatches > 0)
    mult = (finalValue / initValue) ** (1/numBatches)
    lr = initValue
    testOptimizer.param_groups[0]['lr'] = lr
    batch_num = 0
    avgLoss = 0.0
    bestLoss = 0.0
    smoothLosses = []
    rawLosses = []
    log_lrs = []
    dataloader_it = iter(dataloader)
    progress_bar = tqdm(range(numBatches))                
        
    for _ in progress_bar:
        batch_num += 1
        try:
            images, labels = next(dataloader_it)
            #print(images.shape)
        except:
            dataloader_it = iter(dataloader)
            images, labels = next(dataloader_it)

        # Move input and label tensors to the default device
        images = images.to(device)
        labels = labels.to(device)

        # handle exception in criterion
        try:
            # Forward pass
            y_preds = testModel(images.float())
            loss = criterion(y_preds, labels)
        except:
            if len(smoothLosses) > 1:
                grad_loss = np.gradient(smoothLosses)
            else:
                grad_loss = 0.0
            lr_finder_results = {'log_lr':log_lrs, 'raw_loss':rawLosses, 
                                 'smooth_loss':smoothLosses, 'grad_loss': grad_loss}
            return lr_finder_results 
                    
        #Compute the smoothed loss
        avgLoss = beta * avgLoss + (1-beta) *loss.item()
        smoothed_loss = avgLoss / (1 - beta**batch_num)
        
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 50 * bestLoss:
            if len(smoothLosses) > 1:
                grad_loss = np.gradient(smoothLosses)
            else:
                grad_loss = 0.0
            lr_finder_results = {'log_lr':log_lrs, 'raw_loss':rawLosses, 
                                 'smooth_loss':smoothLosses, 'grad_loss': grad_loss}
            return lr_finder_results
        
        #Record the best loss
        if smoothed_loss < bestLoss or batch_num==1:
            bestLoss = smoothed_loss
        
        #Store the values
        rawLosses.append(loss.item())
        smoothLosses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        
        # Backward pass
        testOptimizer.zero_grad()
        loss.backward()
        testOptimizer.step()
        
        # print info
        progress_bar.set_description(f"loss: {loss.item()},smoothed_loss: {smoothed_loss},lr : {lr}")

        #Update the lr for the next step
        lr *= mult
        testOptimizer.param_groups[0]['lr'] = lr
    
    grad_loss = np.gradient(smoothLosses)
    lr_finder_results = {'log_lr':log_lrs, 'raw_loss':rawLosses, 
                         'smooth_loss':smoothLosses, 'grad_loss': grad_loss}
    return lr_finder_results


def plotLrFinderResults(lr_finder, modelName='', saveFile=False): 
    # Create subplot grid
    fig = make_subplots(rows=1, cols=2)
    
    # Create a line (trace) for the lr vs loss, gradient of loss
    trace0 = go.Scatter(x=lr_finder['log_lr'], y=lr_finder['smooth_loss'],name='log_lr vs smooth_loss')
    trace1 = go.Scatter(x=lr_finder['log_lr'], y=lr_finder['grad_loss'],name='log_lr vs loss gradient')

    # Add subplot trace & assign to each grid
    fig.add_trace(trace0, row=1, col=1);
    fig.add_trace(trace1, row=1, col=2);
    
    if(saveFile==True):
        fig.write_html(modelName + 'LrFinderResult.html')
    fig.update_layout({'title' : 'Lr_finder_result'})
    fig.show()    