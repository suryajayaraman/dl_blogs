import cv2
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple, Any

import torch


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            if len(img.shape) > 3:
                img = zoom(np.float32(img), [
                           (t_s / i_s) for i_s, t_s in zip(img.shape, target_size[::-1])])
            else:
                img = cv2.resize(np.float32(img), target_size)

        result.append(img)
    result = np.float32(result)

    return result



def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


class BaseCAM:
    def __init__(self, model: torch.nn.Module, target_layers: List[torch.nn.Module],
                 reshape_transform: Callable = None) -> None:
        self.model = model
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """
    def get_cam_weights(self, input_data: Any, target_layers: List[torch.nn.Module], 
                        targets: List[torch.nn.Module], activations: torch.Tensor, grads: torch.Tensor,) -> np.ndarray:
        raise Exception("Not Implemented")


    def get_cam_image(self, input_data: Any, target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module], activations: torch.Tensor, 
                      grads: torch.Tensor, eigen_smooth: bool = False,) -> np.ndarray:
        
        weights = self.get_cam_weights(input_data, target_layer, targets, activations, grads)
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_data: Any, targets: List[torch.nn.Module], eigen_smooth: bool = False, key = '') -> np.ndarray:

        self.outputs = self.activations_and_grads(input_data)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_data, targets, eigen_smooth, key)
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input: Any, key : str = '') -> Tuple[int, int]:
        if(isinstance(input, dict)):
            input_data = input.get(key, None)
        else:
            input_data = input
        
        if(input_data is not None):
            if len(input_data.shape) == 4:
                width, height = input_data.size(-1), input_data.size(-2)
                return width, height
            elif len(input_data.shape) == 5:
                depth, width, height = input_data.size(-1), input_data.size(-2), input_data.size(-3)
                return depth, width, height
            else:
                raise ValueError("Invalid input_data shape. Only 2D or 3D images are supported.")
        else:
            raise ValueError("Nan Input")


    def compute_cam_per_layer(self, input: Any, targets: List[torch.nn.Module], eigen_smooth: bool, key : str = '') -> np.ndarray:
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input, key)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input, target_layer, targets, layer_activations, layer_grads, eigen_smooth)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def __call__(self, input_data: Any, targets: List[torch.nn.Module] = None, 
                 eigen_smooth: bool = False, key : str = '') -> np.ndarray:
        return self.forward(input_data, targets, eigen_smooth, key)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


class EigenCAM(BaseCAM):
    def __init__(self, model, target_layers, 
                 reshape_transform=None):
        super(EigenCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform)

    def get_cam_image(self,
                      input_data,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)


#######################################################


class PlannerTarget:
    def __init__(self, target_wp):
        self.target_wp = target_wp
        if torch.cuda.is_available():
            self.target_wp = self.target_wp.cuda()
        
    def __call__(self, model_output):
        _, debug_outputs = model_output
        l1_loss = torch.mean(torch.abs(torch.from_numpy(debug_outputs['pred_wp']) - self.target_wp))
        return l1_loss


if __name__ == "__main__":

    from config import GlobalConfig
    root_dir = '~/Downloads/transfuser-2022/data/demo/scenario1/'
    config = GlobalConfig()

    from data import CARLA_Data
    demo_set = CARLA_Data(root=root_dir, config=config, routeKey='route0', load_raw_lidar=True)
    print(f"There are {len(demo_set)} samples in Demo dataset")

    from torch.utils.data import DataLoader
    dataloader_demo = DataLoader(demo_set, shuffle=False, batch_size=2, num_workers=4)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    from model import LidarCenterNet
    model = LidarCenterNet(config, device, config.backbone, image_architecture='regnety_032', 
                            lidar_architecture='regnety_032', estimate_loss=False)
    model.to(device);
    model.config.debug = True
    checkpt = torch.load('~/Downloads/transfuser-2022/model_ckpt/transfuser/transfuser_regnet032_seed1_39.pth', map_location=device)
    model.load_state_dict(checkpt)
    model = model.eval();

    # which layers should be focus on
    target_layers = [model._model.image_encoder.features.layer4]


    with EigenCAM(model=model, target_layers=target_layers) as cam:
        from tqdm import tqdm

        frameIdx = 0
        for data in tqdm(dataloader_demo):
        
            # load data to gpu, according to type
            for k in ['rgb', 'depth', 'lidar', 'label', 'ego_waypoint', \
                        'target_point', 'target_point_image', 'speed']:
                data[k] = data[k].to(device, torch.float32)
            for k in ['semantic', 'bev']:
                data[k] = data[k].to(device, torch.long)

            targets = [PlannerTarget(data['ego_waypoint'])]
            grayscale_cam = cam(input_data=data, targets=targets, key='rgb')
            outputs = cam.outputs[1]
            bs = data['rgb'].shape[0]

            for i in range(bs):
                rgb_image = data['rgb'][i].permute(1, 2, 0).detach().cpu().numpy()
                rgb_image = rgb_image / 255.0
                grayscale_cam_i = grayscale_cam[i]
                cam_image = show_cam_on_image(rgb_image, grayscale_cam_i, use_rgb=True)

                combined_image = np.vstack(( np.uint8(rgb_image *255), cam_image))
                plt.imsave(f'eigen_cam_test_{frameIdx}.png', combined_image)
                frameIdx +=1