# basic imports
import base64
import numpy as np
from PIL import Image
from io import BytesIO

# DL imports
import torch

# plot library imports
import plotly.graph_objects as go


###################################
# FILE CONSTANTS
###################################

CAMERA_HEIGHT = 2.3   # camera height from ground in meters
LIDAR_HEIGHT = 2.5    # LIDAR height from ground in meters
COORDINATE_AXIS = ['x', 'y', 'z']
VEHICLE_TO_LIDAR_FWD = 1.3  # distance from vehicle origin to lidar in forward direction

INDICES_1 = [5,4,0,1]
INDICES_2 = [6,5,1,2]
INDICES_3 = [7,6,2,3]
INDICES_4 = [4,7,3,0]


###################################
# FUNCTION DEFINITIONS  #
###################################

def get_virtual_lidar_to_vehicle_transform():
    # This is a fake lidar coordinate
    T = np.eye(4)
    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T
        
def get_vehicle_to_virtual_lidar_transform():
    return np.linalg.inv(get_virtual_lidar_to_vehicle_transform())

def get_lidar_to_vehicle_transform():
    rot = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]], dtype=np.float32)
    T = np.eye(4)
    T[:3, :3] = rot

    T[0, 3] = 1.3
    T[1, 3] = 0.0
    T[2, 3] = 2.5
    return T

def get_vehicle_to_lidar_transform():
    return np.linalg.inv(get_lidar_to_vehicle_transform())

def get_lidar_to_bevimage_transform():
    # rot 
    T = np.array([[0, -1, 16],
                  [-1, 0, 32],
                  [0, 0, 1]], dtype=np.float32)
    # scale 
    T[:2, :] *= 8

    return T

def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def normalize_angle_degree(x):
    x = x % 360.0
    if (x > 180.0):
        x -= 360.0
    return x

####################################################################

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


####################################################################

def print_data_range(data):
    for i,ax in enumerate(COORDINATE_AXIS):
        print(f"{ax} axis | min = {data[:,i].min()} | max = {data[:,i].max()}")
        
####################################################################


def get_scatter3d_plot(x,y,z, mode='lines', marker_size=1, color=None, opacity=1, colorscale=None, **kwargs):
    return go.Scatter3d(x=x, y=y, z=z, mode=mode, hoverinfo='skip',showlegend=False, 
                        marker = dict(size=marker_size, color=color, opacity=opacity, colorscale=colorscale), **kwargs)

def plot_pc_data3d(x,y,z, apply_color_gradient=True, color=None, marker_size=1, colorscale=None, **kwargs):
    if apply_color_gradient:
        color = np.sqrt(x**2 + y **2 + z **2)
    return get_scatter3d_plot(x,y,z, mode='markers', color=color, colorscale=colorscale, marker_size=marker_size, **kwargs)


def plot_box_corners3d(box3d, color,**kwargs):
    return [
        get_scatter3d_plot(box3d[INDICES_1, 0], box3d[INDICES_1, 1], box3d[INDICES_1, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_2, 0], box3d[INDICES_2, 1], box3d[INDICES_2, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_3, 0], box3d[INDICES_3, 1], box3d[INDICES_3, 2], color=color, **kwargs),
        get_scatter3d_plot(box3d[INDICES_4, 0], box3d[INDICES_4, 1], box3d[INDICES_4, 2], color=color, **kwargs),
    ]


def plot_bboxes_3d(boxes3d, box_colors, **kwargs):
    # boxes3d shape = (N,8,3) = bounding box corners in 3d coordinates
    # box_colors = (N) length vector
    boxes3d_objs = []
    for obj_i in range(boxes3d.shape[0]):
        boxes3d_objs.extend(plot_box_corners3d(boxes3d[obj_i], color = box_colors[obj_i], **kwargs))
    return boxes3d_objs


def get_lidar3d_plots(points, pc_kwargs={}, gt_box_corners=None, gt_box_colors=None, 
                      pred_box_corners=None, pred_box_colors=None, **kwargs):
    lidar3d_plots = []
    #  point cloud data
    lidar3d_plots.append(plot_pc_data3d(x=points[:,0], y=points[:,1], z=points[:,2], **pc_kwargs))      
    # gt bounding boxes
    if((gt_box_corners is not None) and (gt_box_colors is not None)):
        lidar3d_plots.extend(plot_bboxes_3d(gt_box_corners, gt_box_colors, **kwargs))  
    # predicted bounding boxes
    if((pred_box_corners is not None) and (pred_box_colors is not None)):
        lidar3d_plots.extend(plot_bboxes_3d(pred_box_corners, pred_box_colors, **kwargs))  
    return lidar3d_plots

#############################################################################


def get_base64_string(rgb_image):
    pil_img = Image.fromarray(rgb_image) # PIL image object
    prefix = "data:image/png;base64,"
    with BytesIO() as stream:
        pil_img.save(stream, format="png")
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    return base64_string


def get_image2d_plots(rgb_image):
    return go.Image(source=get_base64_string(rgb_image), hoverinfo='skip')