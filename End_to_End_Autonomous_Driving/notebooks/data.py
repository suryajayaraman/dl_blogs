import os
import cv2
import ujson
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
from config import GlobalConfig

from utils import get_vehicle_to_virtual_lidar_transform, \
      get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform


class CARLA_Data(Dataset):
    def __init__(self, root, config : GlobalConfig, routeKey = None):
        self.seq_len = np.array(config.seq_len)
        self.pred_len = np.array(config.pred_len)
        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)        
        self.converter = np.uint8(config.converter)

        self.images = []
        self.bevs = []
        self.depths = []
        self.semantics = []
        self.lidars = []
        self.labels = []
        self.measurements = []

        # each route ~= video sequence with all sensor data
        routes = sorted(os.listdir(root))

        # filter by route, if key is provided
        if routeKey is not None:
            routes = [x for x in routes if routeKey in x]

        # create absolute path
        routes = [os.path.join(root,x) for x in routes]

        for route_dir_str in tqdm(routes, file=sys.stdout):
            route_dir = Path(route_dir_str)
            num_seq = len(os.listdir(route_dir / "lidar"))

            # ignore the first two and last two frame
            for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                # load input seq and pred seq jointly
                image = route_dir / "rgb" / ("%04d.png" % (seq))
                bev = route_dir / "topdown" / ("encoded_%04d.png" % (seq))
                depth = route_dir / "depth" / ("%04d.png" % (seq))
                semantic = route_dir / "semantics" / ("%04d.png" % (seq))
                lidar = route_dir / "lidar" / ("%04d.npy" % (seq))
                measurement = route_dir / "measurements" / ("%04d.json"%(seq))

                # Additionally load future labels of the waypoints
                label = []                        
                for idx in range(self.seq_len + self.pred_len):
                    label.append(route_dir / "label_raw" / ("%04d.json" % (seq + idx)))

                self.images.append(image)
                self.bevs.append(bev)
                self.depths.append(depth)
                self.semantics.append(semantic)
                self.lidars.append(lidar)
                self.labels.append(label)
                self.measurements.append(measurement)

        # There is a complex "memory leak"/performance issue when using Python objects like 
        # lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here 
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.images       = np.array(self.images      ).astype(np.string_)
        self.bevs         = np.array(self.bevs        ).astype(np.string_)
        self.depths       = np.array(self.depths      ).astype(np.string_)
        self.semantics    = np.array(self.semantics   ).astype(np.string_)
        self.lidars       = np.array(self.lidars      ).astype(np.string_)
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)

    def __len__(self):
        """Returns the length of the dataset. """
        return self.lidars.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        cv2.setNumThreads(0) # Disable threading because the data loader will already split in threads.
        data = {}

        # load RGB image, scale to resolution, change to (C, H, W) format
        rgb_image = cv2.imread(str(self.images[index], encoding='utf-8'), cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        data['rgb'] = crop_image_cv2(rgb_image, crop=self.img_resolution, channelFirst = True)

        # BEV image -> load, decode, crop 
        bev_array = cv2.imread(str(self.bevs[index], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
        bev_array = cv2.cvtColor(bev_array, cv2.COLOR_BGR2RGB)
        bev_array = np.moveaxis(bev_array, -1, 0)
        loaded_bevs = decode_pil_to_npy(bev_array).astype(np.uint8)
        data['bev'] = load_crop_bev_npy(loaded_bevs, degree=0)        

        # Depth image
        depth_image = cv2.imread(str(self.depths[index], encoding='utf-8'), cv2.IMREAD_COLOR)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
        data['depth'] = get_depth(crop_image_cv2(depth_image, crop=self.img_resolution, channelFirst = True))

        # Semantic segmented image
        semantic_image = cv2.imread(str(self.semantics[index], encoding='utf-8'), cv2.IMREAD_UNCHANGED)
        data['semantic'] = self.converter[crop_image_cv2(semantic_image, crop=self.img_resolution)]

        # vehicle measurements
        with open(str(self.measurements[index], encoding='utf-8'), 'r') as f1:
            measurements = ujson.load(f1)

        for k in ['speed', 'x_command', 'y_command']:
            data[k] = measurements[k]

        # target points
        # convert x_command, y_command to local coordinates
        # taken from LBC code (uses 90+theta instead of theta)
        ego_theta = measurements['theta']
        ego_x = measurements['x']
        ego_y = measurements['y']
        x_command = measurements['x_command']
        y_command = measurements['y_command']
        
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        local_command_point = R.T.dot(local_command_point)
        data['target_point'] = local_command_point
        data['target_point_image'] = draw_target_point(local_command_point)

        # load point cloud (XYZI), flip y-axis
        # compute Lidar BEV features
        lidars_pc = np.load(str(self.lidars[index], encoding='utf-8'), allow_pickle=True)[1] 
        lidars_pc[:, 1] *= -1
        data['lidar'] = lidar_to_histogram_features(lidars_pc)


        # Because the strings are stored as numpy byte objects we need to convert them back to utf-8 strings
        # Since we also load labels for future timesteps, we load and store them separately
        labels = []
        for i in range(self.seq_len+self.pred_len):
            with open(str(self.labels[index][i], encoding='utf-8'), 'r') as f2:
                labels_i = ujson.load(f2)
            labels.append(labels_i)

        # ego car is always the first one in label file
        ego_id = labels[0][0]['id']

        # Bounding boxes of objects within BEV FOV(32m x 32m)
        # considering only current frame objects
        bboxes = parse_labels(labels[0])
        label = np.array(list(bboxes.values()))
        label_pad = np.zeros((20, 7), dtype=np.float32)
        if label.shape[0] > 0:
            label_pad[:label.shape[0], :] = label
        data['label'] = label_pad

        # use position of ego vehicle in future frames
        # as groundtruth reference
        waypoints = get_waypoints(labels[self.seq_len-1:], self.pred_len+1)
        waypoints = transform_waypoints(waypoints)
        ego_waypoints = np.array([x[0][:2,3] for x in waypoints[ego_id][1:]])
        data['ego_waypoint'] = ego_waypoints
        return data



def get_depth(data):
    """
    Computes the normalized depth
    """
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    # in_meters = 1000 * normalized
    #clip to 50 meters
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 # Rescale map to lie in [0,1]

    return normalized


def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    # padding here
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

# this is only for visualization, For training, we should use vehicle coordinate

def transform_waypoints(waypoints):
    """transform waypoints to be origin at ego_matrix"""

    T = get_vehicle_to_virtual_lidar_transform()
    
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
            
    return waypoints

def align(lidar_0, measurements_0, measurements_1, degree=0):
    
    matrix_0 = measurements_0['ego_matrix']
    matrix_1 = measurements_1['ego_matrix']

    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)
   
    Tr_lidar_to_vehicle = get_lidar_to_vehicle_transform()
    Tr_vehicle_to_lidar = get_vehicle_to_lidar_transform()

    transform_0_to_1 = Tr_vehicle_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_vehicle

    # augmentation
    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0],
                              [-np.sin(rad), np.cos(rad), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1
                            
    lidar = lidar_0.copy()
    lidar[:, -1] = 1.
    #important we should convert the points back to carla format because when we save the data we negatived y component
    # and now we change it back 
    lidar[:, 1] *= -1.
    lidar = transform_0_to_1 @ lidar.T
    lidar = lidar.T
    lidar[:, -1] = lidar_0[:, -1]
    # and we change back here
    lidar[:, 1] *= -1.

    return lidar


def lidar_to_histogram_features(lidar):
    """
    Convert LiDAR point cloud into 2-bin histogram over 256x256 grid
    """
    def splat_points(point_cloud):
        # 256 x 256 grid
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def get_bbox_label(bbox, rad=0):
    # dx, dy, dz, x, y, z, yaw
    # ignore z
    dz, dx, dy, x, y, z, yaw, speed, brake =  bbox

    pixels_per_meter = 8

    # augmentation
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                              [-np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position

    position = np.clip(position, 0., 255.)
    x, y = position[:2, 0]
    # center_x, center_y, w, h, yaw
    bbox = np.array([x, y, dy*pixels_per_meter, dx*pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox


def parse_labels(labels, rad=0):
    bboxes = {}
    for result in labels:
        num_points = result['num_points']
        bbox = result['extent'] + result['position'] + [result['yaw'], result['speed'], result['brake']]
        bbox = get_bbox_label(bbox, rad)

        # Filter bb that are outside of the LiDAR after the random augmentation. The bounding box is now in image space
        if num_points <= 1 or bbox[0] <= 0.0 or bbox[0] >= 255.0 or bbox[1] <= 0.0 or bbox[1] >=255.0:
            continue

        bboxes[result['id']] = bbox
    return bboxes


def crop_image_cv2(image, crop=(128, 640), channelFirst=False):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2

    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    if channelFirst:
        cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image


def load_crop_bev_npy(bev_array, degree):
    """
    Load and crop an Image.
    Crop depends on augmentation angle.
    """
    PIXELS_PER_METER_FOR_BEV = 5
    PIXLES = 32 * PIXELS_PER_METER_FOR_BEV
    start_x = 250 - PIXLES // 2
    start_y = 250 - PIXLES

    # shift the center by 7 because the lidar is + 1.3 in x 
    bev_array = np.moveaxis(bev_array, 0, -1).astype(np.float32)
    bev_shift = np.zeros_like(bev_array)
    bev_shift[7:] = bev_array[:-7]

    bev_shift = rotate(bev_shift, degree)
    cropped_image = bev_shift[start_y:start_y+PIXLES, start_x:start_x+PIXLES]
    cropped_image = np.moveaxis(cropped_image, -1, 0)

    # we need to predict others so append 0 to the first channel
    cropped_image = np.concatenate((np.zeros_like(cropped_image[:1]), 
                                    cropped_image[:1],
                                    cropped_image[:1] + cropped_image[1:2]), axis=0)

    cropped_image = np.argmax(cropped_image, axis=0)
    
    return cropped_image



def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()

    # convert to lidar coordinate
    target_point[1] += 1.3
    point = target_point * 8.
    point[1] *= -1
    point[1] = 256 - point[1] 
    point[0] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float32) / 255.


def decode_pil_to_npy(img):
    (channels, width, height) = (15, img.shape[1], img.shape[2])

    bev_array = np.zeros([channels, width, height])

    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos

    # hard coded to select
    return bev_array[10:12]


if __name__ == "__main__":
    root_dir = '/home/surya/Downloads/transfuser-2022/data/demo/scenario1/'
    config = GlobalConfig()
    train_set = CARLA_Data(root=root_dir, config=config, routeKey='route0')
    print(len(train_set))
    sample = train_set[0]
    print('sample data loaded')