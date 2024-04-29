import cv2
from collections import deque
from copy import deepcopy
from PIL import Image, ImageFont, ImageDraw
import numpy as np


pixels_per_meter = 8
bounding_box_divisor = 2.0 


def get_rotated_bbox(bbox):
    x, y, w, h, yaw, speed, brake =  bbox

    bbox = np.array([[h,   w, 1],
                        [h,  -w, 1],
                        [-h, -w, 1],
                        [-h,  w, 1],
                        [0, 0, 1],
                        [-h * speed * 0.5, 0, 1]])
    
    # The height and width of the bounding box value was changed by this factor 
    # during data collection. Fix that for future datasets and remove    
    bbox[:, :2] /= bounding_box_divisor
    bbox[:, :2] = bbox[:, [1, 0]]

    c, s = np.cos(yaw), np.sin(yaw)
    # use y x because coordinate is changed
    r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
    bbox = r1_to_world @ bbox.T
    bbox = bbox.T
    return bbox, brake


def draw_bboxes(self, bboxes, image, color=(255, 255, 255), brake_color=(0, 0, 255)):
    idx = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5]]
    for bbox, brake in bboxes:
        bbox = bbox.astype(np.int32)[:, :2]
        for s, e in idx:
            if brake >= self.config.draw_brake_threshhold:
                color = brake_color
            else:
                color = color
            # brake is true while still have high velocity
            cv2.line(image, tuple(bbox[s]), tuple(bbox[e]), color=color, thickness=1)
    return image


def draw_waypoints(label, waypoints, image, color = (255, 255, 255)):
    waypoints = waypoints.detach().cpu().numpy()
    label = label.detach().cpu().numpy()

    for bbox, points in zip(label, waypoints):
        x, y, w, h, yaw, speed, brake =  bbox
        c, s = np.cos(yaw), np.sin(yaw)
        # use y x because coordinate is changed
        r1_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])

        # convert to image space
        # need to negate y componet as we do for lidar points
        # we directly construct points in the image coordiante
        # for lidar, forward +x, right +y
        #            x
        #            +
        #            |
        #            |
        #            |---------+y
        #
        # for image, ---------> x
        #            |
        #            |
        #            +
        #            y

        points[:, 0] *= -1
        points = points * pixels_per_meter
        points = points[:, [1, 0]]
        points = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1)

        points = r1_to_world @ points.T
        points = points.T

        points_to_draw = []
        for point in points[:, :2]:
            points_to_draw.append(point.copy())
            point = point.astype(np.int32)
            cv2.circle(image, tuple(point), radius=3, color=color, thickness=3)
    return image


def draw_target_point(target_point, image, color = (255, 255, 255)):
    target_point = target_point.copy()

    target_point[1] += self.config.lidar_pos[0]
    point = target_point * self.config.pixels_per_meter
    point[1] *= -1
    point[1] = self.config.lidar_resolution_width - point[1] #Might be LiDAR height
    point[0] += int(self.config.lidar_resolution_height / 2.0) #Might be LiDAR width
    point = point.astype(np.int32)
    point = np.clip(point, 0, 512)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    return image

def visualize_model_io(self, save_path, step, config, rgb, lidar_bev, target_point,
                    pred_wp, pred_bev, pred_semantic, pred_depth, bboxes, device,
                    gt_bboxes=None, expert_waypoints=None, stuck_detector=0, forced_move=False):
    font = ImageFont.load_default()
    i = 0 # We only visualize the first image if there is a batch of them.
    if config.multitask:
        classes_list = config.classes_list
        converter = np.array(classes_list)

        depth_image = pred_depth[i].detach().cpu().numpy()

        indices = np.argmax(pred_semantic.detach().cpu().numpy(), axis=1)
        semantic_image = converter[indices[i, ...], ...].astype('uint8')

        ds_image = np.stack((depth_image, depth_image, depth_image), axis=2)
        ds_image = (ds_image * 255).astype(np.uint8)
        ds_image = np.concatenate((ds_image, semantic_image), axis=0)
        ds_image = cv2.resize(ds_image, (640, 256))
        ds_image = np.concatenate([ds_image, np.zeros_like(ds_image[:50])], axis=0)

    images = np.concatenate(list(lidar_bev.detach().cpu().numpy()[i][:2]), axis=1)
    images = (images * 255).astype(np.uint8)
    images = np.stack([images, images, images], axis=-1)
    images = np.concatenate([images, np.zeros_like(images[:50])], axis=0)

    # draw bbox GT
    if (not (gt_bboxes is None)):
        rotated_bboxes_gt = []
        for bbox in gt_bboxes.detach().cpu().numpy()[i]:
            bbox = self.get_rotated_bbox(bbox)
            rotated_bboxes_gt.append(bbox)
        images = self.draw_bboxes(rotated_bboxes_gt, images, color=(0, 255, 0), brake_color=(0, 255, 128))

    rotated_bboxes = []
    for bbox in bboxes.detach().cpu().numpy():
        bbox = self.get_rotated_bbox(bbox[:7])
        rotated_bboxes.append(bbox)
    images = self.draw_bboxes(rotated_bboxes, images, color=(255, 0, 0), brake_color=(0, 255, 255))

    label = torch.zeros((1, 1, 7)).to(device)
    label[:, -1, 0] = 128.
    label[:, -1, 1] = 256.

    if not expert_waypoints is None:
        images = self.draw_waypoints(label[0], expert_waypoints[i:i+1], images, color=(0, 0, 255))

    images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), images, color=(255, 255, 255)) # Auxliary waypoints in white
    images = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), images, color=(255, 0, 0))     # First two, relevant waypoints in blue

    # draw target points
    images = self.draw_target_point(target_point[i].detach().cpu().numpy(), images)

    # stuck text
    images = Image.fromarray(images)
    draw = ImageDraw.Draw(images)
    draw.text((10, 0), "stuck detector:   %04d" % (stuck_detector), font=font)
    draw.text((10, 30), "forced move:      %s" % (" True" if forced_move else "False"), font=font,
                fill=(255, 0, 0, 255) if forced_move else (255, 255, 255, 255))
    images = np.array(images)

    bev = pred_bev[i].detach().cpu().numpy().argmax(axis=0) / 2.
    bev = np.stack([bev, bev, bev], axis=2) * 255.
    bev_image = bev.astype(np.uint8)
    bev_image = cv2.resize(bev_image, (256, 256))
    bev_image = np.concatenate([bev_image, np.zeros_like(bev_image[:50])], axis=0)

    if not expert_waypoints is None:
        bev_image = self.draw_waypoints(label[0], expert_waypoints[i:i+1], bev_image, color=(0, 0, 255))

    bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, 2:]), bev_image, color=(255, 255, 255))
    bev_image = self.draw_waypoints(label[0], deepcopy(pred_wp[i:i + 1, :2]), bev_image, color=(255, 0, 0))

    bev_image = self.draw_target_point(target_point[i].detach().cpu().numpy(), bev_image)

    if (not (expert_waypoints is None)):
        aim = expert_waypoints[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
        expert_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))

        aim = pred_wp[i:i + 1, :2].detach().cpu().numpy()[0].mean(axis=0)
        ego_angle = np.degrees(np.arctan2(aim[1], aim[0] + self.config.lidar_pos[0]))
        angle_error = normalize_angle_degree(expert_angle - ego_angle)

        bev_image = Image.fromarray(bev_image)
        draw = ImageDraw.Draw(bev_image)
        draw.text((0, 0), "Angle error:        %.2f°" % (angle_error), font=font)

    bev_image = np.array(bev_image)

    rgb_image = rgb[i].permute(1, 2, 0).detach().cpu().numpy()[:, :, [2, 1, 0]]
    rgb_image = cv2.resize(rgb_image, (1280 + 128, 320 + 32))
    assert (config.multitask)
    images = np.concatenate((bev_image, images, ds_image), axis=1)

    images = np.concatenate((rgb_image, images), axis=0)

    cv2.imwrite(str(save_path + ("/%d.png" % (step // 2))), images)

