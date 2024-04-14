import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data

from torch.distributed.elastic.multiprocessing.errors import record
import random
import torch.multiprocessing as mp

log_dir = os.path.join('/home/surya/Downloads/transfuser_logs', 'test')
root_dir = '/home/surya/Downloads/CARLA_data'
N_EPOCHS = 1

# Records error and tracebacks in case of failure
@record
def main():
    torch.cuda.empty_cache()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # Configure config
    config = GlobalConfig(root_dir=root_dir, setting='all')

    # Create model and optimizers
    model = LidarCenterNet(config, device, config.backbone, image_architecture='regnety_032', 
                           lidar_architecture='regnety_032', use_velocity=False)

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4) # For single GPU training
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)

    # Data
    train_set = CARLA_Data(root=config.train_data, config=config)
    val_set   = CARLA_Data(root=config.val_data,   config=config)

    g_cuda = torch.Generator(device='cpu')
    g_cuda.manual_seed(torch.initial_seed())

    dataloader_train = DataLoader(train_set, shuffle=True, batch_size=2, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)
    dataloader_val   = DataLoader(val_set,   shuffle=True, batch_size=2, worker_init_fn=seed_worker, generator=g_cuda, num_workers=0, pin_memory=True)

    writer = SummaryWriter(log_dir=log_dir)
    trainer = Engine(model=model, optimizer=optimizer, dataloader_train=dataloader_train, dataloader_val=dataloader_val,
                     config=config, writer=writer, device=device)

    for epoch in range(trainer.cur_epoch, N_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print("Reduce learning rate by factor 10 to:", new_lr)
        for g in optimizer.param_groups:
            g['lr'] = new_lr
        trainer.train()
        trainer.validate()
        trainer.save()


class Engine(object):
    """
    Engine that runs training.
    """

    def __init__(self, model, optimizer, dataloader_train, dataloader_val, config : GlobalConfig, writer, device):
        self.cur_epoch = 0
        self.bestval_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.model = model
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.dataloader_val   = dataloader_val
        self.config = config
        self.writer = writer
        self.device = device
        self.vis_save_path = self.log_dir + r'/visualizations'
        self.detailed_losses = config.detailed_losses
        detailed_losses_weights = config.detailed_losses_weights
        self.detailed_weights = {key: detailed_losses_weights[idx] for idx, key in enumerate(self.detailed_losses)}

    def load_data_compute_loss(self, data):
        # Move data to GPU
        rgb = data['rgb'].to(self.device, dtype=torch.float32)
        depth = data['depth'].to(self.device, dtype=torch.float32)
        semantic = data['semantic'].squeeze(1).to(self.device, dtype=torch.long)
        bev = data['bev'].to(self.device, dtype=torch.long)
        lidar = data['lidar'].to(self.device, dtype=torch.float32)
        num_points = None
        label = data['label'].to(self.device, dtype=torch.float32)
        ego_waypoint = data['ego_waypoint'].to(self.device, dtype=torch.float32)
        target_point = data['target_point'].to(self.device, dtype=torch.float32)
        target_point_image = data['target_point_image'].to(self.device, dtype=torch.float32)
        ego_vel = data['speed'].to(self.device, dtype=torch.float32)

        losses = self.model(rgb, lidar, ego_waypoint=ego_waypoint, target_point=target_point,
                        target_point_image=target_point_image,
                        ego_vel=ego_vel.reshape(-1, 1), bev=bev,
                        label=label, save_path=self.vis_save_path,
                        depth=depth, semantic=semantic, num_points=num_points)
        return losses


    def train(self):
        self.model.train()
        num_batches = 0
        loss_epoch = 0.0
        detailed_losses_epoch  = {key: 0.0 for key in self.detailed_losses}
        self.cur_epoch += 1

        # Train loop
        for data in tqdm(self.dataloader_train):
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.load_data_compute_loss(data)
            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)
            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
            loss.backward()

            self.optimizer.step()
            num_batches += 1
            loss_epoch += float(loss.item())

        self.log_losses(loss_epoch, detailed_losses_epoch, num_batches, '')


    @torch.inference_mode() # Faster version of torch_no_grad
    def validate(self):
        self.model.eval()

        num_batches = 0
        loss_epoch = 0.0
        detailed_val_losses_epoch  = {key: 0.0 for key in self.detailed_losses}

        # Evaluation loop loop
        for data in tqdm(self.dataloader_val):
            losses = self.load_data_compute_loss(data)
            loss = torch.tensor(0.0).to(self.device, dtype=torch.float32)
            for key, value in losses.items():
                loss += self.detailed_weights[key] * value
                detailed_val_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
            num_batches += 1
            loss_epoch += float(loss.item())
        self.log_losses(loss_epoch, detailed_val_losses_epoch, num_batches, 'val_')


    def log_losses(self, loss_epoch, detailed_losses_epoch, num_batches, prefix=''):
        # Average all the batches into one number
        loss_epoch = loss_epoch / num_batches
        for key, value in detailed_losses_epoch.items():
            detailed_losses_epoch[key] = value / num_batches

        # In parallel training aggregate all values onto the master node.
        gathered_detailed_losses = [detailed_losses_epoch]
        gathered_loss = [loss_epoch]
            
        # Log main loss
        aggregated_total_loss = sum(gathered_loss) / len(gathered_loss)
        self.writer.add_scalar(prefix + 'loss_total', aggregated_total_loss, self.cur_epoch)

        # Log detailed losses
        for key, value in detailed_losses_epoch.items():
            aggregated_value = 0.0
            for i in range(self.world_size):
                aggregated_value += gathered_detailed_losses[i][key]

            aggregated_value = aggregated_value / self.world_size
            self.writer.add_scalar(prefix + key, aggregated_value, self.cur_epoch)

    def save(self):
        # NOTE saving the model with torch.save(model.module.state_dict(), PATH) if parallel processing is used would be cleaner, we keep it for backwards compatibility
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model_%d.pth' % self.cur_epoch))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'optimizer_%d.pth' % self.cur_epoch))

# We need to seed the workers individually otherwise random processes in the dataloader return the same values across workers!
def seed_worker(worker_id):
    # Torch initial seed is properly set across the different workers, we need to pass it to numpy and random.
    worker_seed = (torch.initial_seed()) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    # The default method fork can run into deadlocks.
    # To use the dataloader with multiple workers forkserver or spawn should be used.
    mp.set_start_method('fork')
    print("Start method of multiprocessing:", mp.get_start_method())
    main()
