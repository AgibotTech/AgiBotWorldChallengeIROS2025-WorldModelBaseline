
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import random
import math
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from einops import rearrange
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
import jsonlines
from tqdm import tqdm
import torch.nn.functional as F
import cv2

from evac.lvdm.data.domain_table import DomainTable
from evac.lvdm.data.statistics import StatisticInfo
from evac.lvdm.data.traj_vis_statistics import ColorMapLeft, ColorMapRight, ColorListLeft, ColorListRight, EndEffectorPts, Gripper2EEFCvt
from evac.lvdm.data.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform, get_transformation_matrix_from_quat
from utils.general_utils import zero_rank_print
from evac.lvdm.data.get_actions import parse_h5, get_actions


class AgiBotWorldIROSChallenge(Dataset):
    def __init__(self,
        data_roots,
        domains,
        split="train",
        sample_size=(320,512), 
        sample_n_frames=64,
        preprocess = 'resize',
        valid_cam = 'head',
        chunk=1,
        n_previous=-1,
        previous_pick_mode='uniform',
        random_crop=True,
        min_sep=1,
        max_sep=3,
        fps=2,
    ):

        zero_rank_print(f"loading annotations...")

        self.random_crop = random_crop
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.valid_cam = valid_cam

        assert(self.valid_cam == "head")

        self.data_roots = data_roots
        self.dataset = []
        for _data_root, _domain_name in zip(self.data_roots, domains):
            file_list = os.listdir(os.path.join(_data_root, split))
            file_list.sort()
            for file_name in tqdm(file_list):
                task_id = file_name.split("-")[0]
                episode_id = file_name.split("-")[1]
                step_id = file_name.split("-")[2]
                info = [
                    os.path.join(_data_root, split, file_name), _domain_name, DomainTable[_domain_name],
                ]
                self.dataset.append(info)

        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.chunk = chunk
        self.sample_n_frames = sample_n_frames
        
        self.sample_size = sample_size

        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),  # the size of shape (1,) means the smaller edge will be resized to it and the img will keep the h-w ratio.
                transforms.CenterCrop(sample_size),
            ])
        if preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        self.pixel_transforms_norm = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.preprocess = preprocess
        self.random_erasing = None #transforms.RandomErasing(p=0.8, value=(-1.,0.38,-0.5), inplace=False)

        if n_previous > 1:
            self.n_previous = n_previous
            self.previous_pick_mode = previous_pick_mode
        else:
            self.n_previous = self.sample_n_frames - self.chunk
            self.previous_pick_mode = 'uniform'

        self.fps = fps

    def get_total_timesteps(self, data_root, cam_name):
        with open(os.path.join(data_root, cam_name+"_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        total_frames = len(info)
        return total_frames


    def get_frame_indexes(self, total_frames, sep=1, ):
        """
        select self.n_previous memory frames and self.chunk prediction frmaes
        1. randomly select the end frame
        2. take frames from {end-chunk*sep} to {end} as the prediction frames
        3. uniformly/randomly select memory frames from {end-self.sample_n_frames*sep} to {end-chunk*sep}
        """
        chunk_end = random.randint(self.chunk*sep, total_frames)
        indexes = np.array(list(range(chunk_end-self.sample_n_frames*sep, chunk_end, sep)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames-1).tolist()
        video_end = indexes[-self.chunk:]
        mem_candidates = [indexes[int(i)] for i in np.linspace(0, self.sample_n_frames-self.chunk-1, self.sample_n_frames-self.chunk).tolist()]
        if self.previous_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates)-1, self.n_previous).tolist()]

        elif self.previous_pick_mode == 'random':
            mem_indexes = [indexes[i] for i in sorted(np.random.choice(list(range(0,len(mem_candidates)-1)), size=self.n_previous-1, replace=False).tolist())] + [mem_candidates[-1]]

        else:
            raise NotImplementedError(f"unsupported previous_pick_mode: {self.previous_pick_mode}")
        frame_indexes = mem_indexes + video_end
        return frame_indexes


    def get_action_bias_std(self, domain_name):
        return torch.tensor(StatisticInfo[domain_name]['mean']).unsqueeze(0), torch.tensor(StatisticInfo[domain_name]['std']).unsqueeze(0)


    def get_action_npy(
        npy_file, slices, domain_name,
    ):
        """
        1. extract actions from .npy files
        content of .npy file to read:
            action (t, 16)                      : {xyz, quat(xyzw), gripper} * 2
        2. obatin End Effector actions and delta_action:
            action (t, 16)                      : {xyz, quat(xyzw), gripper} * 2
            delta_action (t-self.n_previous, 14): {xyz, quat(rpy),  gripper} * 2
        """
        abs_act = np.load(npy_file)
        action, delta_action = get_actions(
            gripper=np.stack((abs_act[:, 7], abs_act[:, 15]), axis=1),
            all_ends_p=np.stack((abs_act[:, 0:3], abs_act[:, 8:11]), axis=1),
            all_ends_o=np.stack((abs_act[:, 3:7], abs_act[:, 11:15]), axis=1),
            slices=slices,
            delta_act_sidx=self.n_previous,
        )
        action = torch.FloatTensor(action)
        delta_action = torch.FloatTensor(delta_action)
        delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name)

        delta_action[:, :6] = (delta_action[:, :6] - self.max_sep*delta_act_meanv[:, :6]) / (self.max_sep*delta_act_stdv[:, :6])
        delta_action[:, 7:13] = (delta_action[:, 7:13] - self.max_sep*delta_act_meanv[:, 6:]) / (self.max_sep*delta_act_stdv[:, 6:])
        return action, delta_action


    def get_action(
            self, h5_file, slices, domain_name
        ):
        """
        1. extract actions from .h5 files
        2. obatin End Effector actions and delta_action:
           action (t, 16)                      : {xyz, quat(xyzw), gripper} * 2
           delta_action (t-self.n_previous, 14): {xyz, quat(rpy),  gripper} * 2
        """
        action, delta_action = parse_h5(h5_file, slices=slices, delta_act_sidx=self.n_previous)
        action = torch.FloatTensor(action)
        delta_action = torch.FloatTensor(delta_action)
        delta_act_meanv, delta_act_stdv = self.get_action_bias_std(domain_name)
        delta_action[:, :6] = (delta_action[:, :6] - self.max_sep*delta_act_meanv[:, :6]) / (self.max_sep*delta_act_stdv[:, :6])
        delta_action[:, 7:13] = (delta_action[:, 7:13] - self.max_sep*delta_act_meanv[:, 6:]) / (self.max_sep*delta_act_stdv[:, 6:])

        return action, delta_action


    def seek_mp4(self, video_root, cam_name, slices):
        """
        seek video frames according to the input slices;
        output video shape: (c,t,h,w)
        """
        video_reader = VideoFileClip(os.path.join(video_root, cam_name+'_color.mp4'))
        fps = video_reader.fps
        video = []
        for idx in slices:
            video.append(video_reader.get_frame(float(idx)/fps))
        video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
        video = video.float()/255.
        video_reader.close()
        return video

    def get_intrin_and_extrin(self, cam_name, data_root, slices,):
        """
        get the intrinsic (3x3), c2ws (Tx4x4) and w2cs (Tx4x4) tensors
        """
        with open(os.path.join(data_root, cam_name+"_intrinsic_params.json"), "r") as f:
            info = json.load(f)["intrinsic"]
        intrinsic = torch.eye(3, dtype=torch.float)
        intrinsic[0,0] = info["fx"]
        intrinsic[1,1] = info["fy"]
        intrinsic[0,2] = info["ppx"]
        intrinsic[1,2] = info["ppy"]

        with open(os.path.join(data_root, cam_name+"_extrinsic_params_aligned.json"), "r") as f:
            info = json.load(f)
        c2ws = []
        w2cs = []
        for _i in slices:
            _i_info = info[_i]
            c2w = torch.eye(4, dtype=torch.float)
            c2w[:3, :3] = torch.FloatTensor(_i_info["extrinsic"]["rotation_matrix"])
            c2w[:3, -1] = torch.FloatTensor(_i_info["extrinsic"]["translation_vector"])
            w2c = torch.linalg.inv(c2w)
            c2ws.append(c2w)
            w2cs.append(w2c)
        c2ws = torch.stack(c2ws, dim=0)
        w2cs = torch.stack(w2cs, dim=0)
        return intrinsic, c2ws, w2cs


    def transform_video(self, video, specific_transforms_resize, intrinsic, sample_size):
        """
        crop (optional) and resize the videos, and modify the intrinsic accordingly
        """
        c, t, h, w = video.shape
        if self.random_crop:
            h_start, w_start, h_crop, w_crop = gen_crop_config(video)
            video = video[:,:,h_start:h_start+h_crop,w_start:w_start+w_crop]
            intrinsic = intrin_crop_transform(intrinsic, h_start, w_start)
            h, w = h_crop, w_crop
        intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
        video = specific_transforms_resize(video)
        return video, intrinsic


    def normalize_video(self, video, specific_transforms_norm):
        """
        input video should have shape (c,t,h,w)
        """
        video = specific_transforms_norm(video.permute(1,0,2,3)).permute(1,0,2,3)
        return video


    def get_transform(self, ):
        sample_size = self.sample_size
        specific_transforms_resize = self.pixel_transforms_resize
        specific_transforms_norm = self.pixel_transforms_norm
        return sample_size, specific_transforms_resize, specific_transforms_norm


    def get_traj(self, pose, w2c, c2w, intrinsic, radius=50):
        """
        this function takes camera info. and eef. poses as inputs, and outputs the trajectory maps.
        output traj map shape: (c, t, h, w)
        """        
        h, w = self.sample_size

        if isinstance(pose, np.ndarray):
            pose = torch.tensor(pose, dtype=torch.float32)
        
        ee_key_pts = torch.tensor(EndEffectorPts, dtype=torch.float32, device=pose.device).view(1,4,4).permute(0,2,1)

        ### t, 4, 4
        pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7])
        pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15])

        ### t, 4, 4
        ee2cam_l = torch.matmul(w2c, pose_l_mat)
        ee2cam_r = torch.matmul(w2c, pose_r_mat)

        cvt_matrix = torch.tensor(Gripper2EEFCvt, dtype=torch.float32, device=pose.device).view(1,4,4)
        ee2cam_l = torch.matmul(ee2cam_l, cvt_matrix)
        ee2cam_r = torch.matmul(ee2cam_r, cvt_matrix)
        
        ### t, 4, 4
        pts_l = torch.matmul(ee2cam_l, ee_key_pts)
        pts_r = torch.matmul(ee2cam_r, ee_key_pts)
        
        ### 1, 3, 3
        intrinsic = intrinsic.unsqueeze(0)

        ### t, 3, 4
        uvs_l = torch.matmul(intrinsic, pts_l[:,:3,:])
        uvs_l = (uvs_l / pts_l[:,2:3,:])[:,:2,:].permute(0,2,1).to(dtype=torch.int64)

        ### t, 3, 4
        uvs_r = torch.matmul(intrinsic, pts_r[:,:3,:])
        uvs_r = (uvs_r / pts_r[:,2:3,:])[:,:2,:].permute(0,2,1).to(dtype=torch.int64)

        img_list = []

        for i in range(pose.shape[0]):
            
            img = np.zeros((h, w, 3), dtype=np.uint8) + 50

            ###
            ### Gripper Range in AgiBotWorld < 120
            normalized_value_l = pose[i, 7].item() / 120
            normalized_value_r = pose[i, 15].item() / 120
            color_l = ColorMapLeft(normalized_value_l)[:3]  # Get RGB values
            color_r = ColorMapRight(normalized_value_r)[:3]  # Get RGB values
            color_l = tuple(int(c * 255) for c in color_l)
            color_r = tuple(int(c * 255) for c in color_r)

            i_coord_list = []
            for points, color, colors, lr_tag in zip([uvs_l[i], uvs_r[i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0])
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                point = np.array(points[0][:2])
                cv2.circle(img, tuple(point), radius, color, -1)
                

            for points, color, colors, lr_tag in zip([uvs_l[i], uvs_r[i]], [color_l, color_r], [ColorListLeft, ColorListRight], ["left", "right"]):
                base = np.array(points[0]) # points:[4,3]
                if base[0]<0 or base[0]>=w or base[1]<0 or base[1]>=h:
                    continue
                for i, point in enumerate(points):
                    point = np.array(point[:2])
                    if i == 0:
                        continue
                    else:
                        cv2.line(img, tuple(base), tuple(point), colors[i-1], 8)

            img_list.append(img/255.)

        img_list = np.stack(img_list, axis=0) ### t,h,w,c
        img_list = rearrange(torch.tensor(img_list), "t h w c -> c t h w").float()

        return img_list


    def get_batch_new(self, idx, debug=False):

        video_root = self.dataset[idx][0]
        caminfo_root = self.dataset[idx][0]
        h5_file = os.path.join(self.dataset[idx][0], "proprio_stats.h5")
        domain_name = self.dataset[idx][1]
        domain_id = self.dataset[idx][2]

        total_frames = self.get_total_timesteps(caminfo_root, self.valid_cam)

        ### 
        ### random action-speed
        sep = random.randint(self.min_sep, self.max_sep)

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()

        video_indexes = self.get_frame_indexes(total_frames, sep=sep, )

        action, delta_action = self.get_action(h5_file, video_indexes, domain_name)

        intrinsics, c2ws, w2cs = self.get_intrin_and_extrin(self.valid_cam, caminfo_root, video_indexes)

        ### c, total_frames, h, w
        video = self.seek_mp4(video_root, self.valid_cam, video_indexes)
        video, intrinsics = self.transform_video(
            video, specific_transforms_resize, intrinsics, sample_size
        )
        if isinstance(video, (list, tuple)):
            video = torch.stack(video, dim=1)

        traj_maps = self.get_traj(action, w2cs, c2ws, intrinsics)

        video = self.normalize_video(video, specific_transforms_norm)
        traj_maps = self.normalize_video(traj_maps, specific_transforms_norm)

        cond_id = -(self.n_previous+self.chunk)

        fps = self.fps

        return video, video_root, cond_id, intrinsics, c2ws, domain_id, action, traj_maps, delta_action, fps

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        while True:
            try:
                video, video_root, cond_id, intrinsics, extrinsics, domain_id, action, traj_maps, delta_action, fps = self.get_batch_new(idx)
                break
            except Exception as e:
                ### 
                idx = random.randint(0, self.length-1)
        sample = dict(
            video=video, path=video_root,
            cond_id=cond_id, intrinsic=intrinsics, extrinsic=extrinsics, domain_id=domain_id,
            action=action, traj=traj_maps, delta_action=delta_action, fps=fps
        )
        return sample


