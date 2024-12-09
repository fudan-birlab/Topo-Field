import json
from pathlib import Path
from typing import List, Optional
from zipfile import ZipFile

import liblzfse
import numpy as np
import open3d as o3d
import tqdm
from PIL import Image
from quaternion import as_rotation_matrix, quaternion
from torch.utils.data import Dataset

from dataloaders.scannet_200_classes import CLASS_LABELS_200
import os
import torch
import clip
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class R3DSemanticDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
    ):
        if path.endswith((".zip", ".r3d")):
            self._path = ZipFile(path)
        else:
            self._path = Path(path)

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200

        self._reshaped_depth = []
        self._reshaped_conf = []
        self._depth_images = []
        self._rgb_images = []
        self._confidences = []

        self._metadata = self._read_metadata()
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data()
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self):
        with self._path.open("metadata", "r") as f:
            metadata_dict = json.load(f)

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.fps = metadata_dict["fps"]
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.image_size = (self.rgb_width, self.rgb_height)
        self.poses = np.array(metadata_dict["poses"])
        self.init_pose = np.array(metadata_dict["initPose"])
        self.total_images = len(self.poses)

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        with self._path.open(filepath, "r") as image_file:
            return np.asarray(Image.open(image_file))

    def load_depth(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img: np.ndarray = np.frombuffer(decompressed_bytes, dtype=np.float32)

        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def load_conf(self, filepath):
        with self._path.open(filepath, "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
        if depth_img.shape[0] == 960 * 720:
            depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        else:
            depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        return depth_img

    def _load_data(self):
        assert self.fps  # Make sure metadata is read correctly first.
        for i in tqdm.trange(self.total_images, desc="Loading data"):
            # Read up the RGB and depth images first.
            rgb_filepath = f"rgbd/{i}.jpg"
            depth_filepath = f"rgbd/{i}.depth"
            conf_filepath = f"rgbd/{i}.conf"

            depth_img = self.load_depth(depth_filepath)
            confidence = self.load_conf(conf_filepath)
            rgb_img = self.load_image(rgb_filepath)

            # Now, convert depth image to real world XYZ pointcloud.
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)

    def _reshape_all_depth_and_conf(self):
        for index in tqdm.trange(len(self.poses), desc="Upscaling depth and conf"):
            depth_image = self._depth_images[index]
            # Upscale depth image.
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height))
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

            # Upscale confidence as well
            confidence = self._confidences[index]
            conf_img = Image.fromarray(confidence)
            reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height))
            reshaped_conf = np.asarray(reshaped_conf)
            self._reshaped_conf.append(reshaped_conf)

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        reshaped_img = np.copy(self._reshaped_depth[index])
        # If only confident, replace not confident points with nans
        if only_confident:
            reshaped_img[self._reshaped_conf[index] != 2] = np.nan

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )

        # Flip the pcd
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.poses[index]
        extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # Now transform everything by init pose.
        init_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.init_pose
        init_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        init_matrix[:3, -1] = [px, py, pz]
        pcd.transform(init_matrix)

        return pcd

    def calculate_all_global_xyzs(self, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            self.global_xyzs.append(global_xyz)
            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result

class ApartmentDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
    ):
        self._path = path

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
        
        self._depth_images = []
        self._rgb_images = []
        self.interval = 6
        self.num_frames = -1
        self._metadata = self._read_metadata(w=640, h=360, interval=self.interval)
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data(interval=self.interval)

        # region_list = ['living room', 'utility room', 'bedroom']
        self.region_id = []
        
        self.calculate_all_global_xyzs(interval=self.interval)

    def read_trajectory_file(self, file_path):
        poses = np.loadtxt(file_path)
        reshaped_pose = []
        for pose in poses:
            reshaped_pose.append(pose.reshape([4, 4]))
        return reshaped_pose
    
    def _read_metadata(self, w=0, h=0, interval=1):
        with open(self._path+"/intrinsic.json", "r") as f:
            metadata_dict = json.load(f)
        self.rgb_width = metadata_dict["width"]
        self.rgb_height = metadata_dict["height"]
        self.fps = 15 # for test
        camera_matrix = np.array(metadata_dict["intrinsic_matrix"]).reshape([3, 3])
        if w > 0: 
            scale_x = w / self.rgb_width
            scale_y = h / self.rgb_height
            camera_matrix[0, 0] *= scale_x
            camera_matrix[1, 1] *= scale_y
            camera_matrix[0, 2] *= scale_x
            camera_matrix[1, 2] *= scale_y
            self.rgb_width = w
            self.rgb_height = h
        self.camera_matrix = camera_matrix

        self.image_size = (self.rgb_width, self.rgb_height)
        self.total_poses = self.read_trajectory_file(self._path+'/train_trajectory.txt')
        self.total_images = len(self.total_poses)
        self.poses = []
        for i in range(0, len(self.total_poses)):
            if i > self.num_frames and self.num_frames > 0:
                break
            if i % interval == 0:
                self.poses.append(self.total_poses[i])
        self.init_pose = self.poses[0]
        
        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        image = Image.open(filepath)
        if image.size != self.image_size:
            image = image.resize(self.image_size)
        image_array = np.array(image)
        return image_array
    
    def load_depth(self, filepath):
        depth_image = Image.open(filepath)
        if depth_image.size != self.image_size:
            depth_image = depth_image.resize(self.image_size)
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    
    def _load_data(self, interval=1):
        assert self.fps  # Make sure metadata is read correctly first.
        i = 0
        files = os.listdir(self._path + "/color_train")
        sorted_files = sorted(files)
        for filename in sorted_files:
            if i > self.num_frames and self.num_frames > 0:
                break
            # Read up the RGB and depth images.
            if i % interval == 0:
                rgb_filepath = self._path + "/color_train/" + filename
                depth_filepath = self._path + "/depth_train/" + filename.replace('.jpg', '.png')
                # conf_filepath = self._path+f"color/{i}.conf"

                depth_img = self.load_depth(depth_filepath)
                # confidence = self.load_conf(conf_filepath)
                rgb_img = self.load_image(rgb_filepath)

                # Now, convert depth image to real world XYZ pointcloud.
                self._depth_images.append(depth_img)
                self._rgb_images.append(rgb_img)
                # self._confidences.append(confidence)
            i += 1
    
    def get_global_xyz(self, index, depth_scale=1.0, only_confident=True):
        reshaped_img = np.copy(self._depth_images[index])

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the pcd.  done when loading pose
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pcd.transform(self.poses[index])
        # pcd.transform(self.init_pose)

        pose_trans = np.eye(4)
        qx, qy, qz, qw, px, py, pz = [-0.1538211, -0.02841006, -0.01133235, 0.98762512, 0.0297593, -0.0121457, -0.01638832]
        pose_trans[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        pose_trans[:3, -1] = [px, py, pz]
        pcd.transform(pose_trans)

        return pcd

    def calculate_all_global_xyzs(self, interval=1, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            
            self.global_xyzs.append(global_xyz)

            xyz_region_id = []
            for xyz in global_xyz:
                if (xyz[2] > (-4.9 * xyz[0] + 20.3)) and (xyz[2] < (0.23 * xyz[0] + 0.8)):
                    xyz_region_id.append(1)
                elif (xyz[2] > (-4.9 * xyz[0] + 23.6667)) and (xyz[2] > (0.23 * xyz[0] + 0.8)):
                    xyz_region_id.append(2)
                else:
                    xyz_region_id.append(0)
                # z > -4.7368x + 19.7367 ->客厅
                # z > -4.7368x + 19.7367 and z < 0.1769x + 1.1308 -> 健身器材、衣服、箱子
                # z > -4.6667x + 23.6667 and z > 0.1769x + 1.1308 -> 卧室
            self.region_id.append([xyz_region_id])

            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._depth_images[idx],
            "camera_pose": self.poses[idx],
            # "xyz_clip_token": self.region_clip_tokens[idx],
            # "xyz_st_token": self.region_st_tokens[idx],
            "xyz_region_id": self.region_id[idx],
            # "conf": self._reshaped_conf[idx],
        }
        return result
    

class TestApartmentDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
    ):
        self._path = path

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
        
        self._depth_images = []
        self._rgb_images = []
        
        self.interval = 1
        self.num_frames = -1
        self._metadata = self._read_metadata(w=640, h=360, interval=self.interval)
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data(interval=self.interval)

        # region_list = ['living room', 'utility room', 'bedroom']
        self.region_id = []
        
        self.calculate_all_global_xyzs(interval=self.interval)

    def read_trajectory_file(self, file_path):
        poses = np.loadtxt(file_path)
        reshaped_pose = []
        for pose in poses:
            reshaped_pose.append(pose.reshape([4, 4]))
        return reshaped_pose
    
    def _read_metadata(self, w=0, h=0, interval=1):
        with open(self._path+"/intrinsic.json", "r") as f:
            metadata_dict = json.load(f)
        self.rgb_width = metadata_dict["width"]
        self.rgb_height = metadata_dict["height"]
        self.fps = 15 # for test
        camera_matrix = np.array(metadata_dict["intrinsic_matrix"]).reshape([3, 3])
        if w > 0: 
            scale_x = w / self.rgb_width
            scale_y = h / self.rgb_height
            camera_matrix[0, 0] *= scale_x
            camera_matrix[1, 1] *= scale_y
            camera_matrix[0, 2] *= scale_x
            camera_matrix[1, 2] *= scale_y
            self.rgb_width = w
            self.rgb_height = h
        self.camera_matrix = camera_matrix

        self.image_size = (self.rgb_width, self.rgb_height)
        self.total_poses = self.read_trajectory_file(self._path+'/test_trajectory.txt')
        self.total_images = len(self.total_poses)
        self.poses = []
        for i in range(0, len(self.total_poses)):
            if i > self.num_frames and self.num_frames > 0:
                break
            if i % interval == 0:
                self.poses.append(self.total_poses[i])
        self.init_pose = self.poses[0]
        
        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def load_image(self, filepath):
        image = Image.open(filepath)
        if image.size != self.image_size:
            image = image.resize(self.image_size)
        image_array = np.array(image)
        return image_array
    
    def load_depth(self, filepath):
        depth_image = Image.open(filepath)
        if depth_image.size != self.image_size:
            depth_image = depth_image.resize(self.image_size)
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    
    def _load_data(self, interval=1):
        assert self.fps  # Make sure metadata is read correctly first.
        # for i in tqdm.trange(0, len(self.total_poses), desc=f"Loading data (interval={interval})"):
        i = 0
        for filename in os.listdir(self._path + "/color_test"):
            if i > self.num_frames and self.num_frames > 0:
                break
            if i % interval == 0:
                # Read up the RGB and depth images.
                rgb_filepath = self._path + "/color_test/" + filename
                depth_filepath = self._path + "/depth_test/" + filename.replace('.jpg', '.png')
                # conf_filepath = self._path+f"color/{i}.conf"

                depth_img = self.load_depth(depth_filepath)
                # confidence = self.load_conf(conf_filepath)
                rgb_img = self.load_image(rgb_filepath)

                # Now, convert depth image to real world XYZ pointcloud.
                self._depth_images.append(depth_img)
                self._rgb_images.append(rgb_img)
                # self._confidences.append(confidence)
            i += 1
    
    def get_global_xyz(self, index, depth_scale=1.0, only_confident=True):
        reshaped_img = np.copy(self._depth_images[index])

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the pcd.  done when loading pose
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        pcd.transform(self.poses[index])
        pcd.transform(self.init_pose)

        pose_trans = np.eye(4)
        qx, qy, qz, qw, px, py, pz = [-0.1538211, -0.02841006, -0.01133235, 0.98762512, 0.0297593, -0.0121457, -0.01638832]
        pose_trans[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        pose_trans[:3, -1] = [px, py, pz]
        pcd.transform(pose_trans)

        return pcd

    def calculate_all_global_xyzs(self, interval=1, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            
            self.global_xyzs.append(global_xyz)

            xyz_region_id = []
            for xyz in global_xyz:
                if (xyz[2] > (-4.9 * xyz[0] + 20.3)) and (xyz[2] < (0.23 * xyz[0] + 0.8)):
                    xyz_region_id.append(1)
                elif (xyz[2] > (-4.9 * xyz[0] + 23.6667)) and (xyz[2] > (0.23 * xyz[0] + 0.8)):
                    xyz_region_id.append(2)
                else:
                    xyz_region_id.append(0)
                # z > -4.7368x + 19.7367 ->客厅
                # z > -4.7368x + 19.7367 and z < 0.1769x + 1.1308 -> 健身器材、衣服、箱子
                # z > -4.6667x + 23.6667 and z > 0.1769x + 1.1308 -> 卧室
            self.region_id.append([xyz_region_id])

            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._depth_images[idx],
            "camera_pose": self.poses[idx],
            # "xyz_clip_token": self.region_clip_tokens[idx],
            # "xyz_st_token": self.region_st_tokens[idx],
            "xyz_region_id": self.region_id[idx],
            # "conf": self._reshaped_conf[idx],
        }
        return result

class Matterport3DDataset(Dataset):

    REGION_LABLE_MAPPING = {
        'a':"bathroom (should have a toilet and a sink)", #1
        'b':"bedroom",#2
        'c':"closet",
        'd':"dining room (includes “breakfast rooms” other rooms people mainly eat in)",#3
        'e':"entryway/foyer/lobby (should be the front door, not any door)",#4
        'f':"familyroom (should be a room that a family hangs out in, not any area with couches)",#5
        'g':"garage",
        'h':"hallway",
        'i':"library (should be room like a library at a university, not an individual study)",
        'j':"laundryroom/mudroom (place where people do laundry, etc.)",
        'k':"kitchen",#6
        'l':"living room (should be the main “showcase” living room in a house, not any area with couches)",#5
        'm':"meetingroom/conferenceroom",
        'n':"lounge (any area where people relax in comfy chairs/couches that is not the family room or living room",#5
        'o':"office (usually for an individual, or a small set of people)",#7
        'p':"porch/terrace/deck/driveway (must be outdoors on ground level)",
        'r':"rec/game (should have recreational objects, like pool table, etc.)",
        's':"stairs",
        't':"toilet (should be a small room with ONLY a toilet)",#1
        'u':"utilityroom/toolroom ",
        'v':"tv (must have theater-style seating)",#5
        'w':"workout/gym/exercise",
        'x':"outdoor areas containing grass, plants, bushes, trees, etc.",
        'y':"balcony (must be outside and must not be on ground floor)",
        'z':"other room (it is clearly a room, but the function is not clear)",
        'B':"bar",
        'C':"classroom",
        'D':"dining booth",#3
        'S':"spa/sauna",
        'Z':"junk (reflections of mirrors, random points floating in space, etc.)",
        '-':"no label",
    }

    def __init__(
        self,
        path: str,
        custom_classes: Optional[List[str]] = CLASS_LABELS_200,
        mode: str = 'train',
        mode_seed: int = 42,
        interval: int = 1,
    ):
        self._path = path

        if custom_classes:
            self._classes = custom_classes
        else:
            self._classes = CLASS_LABELS_200
        self.mode = mode
        np.random.seed(mode_seed)

        self._depth_images = []
        self._rgb_images = []
        # self._confidences = []
        
        self.interval = interval
        self.num_frames = -1
        # self._metadata = self._read_metadata()
        # self._metadata = self._read_metadata(w=320, h=256)
        self._metadata = self._read_metadata(w=640, h=512)
        # self._metadata = self._read_metadata(w=960, h=768)
        self._region_infos = self._read_region_infos()
        self._region_label2id = {label: i for i, (label, _) in enumerate(self.REGION_LABLE_MAPPING.items())}
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data()

        # region_list = ['living room', 'utility room', 'bedroom']
        self.region_id = []
        
        self.calculate_all_global_xyzs()
    
    def _read_metadata(self, w=1280, h=1024, interval=1):
        metadata_dict = {}
        with open(self._path + "/undistorted_camera_parameters/{}.conf".format(self._path.split("/")[-1]), 'r') as file:
            current_scan = None
            current_intrinsics = None
            for line in file:
                line = line.strip()
                if line.startswith('dataset'):
                    metadata_dict['dataset'] = line.split()[1]
                elif line.startswith('n_images'):
                    metadata_dict['n_images'] = int(line.split()[1])
                elif line.startswith('depth_directory'):
                    metadata_dict['depth_directory'] = line.split()[1]
                elif line.startswith('color_directory'):
                    metadata_dict['color_directory'] = line.split()[1]
                elif line.startswith('intrinsics_matrix'):
                    current_intrinsics = [float(x) for x in line.split()[1:]]
                elif line.startswith('scan'):
                    current_scan = {}
                    parts = line.split()
                    current_scan['depth_image'] = parts[1]
                    current_scan['color_image'] = parts[2]
                    current_scan['intrinsics_matrix'] = current_intrinsics
                    current_scan['pose'] = [float(x) for x in parts[3:]]
                    metadata_dict.setdefault('scans', []).append(current_scan)
        # metadata_dict['scans'] = metadata_dict['scans'][:18]
        metadata_dict['scans'] = metadata_dict['scans'][::self.interval]
        self.metadata_dict = metadata_dict

        self.rgb_width = 1280
        self.rgb_height = 1024
        self.fps = 15 # for test
        self.camera_matrixes = []
        self.total_poses = []
        for scan in metadata_dict['scans']:
            camera_matrix = np.array(scan["intrinsics_matrix"]).reshape([3, 3])
            if w > 0: 
                scale_x = w / self.rgb_width
                scale_y = h / self.rgb_height
                camera_matrix[0] *= scale_x
                camera_matrix[1] *= scale_y
            self.camera_matrixes.append(camera_matrix)

            pose = np.array(scan["pose"]).reshape([4, 4])
            self.total_poses.append(pose)
        self.rgb_width = w
        self.rgb_height = h

        self.image_size = (self.rgb_width, self.rgb_height)
        self.total_images = metadata_dict['n_images']
        self.poses = []
        for i in range(0, len(self.total_poses)):
            if i > self.num_frames and self.num_frames > 0:
                break
            # if i % interval == 0:
            self.poses.append(self.total_poses[i])
        self.init_pose = self.poses[0]

        self._id_to_name = {i: x for (i, x) in enumerate(self._classes)}

    def _read_region_infos(self):
        region_infos = []
        with open(self._path + "/house_segmentations/{}.house".format(self._path.split('/')[-1]), 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                entry_type = parts[0]
                # are we need level or other infomation?
                if entry_type == 'R': # only need region infos
                    region_infos.append({
                        'type': 'region',
                        'region_index': int(parts[1]),
                        'level_index': int(parts[2]),
                        'label': parts[5],
                        'position': [float(parts[6]), float(parts[7]), float(parts[8])],
                        # the bbox is (xlo, ylo, zlo, xhi, yhi, zhi), an axis-aligned bounding box.
                        'bbox': [float(parts[9]), float(parts[10]), float(parts[11]), float(parts[12]), float(parts[13]), float(parts[14])],
                        # height is the distance from the floor
                        'height': float(parts[15])
                    })
        return region_infos

    def load_image(self, filepath):
        image = Image.open(filepath)
        if image.size != self.image_size:
            image = image.resize(self.image_size)
        image_array = np.array(image)
        return image_array
    
    def load_depth(self, filepath):
        depth_image = Image.open(filepath)
        if depth_image.size != self.image_size:
            depth_image = depth_image.resize(self.image_size, resample=Image.Resampling.NEAREST)
        depth_array = np.array(depth_image, dtype=np.float32)
        return depth_array
    
    def _load_data(self, interval=1):
        assert self.fps  # Make sure metadata is read correctly first.
        i = 0
        for scan in self.metadata_dict['scans']:
            if i > self.num_frames and self.num_frames > 0:
                break
            # if i % interval == 0:
            # Read up the RGB and depth images.
            rgb_filepath = self._path + "/undistorted_color_images/" + scan['color_image']
            depth_filepath = self._path + "/undistorted_depth_images/" + scan['depth_image']
            # conf_filepath = self._path+f"color/{i}.conf"

            depth_img = self.load_depth(depth_filepath)
            # confidence = np.full_like(depth_img, 2)
            # confidence[depth_img == 0] == 0
            rgb_img = self.load_image(rgb_filepath)

            # Now, convert depth image to real world XYZ pointcloud.
            compatible_depth_scale = 1.0 / 4 # convert Matterport3D depth_scale 4000 to usual 1000
            mode_mask_ = np.random.rand(*depth_img.shape)
            if self.mode == 'train':
                mode_mask = mode_mask_ < 0.9
            elif self.mode == 'val':
                mode_mask = mode_mask_ >= 0.9
            self._depth_images.append(depth_img * compatible_depth_scale * mode_mask)
            # self._depth_images.append(depth_img * compatible_depth_scale)
            self._rgb_images.append(rgb_img)
            # self._confidences.append(confidence)
            i += 1
    
    def get_global_xyz(self, index, depth_scale=1.0, only_confident=True):
        reshaped_img = np.copy(self._depth_images[index])
        # If only confident, replace not confident points with nans
        # if only_confident:
        #     reshaped_img[self._confidences[index] != 2] = np.nan

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )
        rgb_o3d = o3d.geometry.Image(
            np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
        )
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width),
            height=int(self.rgb_height),
            fx=self.camera_matrixes[index][0, 0],
            fy=self.camera_matrixes[index][1, 1],
            cx=self.camera_matrixes[index][0, 2],
            cy=self.camera_matrixes[index][1, 2],
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )

        # Flip the pcd.  done when loading pose
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = self.poses[index]
        pcd.transform(extrinsic_matrix)
        world_xyz = np.asarray(pcd.points)

        # # Now transform everything by init pose.
        # init_matrix = np.linalg.inv(self.init_pose)
        # pcd.transform(init_matrix)

        # pose_trans = np.eye(4)
        # qx, qy, qz, qw, px, py, pz = [-0.1538211, -0.02841006, -0.01133235, 0.98762512, 0.0297593, -0.0121457, -0.01638832]
        # pose_trans[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
        # pose_trans[:3, -1] = [px, py, pz]
        # pcd.transform(pose_trans)

        return pcd, world_xyz

    def calculate_all_global_xyzs(self, interval=1, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(len(self.poses), desc="Calculating global XYZs"):
            global_xyz_pcd, world_xyz = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)

            xyz_region_id = []
            for xyz in world_xyz:
                for ri, region_info in enumerate(self._region_infos):
                    region_label = region_info['label']
                    region_bbox  = region_info['bbox']
                    if (xyz[0] >= region_bbox[0] and xyz[0] <= region_bbox[3]) and \
                       (xyz[1] >= region_bbox[1] and xyz[1] <= region_bbox[4]) and \
                       (xyz[2] >= region_bbox[2] and xyz[2] <= region_bbox[5]):
                        xyz_region_id.append(self._region_label2id[region_label])
                        break
                    if (ri + 1) == len(self._region_infos):
                        xyz_region_id.append(self._region_label2id['-'])
            self.region_id.append([xyz_region_id])

            self.global_xyzs.append(global_xyz)
            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds
    
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._depth_images[idx],
            # "conf": self._confidences[idx],
            "camera_matrix": self.camera_matrixes[idx],
            "camera_pose": self.poses[idx],
            # "xyz_clip_token": self.region_clip_tokens[idx],
            # "xyz_st_token": self.region_st_tokens[idx],
            "xyz_region_id": self.region_id[idx],
        }
        return result


if __name__ == "__main__":
    CUSTOM_LABELS = [
        "kitchen counter",
        "kitchen cabinet",
        "stove",
        "cabinet",
        "bathroom counter",
        "refrigerator",
        "microwave",
        "oven",
        "fireplace",
        "door",
        "sink",
        "furniture",
        "dish rack",
        "dining table",
        "shelf",
        "bar",
        "dishwasher",
        "toaster oven",
        "toaster",
        "mini fridge",
        "soap dish",
        "coffee maker",
        "table",
        "bowl",
        "rack",
        "bulletin board",
        "water cooler",
        "coffee kettle",
        "lamp",
        "plate",
        "window",
        "dustpan",
        "trash bin",
        "ceiling",
        "doorframe",
        "trash can",
        "basket",
        "wall",
        "bottle",
        "broom",
        "bin",
        "paper",
        "storage container",
        "box",
        "tray",
        "whiteboard",
        "decoration",
        "board",
        "cup",
        "windowsill",
        "potted plant",
        "light",
        "machine",
        "fire extinguisher",
        "bag",
        "paper towel roll",
        "chair",
        "book",
        "fire alarm",
        "blinds",
        "crate",
        "tissue box",
        "towel",
        "paper bag",
        "column",
        "fan",
        "object",
        "range hood",
        "plant",
        "structure",
        "poster",
        "mat",
        "water bottle",
        "power outlet",
        "storage bin",
        "radiator",
        "picture",
        "water pitcher",
        "pillar",
        "light switch",
        "bucket",
        "storage organizer",
        "vent",
        "counter",
        "ceiling light",
        "case of water bottles",
        "pipe",
        "scale",
        "recycling bin",
        "clock",
        "sign",
        "folded chair",
        "power strip",
    ]
    dataset = ApartmentDataset('data/Apartment', CUSTOM_LABELS)