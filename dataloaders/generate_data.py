import sys
sys.path.append('..')

import torch
from record3d import ApartmentDataset
from real_dataset import DeticDenseLabelledRegionDataset
import open3d as o3d
# Load the dataset
# If you are following up after tutorial 1
# dataset = torch.load("../apartment_views.pth")
# Otherwise, create from scratch.
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
DATA_PATH = '../data/Apartment'
dataset = ApartmentDataset(DATA_PATH, CUSTOM_LABELS)

# labelled_dataset = DeticDenseLabelledDataset(
#     dataset, 
#     use_extra_classes=False, 
#     exclude_gt_images=False, 
#     use_lseg=False, 
#     subsample_prob=0.01, 
#     visualize_results=True, 
#     detic_threshold=0.4,
#     visualization_path="detic_labelled_apartment_results",
# )

labelled_dataset = DeticDenseLabelledRegionDataset(
    dataset, 
    use_extra_classes=False, 
    exclude_gt_images=False, 
    use_lseg=False, 
    subsample_prob=0.01, 
    visualize_results=True, 
    detic_threshold=0.4,
    visualization_path="detic_labelled_apartment_results",
)

merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(labelled_dataset._label_xyz)
merged_pcd.colors = o3d.utility.Vector3dVector(labelled_dataset._label_rgb)
merged_downpcd = merged_pcd.voxel_down_sample(voxel_size=0.03)
o3d.io.write_point_cloud("2_point_cloud.ply", merged_pcd)
o3d.io.write_point_cloud("2_point_cloud_down.ply", merged_downpcd)

import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# NUM_IMAGES = 10
# TOTAL_FRAMES = glob.glob("detic_labelled_apartment_results/*.jpg")
# fig, axes = plt.subplots(NUM_IMAGES, 1, figsize=(3, 3 * NUM_IMAGES))

# for ax, data in zip(axes, range(NUM_IMAGES)):
#     random_index = np.random.randint(0, len(TOTAL_FRAMES))
#     ax.imshow(Image.open(TOTAL_FRAMES[random_index]))
#     ax.axis("off")
#     ax.set_title(f"Frame {random_index}")

torch.save(labelled_dataset, "../detic_labeled_apartment_dataset_region.pt")