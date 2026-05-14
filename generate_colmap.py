# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import glob
import os
import random
from pathlib import Path

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
import trimesh
from tqdm import tqdm
from vggt.dependency.np_to_pycolmap import (
    batch_np_matrix_to_pycolmap,
    batch_np_matrix_to_pycolmap_wo_track,
)
from vggt.dependency.track_predict import predict_tracks
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


_PRETRAINED_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VGGT batch inference + bundle adjustment demo"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing sorted input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write COLMAP sparse outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of images to infer in each batch",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_ba",
        action="store_true",
        default=False,
        help="Skip bundle adjustment and only export initial reconstruction",
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Maximum reprojection error for BA in pixels",
    )
    parser.add_argument(
        "--vis_thresh",
        type=float,
        default=0.2,
        help="Visibility threshold for track selection",
    )
    parser.add_argument(
        "--query_frame_num",
        type=int,
        default=8,
        help="Number of query frames for track prediction",
    )
    parser.add_argument(
        "--max_query_pts",
        type=int,
        default=4096,
        help="Maximum number of query points for tracking",
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Use fine tracking for predicted tracks",
    )
    parser.add_argument(
        "--conf_thres_value",
        type=float,
        default=5.0,
        help="Confidence threshold for point selection if BA is skipped",
    )
    parser.add_argument(
        "--img_load_resolution",
        type=int,
        default=1024,
        help="Resolution used when loading and preprocessing images",
    )
    parser.add_argument(
        "--model_resolution",
        type=int,
        default=518,
        help="Resolution used by VGGT for pose and depth inference",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Force shared camera model for BA reconstruction",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="Camera model type used for COLMAP reconstruction",
    )
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def batch_image_paths(image_paths, batch_size):
    for idx in range(0, len(image_paths), batch_size):
        yield image_paths[idx : idx + batch_size]


def run_vggt_on_batch(model, images, dtype, resolution=518):
    assert images.ndim == 4 and images.shape[1] == 3
    images = F.interpolate(
        images, size=(resolution, resolution), mode="bilinear", align_corners=False
    )

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
):
    rescale_camera = True

    for pyimageid in tqdm(
        reconstruction.images, desc="Rescaling cameras and renaming images"
    ):
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)
            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp
            pycamera.params = pred_params
            pycamera.width = int(real_image_size[0])
            pycamera.height = int(real_image_size[1])

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


def run_batch_inference(
    model, image_paths, batch_size, img_load_resolution, model_resolution, device, dtype
):
    all_extrinsics = []
    all_intrinsics = []
    all_depth_maps = []
    all_depth_confs = []
    all_images = []
    all_original_coords = []

    num_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch_paths in tqdm(
        batch_image_paths(image_paths, batch_size),
        total=num_batches,
        desc="Running VGGT inference",
    ):
        images, original_coords = load_and_preprocess_images_square(
            batch_paths, img_load_resolution
        )
        images = images.to(device)
        extrinsic, intrinsic, depth_map, depth_conf = run_vggt_on_batch(
            model, images, dtype, model_resolution
        )
        torch.cuda.empty_cache()

        all_extrinsics.append(extrinsic)
        all_intrinsics.append(intrinsic)
        all_depth_maps.append(depth_map)
        all_depth_confs.append(depth_conf)
        all_images.append(images.cpu())
        all_original_coords.append(original_coords)

    extrinsics = np.concatenate(all_extrinsics, axis=0)
    intrinsics = np.concatenate(all_intrinsics, axis=0)
    depth_maps = np.concatenate(all_depth_maps, axis=0)
    depth_confs = np.concatenate(all_depth_confs, axis=0)
    images = torch.cat(all_images, dim=0)
    original_coords = torch.cat(all_original_coords, dim=0)
    return images, original_coords, extrinsics, intrinsics, depth_maps, depth_confs


def build_reconstruction(
    images,
    original_coords,
    extrinsics,
    intrinsics,
    depth_maps,
    depth_confs,
    args,
    device,
    dtype,
):
    num_images, _, height, width = images.shape
    points_3d = unproject_depth_map_to_point_map(depth_maps, extrinsics, intrinsics)

    if args.skip_ba:
        print("Skipping bundle adjustment: exporting initial reconstruction only.")
        image_size = np.array([args.model_resolution, args.model_resolution])
        points_rgb = F.interpolate(
            images.to(torch.float32),
            size=(args.model_resolution, args.model_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (
            (points_rgb.cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        )
        points_xyf = create_pixel_coordinate_grid(
            num_images, args.model_resolution, args.model_resolution
        )
        conf_mask = depth_confs >= args.conf_thres_value
        conf_mask = randomly_limit_trues(conf_mask, 100000)
        points_3d_filtered = points_3d[conf_mask]
        points_xyf_filtered = points_xyf[conf_mask]
        points_rgb_filtered = points_rgb[conf_mask]
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d_filtered,
            points_xyf_filtered,
            points_rgb_filtered,
            extrinsics,
            intrinsics,
            image_size,
            shared_camera=args.shared_camera,
            camera_type=args.camera_type,
        )
        reconstruction_resolution = args.model_resolution
        return (
            reconstruction,
            points_3d_filtered,
            points_rgb_filtered,
            reconstruction_resolution,
        )

    images_device = images.to(device)
    with torch.cuda.amp.autocast(dtype=dtype):
        pred_tracks, pred_vis_scores, pred_confs, points_3d_tracks, points_rgb = (
            predict_tracks(
                images_device,
                conf=depth_confs,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )
        )
    torch.cuda.empty_cache()

    scale = args.img_load_resolution / args.model_resolution
    intrinsics = intrinsics.copy()
    intrinsics[:, :2, :] *= scale
    image_size = np.array([args.img_load_resolution, args.img_load_resolution])
    track_mask = pred_vis_scores > args.vis_thresh

    reconstruction, valid_mask = batch_np_matrix_to_pycolmap(
        points_3d_tracks,
        extrinsics,
        intrinsics,
        pred_tracks,
        image_size,
        masks=track_mask,
        max_reproj_error=args.max_reproj_error,
        shared_camera=args.shared_camera,
        camera_type=args.camera_type,
        points_rgb=points_rgb,
    )

    if reconstruction is None:
        raise RuntimeError(
            "Bundle adjustment reconstruction failed. Not enough inliers."
        )

    print("Running bundle adjustment on the full sequence...")
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)
    reconstruction_resolution = args.img_load_resolution
    return reconstruction, points_3d_tracks, points_rgb, reconstruction_resolution


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    image_paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {args.image_dir}")

    print(
        f"Found {len(image_paths)} images. Running inference in batches of {args.batch_size}."
    )

    print("Loading VGGT model...")
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(_PRETRAINED_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded and moved to device.")

    images, original_coords, extrinsics, intrinsics, depth_maps, depth_confs = (
        run_batch_inference(
            model,
            image_paths,
            args.batch_size,
            args.img_load_resolution,
            args.model_resolution,
            device,
            dtype,
        )
    )

    print("\nInference complete. Building reconstruction from all batches...")
    reconstruction, points_3d, points_rgb, reconstruction_resolution = (
        build_reconstruction(
            images,
            original_coords.cpu().numpy(),
            extrinsics,
            intrinsics,
            depth_maps,
            depth_confs,
            args,
            device,
            dtype,
        )
    )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        [Path(p).name for p in image_paths],
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=args.shared_camera,
    )

    sparse_dir = os.path.join(args.output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    trimesh.PointCloud(points_3d, colors=points_rgb).export(
        os.path.join(sparse_dir, "points.ply")
    )

    print(f"Saved sparse reconstruction to {sparse_dir}")


if __name__ == "__main__":
    main()
