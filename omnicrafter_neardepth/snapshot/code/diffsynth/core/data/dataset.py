"""Training dataset for FrameCrafter.

Loads scenes laid out in the DL3DV-10K-960P fashion::

    <dataset_base_path>/
        <scene_a>/
            images_4/             # rgb frames, sorted alphabetically
            transforms.json       # nerfstudio-style intrinsics + per-frame
                                  # c2w transform_matrix (OpenGL convention)
        <scene_b>/
            ...

For every sampled scene the dataset:
  1. Picks a window of consecutive frames according to ``sampling_strategy``.
  2. Draws ``num_frames`` (M + N) frames from that window without replacement;
     the first M become context, the last N become prediction targets.
  3. Loads + crops + resizes the RGB frames and the matching camera intrinsics.
  4. Builds a Plucker raymap by normalising the M+N poses so the LAST camera
     sits at the world origin (cam-last-origin convention; the prediction
     target is therefore at identity), then pixel-unshuffles to 1/8 resolution.

The returned dict has the keys consumed by the FrameCrafter pipeline and
flow-matching loss:
  * ``input_images``   : list[PIL.Image] of length M (context)
  * ``target_images``  : list[PIL.Image] of length M + N (context then target)
  * ``raymap``         : (M+N, 6*64, H/8, W/8) Plucker features
  * ``prompt``         : "" (empty, unconditional)
"""

import glob
import json
import os
import random

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from accelerate.logging import get_logger
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

random.seed(1234)

logger = get_logger("trainer", "INFO")


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

def _camera_to_raymap(Ks: Tensor, camtoworlds: Tensor, height: int, width: int):
    dtype = Ks.dtype
    x, y = torch.meshgrid(
        torch.arange(width, device=Ks.device),
        torch.arange(height, device=Ks.device),
        indexing="xy",
    )
    coords = torch.stack([x + 0.5, y + 0.5, torch.ones_like(x)], dim=-1).to(dtype)
    dirs = torch.einsum("...ij,...hwj->...hwi", Ks.float().inverse().to(dtype), coords)
    dirs = torch.einsum("...ij,...hwj->...hwi", camtoworlds[..., :3, :3], dirs)
    dirs = F.normalize(dirs, p=2, dim=-1)
    origins = torch.broadcast_to(camtoworlds[..., None, None, :3, -1], dirs.shape)
    return torch.cat([origins, dirs], dim=-1)


def _equirectangular_camera_to_raymap(camtoworlds: Tensor, height: int, width: int):
    """Generate rays for a full 2:1 equirectangular image.

    Camera coordinates follow OpenCV (x right, y down, z forward). Longitude
    spans [-pi, pi] from the left to right image edge and latitude spans
    [pi/2, -pi/2] from top to bottom. Pixel centers are used so no sample lies
    exactly on the duplicated longitude seam or either pole.
    """
    dtype = camtoworlds.dtype
    x, y = torch.meshgrid(
        torch.arange(width, device=camtoworlds.device, dtype=torch.float32),
        torch.arange(height, device=camtoworlds.device, dtype=torch.float32),
        indexing="xy",
    )
    longitude = (x + 0.5) * (2.0 * torch.pi / width) - torch.pi
    latitude = torch.pi / 2.0 - (y + 0.5) * (torch.pi / height)
    cos_latitude = torch.cos(latitude)
    dirs_camera = torch.stack(
        [
            cos_latitude * torch.sin(longitude),
            -torch.sin(latitude),
            cos_latitude * torch.cos(longitude),
        ],
        dim=-1,
    ).to(dtype)
    dirs_world = torch.einsum(
        "...ij,hwj->...hwi", camtoworlds[..., :3, :3], dirs_camera
    )
    dirs_world = F.normalize(dirs_world, p=2, dim=-1)
    origins = torch.broadcast_to(
        camtoworlds[..., None, None, :3, -1], dirs_world.shape
    )
    return torch.cat([origins, dirs_world], dim=-1)


def _raymap_to_plucker(raymap: Tensor) -> Tensor:
    ray_origins, ray_directions = torch.split(raymap, [3, 3], dim=-1)
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
    return torch.cat([ray_directions, plucker_normal], dim=-1)


def get_plucker_rays(pose, intrinsic, height, width, no_pixel_unshuffle=False, downsample_factor=8):
    """Compute Plucker features for ``pose`` (c2w) at original resolution and
    downsample to 1/``downsample_factor`` via PixelUnshuffle (or bilinear)."""
    raymap = _camera_to_raymap(intrinsic, pose, height=height, width=width)
    plucker = _raymap_to_plucker(raymap).permute(0, 3, 1, 2)
    if not no_pixel_unshuffle:
        plucker = torch.nn.PixelUnshuffle(downscale_factor=downsample_factor)(plucker)
    else:
        plucker = F.interpolate(plucker, scale_factor=1 / 8, mode="bilinear", align_corners=False)
    return plucker


def get_equirectangular_plucker_rays(
    pose, height, width, no_pixel_unshuffle=False, downsample_factor=8
):
    """Compute pixel-unshuffled Plucker features for 360-degree panoramas."""
    if width != 2 * height:
        raise ValueError(
            f"Equirectangular training requires a 2:1 image, got {width}x{height}"
        )
    raymap = _equirectangular_camera_to_raymap(
        pose, height=height, width=width
    )
    plucker = _raymap_to_plucker(raymap).permute(0, 3, 1, 2)
    if not no_pixel_unshuffle:
        plucker = torch.nn.PixelUnshuffle(
            downscale_factor=downsample_factor
        )(plucker)
    else:
        plucker = F.interpolate(
            plucker,
            scale_factor=1 / downsample_factor,
            mode="bilinear",
            align_corners=False,
        )
    return plucker


@torch.no_grad()
def normalize_w2c_make_cam_last_origin(w2c: torch.Tensor):
    """Rotate / translate / uniformly scale ``w2c`` so the LAST camera is at
    the world origin and the mean camera-center distance is 1.

    Returns ``(w2c_norm, c2w_norm, scale)``.
    """
    assert w2c.ndim == 3 and w2c.shape[-2:] == (4, 4)
    device, dtype = w2c.device, w2c.dtype

    c2w = torch.linalg.inv(w2c)
    R = c2w[:, :3, :3]
    t = c2w[:, :3, 3]

    R0 = R[-1]
    t0 = t[-1]

    R_align = R0.transpose(0, 1)
    t_shift = t - t0
    t_rot = (R_align @ t_shift.unsqueeze(-1)).squeeze(-1)
    R_rot = R_align @ R

    dists = t_rot.norm(dim=-1)
    scale = dists.mean().clamp_min(1e-12)
    t_norm = t_rot / scale

    c2w_norm = torch.zeros_like(c2w)
    c2w_norm[:, :3, :3] = R_rot
    c2w_norm[:, :3, 3] = t_norm
    c2w_norm[:, 3, :] = torch.tensor([0, 0, 0, 1], device=device, dtype=dtype)

    w2c_norm = torch.linalg.inv(c2w_norm)
    return w2c_norm, c2w_norm, scale


# ---------------------------------------------------------------------------
# Frame / image loading helpers
# ---------------------------------------------------------------------------

class LoadVideo:
    """Load all frames from an image folder or a video file as PIL images."""

    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor

    def _get_num_frames(self, reader):
        num_frames = self.num_frames
        total = int(reader.count_frames())
        if total < num_frames:
            num_frames = total
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames

    @staticmethod
    def list_image_files(folder_path):
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        return sorted(image_files)

    @staticmethod
    def load_image(image_path):
        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    def _load_from_folder(self, folder_path):
        image_files = self.list_image_files(folder_path)
        frames = []
        for img_path in image_files:
            try:
                frame = self.load_image(img_path)
                frames.append(self.frame_processor(frame))
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to load image {img_path}: {e}")
        return frames

    def __call__(self, data: str):
        if os.path.isdir(data):
            return self._load_from_folder(data)
        reader = imageio.get_reader(data)
        num_frames = self._get_num_frames(reader)
        frames = []
        for frame_id in range(num_frames):
            frame = Image.fromarray(reader.get_data(frame_id))
            frames.append(self.frame_processor(frame))
        reader.close()
        return frames


class ImageCropAndResize:
    """Resize-to-cover + center-crop a PIL image to (height, width)."""

    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def _crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def _get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width

    def __call__(self, data: Image.Image):
        return self._crop_and_resize(data, *self._get_height_width(data))


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class WanNVSDataset(Dataset):
    """Multi-view dataset for FrameCrafter training.

    Each scene under ``base_path`` is expected to contain ``images_4/`` (RGB
    frames) and ``transforms.json`` (nerfstudio-style intrinsics + c2w poses
    in OpenGL convention).
    """

    def __init__(
        self,
        base_path,
        metadata_path,
        repeat,
        num_frames,
        height,
        width,
        height_division_factor,
        width_division_factor,
        time_division_factor,
        time_division_remainder,
        sampling_strategy="prob_random",
        num_dataset_samples=1000,
        no_pixel_unshuffle=False,
        num_input_frames=None,
        num_output_frames=None,
        min_input_frames=3,
        min_output_frames=1,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.num_frames = num_frames
        self.height = height
        self.width = width

        # M-to-N split:
        #   Both None  -> random split per sample
        #   M fixed    -> deterministic M-to-(num_frames-M)
        self.random_split = (num_input_frames is None and num_output_frames is None)
        if self.random_split:
            self.num_input_frames = None
            self.num_output_frames = None
            self.min_input_frames = min_input_frames
            self.min_output_frames = min_output_frames
            assert min_input_frames + min_output_frames <= num_frames, (
                f"min_input ({min_input_frames}) + min_output ({min_output_frames}) > num_frames ({num_frames})"
            )
        elif num_input_frames is not None:
            self.num_input_frames = num_input_frames
            self.num_output_frames = num_output_frames if num_output_frames is not None else 1
            assert self.num_input_frames + self.num_output_frames == self.num_frames, (
                f"num_input_frames ({self.num_input_frames}) + num_output_frames "
                f"({self.num_output_frames}) != num_frames ({self.num_frames})"
            )
        else:
            num_output_frames = num_output_frames if num_output_frames is not None else 1
            self.num_input_frames = num_frames - num_output_frames
            self.num_output_frames = num_output_frames

        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.no_pixel_unshuffle = no_pixel_unshuffle
        self.load_from_cache = False

        # Sampling strategy for the temporal window length:
        #   all_random  : always use the full available range
        #   prob_random : 80% full range, 20% window [24, 48]
        #   all_window  : always window [24, 48]
        #   curriculum  : first half of epochs -> window, second half -> full
        assert sampling_strategy in ("all_random", "prob_random", "all_window", "curriculum"), (
            f"Unknown sampling_strategy: {sampling_strategy}"
        )
        self.sampling_strategy = sampling_strategy

        # The runner sets these so the curriculum schedule has context.
        self.current_epoch = 0
        self.num_epochs = 1

        scenes = sorted(
            os.path.join(base_path, f, "images_4")
            for f in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, f))
        )
        self.video_list = scenes[:num_dataset_samples]
        print(f"Total number of videos: {len(self.video_list)}")

    # ---- camera params ----------------------------------------------------

    def _scale_intrinsics(self, fx, fy, cx, cy, orig_width, orig_height, target_width, target_height):
        scale = max(target_width / orig_width, target_height / orig_height)
        resized_width = round(orig_width * scale)
        resized_height = round(orig_height * scale)
        fx_s, fy_s = fx * scale, fy * scale
        cx_s, cy_s = cx * scale, cy * scale
        crop_x = (resized_width - target_width) / 2.0
        crop_y = (resized_height - target_height) / 2.0
        return np.array([
            [fx_s, 0, cx_s - crop_x],
            [0, fy_s, cy_s - crop_y],
            [0, 0, 1],
        ])

    def _load_camera_parameters(self, video_name, frame_indices):
        transforms_path = video_name.replace("images_4", "transforms.json")
        if not os.path.exists(transforms_path):
            logger.warning(f"transforms.json not found: {transforms_path}")
            return None, None
        try:
            with open(transforms_path, "r") as f:
                td = json.load(f)
            projection = td.get("projection", "perspective").lower()
            if projection == "equirectangular":
                K = None
            elif projection == "perspective":
                K = self._scale_intrinsics(
                    td["fl_x"], td["fl_y"], td["cx"], td["cy"],
                    td["w"], td["h"], self.width, self.height,
                )
            else:
                raise ValueError(f"Unsupported camera projection: {projection}")
            frames_data = td["frames"]
            intrinsics_list, camera_poses_list = [], []
            for frame_idx in frame_indices:
                if frame_idx >= len(frames_data):
                    logger.warning(
                        f"Frame index {frame_idx} out of range for {video_name} "
                        f"(has {len(frames_data)} frames)"
                    )
                    return None, None, None
                fd = frames_data[frame_idx]
                if K is not None:
                    intrinsics_list.append(K)
                camera_poses_list.append(np.array(fd["transform_matrix"], dtype=np.float32))
            intrinsics = np.stack(intrinsics_list, axis=0) if intrinsics_list else None
            return intrinsics, np.stack(camera_poses_list, axis=0), projection
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error loading camera parameters from transforms.json: {e}")
            return None, None, None

    def _get_projection(self, video_path):
        transforms_path = video_path.replace("images_4", "transforms.json")
        try:
            with open(transforms_path, "r") as f:
                return json.load(f).get("projection", "perspective").lower()
        except Exception:  # noqa: BLE001
            return "perspective"

    def _get_num_camera_frames(self, video_path):
        transforms_path = video_path.replace("images_4", "transforms.json")
        if not os.path.exists(transforms_path):
            return None
        try:
            with open(transforms_path, "r") as f:
                return len(json.load(f)["frames"])
        except Exception:  # noqa: BLE001
            return None

    # ---- dataset interface ------------------------------------------------

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        input_video_path = self.video_list[idx]

        image_files = None
        video_frames = None
        if os.path.isdir(input_video_path):
            image_files = LoadVideo.list_image_files(input_video_path)
            num_frames = len(image_files)
        else:
            video_frames = LoadVideo(
                num_frames=self.num_frames,
                time_division_factor=self.time_division_factor,
                time_division_remainder=self.time_division_remainder,
            )(input_video_path)
            num_frames = len(video_frames)
        if num_frames == 0:
            raise RuntimeError(f"Empty video for idx {idx}: {input_video_path}")

        num_camera_frames = self._get_num_camera_frames(input_video_path)
        if num_camera_frames is not None:
            num_frames = min(num_frames, num_camera_frames)

        # Choose temporal window size.
        if self.sampling_strategy == "all_random":
            sep_num = num_frames
        elif self.sampling_strategy == "prob_random":
            sep_num = num_frames if random.random() < 0.8 else random.randint(24, 48)
        elif self.sampling_strategy == "all_window":
            sep_num = random.randint(24, 48)
        else:  # curriculum
            if self.current_epoch < self.num_epochs // 2:
                sep_num = random.randint(24, 48)
            else:
                sep_num = num_frames
        sep_num = max(self.num_frames, min(sep_num, num_frames))
        start_idx = random.randint(0, num_frames - sep_num)
        window_indices = list(range(start_idx, start_idx + sep_num))

        if self.random_split:
            max_input = self.num_frames - self.min_output_frames
            cur_num_input = random.randint(self.min_input_frames, max_input)
            cur_num_output = self.num_frames - cur_num_input
        else:
            cur_num_input = self.num_input_frames
            cur_num_output = self.num_output_frames

        # Random M+N draw from the window (no replacement).
        pick = np.random.choice(sep_num, self.num_frames, replace=False)
        sampled_indices = [window_indices[i] for i in pick]

        if image_files is not None:
            sampled_frames = [LoadVideo.load_image(image_files[i]) for i in sampled_indices]
        else:
            sampled_frames = [video_frames[i] for i in sampled_indices]

        projection = self._get_projection(input_video_path)
        if projection == "equirectangular":
            if self.width != 2 * self.height:
                raise ValueError(
                    f"Equirectangular training requires a 2:1 image, got "
                    f"{self.width}x{self.height}"
                )
            sampled_images = [
                torchvision.transforms.functional.resize(
                    frame,
                    (self.height, self.width),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                )
                for frame in sampled_frames
            ]
        else:
            process = ImageCropAndResize(
                height=self.height, width=self.width, max_pixels=1920 * 1080,
                height_division_factor=self.height_division_factor,
                width_division_factor=self.width_division_factor,
            )
            sampled_images = [process(frame) for frame in sampled_frames]

        context_frames = sampled_images[:cur_num_input]
        target_frames = sampled_images[cur_num_input:]
        # cam-last-origin: context then target, so the last frame is the
        # prediction target.
        target_images = context_frames + target_frames
        input_images = context_frames

        intrinsics, camera_poses, projection = self._load_camera_parameters(
            input_video_path, sampled_indices
        )
        if camera_poses is None:
            logger.warning(f"Could not load camera parameters for {input_video_path}; using zeros")
            camera_conditions = torch.zeros(
                self.num_frames,
                6 * self.height_division_factor * self.width_division_factor,
                self.height // self.height_division_factor,
                self.width // self.width_division_factor,
            )
        else:
            camera_poses = torch.from_numpy(camera_poses).float()
            w2cs = torch.linalg.inv(camera_poses)
            # transforms.json stores c2w in OpenGL convention; flip y/z to
            # land in OpenCV (x-right, y-down, z-forward).
            w2cs[:, [1, 2], :] *= -1
            _, c2w_norm, _ = normalize_w2c_make_cam_last_origin(w2cs)
            if projection == "equirectangular":
                camera_conditions = get_equirectangular_plucker_rays(
                    c2w_norm,
                    height=self.height,
                    width=self.width,
                    no_pixel_unshuffle=self.no_pixel_unshuffle,
                )
            else:
                camera_conditions = get_plucker_rays(
                    c2w_norm,
                    torch.from_numpy(intrinsics).float(),
                    height=self.height, width=self.width,
                    no_pixel_unshuffle=self.no_pixel_unshuffle,
                )

        return {
            "input_images": input_images,
            "target_images": target_images,
            "raymap": camera_conditions,
            "prompt": "",
        }
