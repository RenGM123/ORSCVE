import os
import glob
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

class _BaseORSLDBDataset(Dataset):
    """

    """

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        gop: int = 8,
        gt_size: Optional[int] = 256,
        use_flip: bool = True,
        use_rot: bool = True,
        pad_value_y: float = 0.0,
        pad_value_res: float = 128.0 / 255.0,
        lq_dir_name: str = "compress",
        res_dir_name: str = "residue",
        gt_subdir_name: Optional[str] = "HQ",
    ):
        super().__init__()

        assert int(gop) >= 1, "gop must be >= 1"

        self.lq_root = lq_root
        self.gt_root = gt_root
        self.gop = int(gop)
        self.gt_size = gt_size
        self.use_flip = use_flip
        self.use_rot = use_rot
        self.pad_value_y = float(pad_value_y)
        self.pad_value_res = float(pad_value_res)

        self.lq_dir_name = lq_dir_name
        self.res_dir_name = res_dir_name
        self.gt_subdir_name = gt_subdir_name

        self.items: List[Tuple[int, str, int, int]] = []
        self.videos: List[str] = []

        video_names = sorted(
            name for name in os.listdir(lq_root)
            if os.path.isdir(os.path.join(lq_root, name))
        )

        assert len(video_names) > 0, f"No video directory found in {lq_root}."

        for video_idx, video_name in enumerate(video_names):
            lq_dir = os.path.join(lq_root, video_name, self.lq_dir_name)
            mv_dir = os.path.join(lq_root, video_name, "mv")
            res_dir = os.path.join(lq_root, video_name, self.res_dir_name)

            if self.gt_subdir_name is None:
                gt_dir = os.path.join(gt_root, video_name)
            else:
                gt_dir = os.path.join(gt_root, video_name, self.gt_subdir_name)

            if not os.path.isdir(lq_dir):
                print(f"[warn] {video_name} missing {self.lq_dir_name}, skipped.")
                continue
            if not os.path.isdir(mv_dir):
                print(f"[warn] {video_name} missing mv, skipped.")
                continue
            if not os.path.isdir(res_dir):
                print(f"[warn] {video_name} missing {self.res_dir_name}, skipped.")
                continue
            if not os.path.isdir(gt_dir):
                print(f"[warn] {video_name} missing GT directory, skipped.")
                continue

            lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))
            if len(lq_paths) == 0:
                print(f"[warn] {video_name} has no PNG frames in {lq_dir}, skipped.")
                continue

            num_frames = len(lq_paths)

            gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
            mv_paths = sorted(glob.glob(os.path.join(mv_dir, "*.npy")))
            res_paths = sorted(glob.glob(os.path.join(res_dir, "*.npy")))

            if len(gt_paths) != num_frames:
                print(
                    f"[warn] {video_name} GT frames ({len(gt_paths)}) "
                    f"!= LQ frames ({num_frames}); using LQ frame count."
                )

            if len(mv_paths) != num_frames:
                print(
                    f"[warn] {video_name} MV files ({len(mv_paths)}) "
                    f"!= LQ frames ({num_frames})."
                )

            if len(res_paths) != num_frames:
                print(
                    f"[warn] {video_name} residual files ({len(res_paths)}) "
                    f"!= LQ frames ({num_frames})."
                )

            for frame_idx in range(num_frames):
                self.items.append((video_idx, video_name, frame_idx, num_frames))

            self.videos.append(video_name)

        assert len(self.items) > 0, "No valid frames found."

    def __len__(self):
        return len(self.items)

    def _select_ors_reference_index(self, frame_idx: int, num_frames: int) -> int:
        """
        LDB-specialized closed-form ORS reference selection.

        This is equivalent to the second reference branch in the previous
        implementation:
            ors_idx = min((frame_idx // gop + 1) * gop, num_frames - 1)

        Keeping this rule preserves the training data actually consumed
        by the previous ORS-CVE model.
        """
        gop_idx = frame_idx // self.gop
        ors_idx = min((gop_idx + 1) * self.gop, num_frames - 1)
        return ors_idx

    @staticmethod
    def _read_png_gray(path: str) -> np.ndarray:
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)

    @staticmethod
    def _read_npy(path: str) -> np.ndarray:
        return np.load(path)

    @staticmethod
    def _to_chw01(x_uint8: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(
            x_uint8.astype(np.float32) / 255.0
        ).unsqueeze(0)

    @staticmethod
    def _to_chw_mv(x: np.ndarray) -> torch.Tensor:
        """
        Convert MV to (C,H,W) float32.

        Supports:
            (H,W,2), (2,H,W), or (H,W).
        """
        if x.ndim == 2:
            x = x[..., None]

        if x.ndim != 3:
            raise ValueError(f"Unsupported MV shape: {x.shape}")

        if x.shape[0] in (1, 2, 3, 4) and x.shape[0] <= x.shape[-1]:
            arr = x.astype(np.float32)
        else:
            arr = np.transpose(x, (2, 0, 1)).astype(np.float32)

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if x.dtype in (
            np.int8,
            np.int16,
            np.int32,
            np.uint8,
            np.uint16,
            np.uint32,
        ):
            arr = arr / 127.0
        else:
            max_abs = np.max(np.abs(arr))
            if max_abs > 0:
                arr = np.clip(arr / max_abs, -1.0, 1.0)

        return torch.from_numpy(arr)

    @staticmethod
    def _fix_residual_shape(residual: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Normalize residual shape to (H,W).
        """
        if residual.ndim == 2:
            return residual

        if residual.ndim == 3 and residual.shape[0] == 1:
            _, h_like, w_like = residual.shape

            if (h_like, w_like) == (target_w, target_h):
                residual = np.transpose(residual, (0, 2, 1))
                h_like, w_like = residual.shape[1], residual.shape[2]

            if (h_like, w_like) == (target_h, target_w):
                residual = residual[0]

        if residual.ndim == 3 and residual.shape[-1] == 1:
            residual = residual[..., 0]

        return residual

    def _ensure_min_size(
        self,
        arr: np.ndarray,
        target_h: Optional[int],
        target_w: Optional[int],
        is_res: bool = False,
    ) -> np.ndarray:
        if target_h is None or target_w is None:
            return arr

        if arr.ndim == 2:
            h, w = arr.shape
            layout = "HW"
        elif arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] <= arr.shape[-1]:
                _, h, w = arr.shape
                layout = "CHW"
            else:
                h, w, _ = arr.shape
                layout = "HWC"
        else:
            return arr

        if h >= target_h and w >= target_w:
            return arr

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        if arr.ndim == 2:
            pad_value = self.pad_value_res if is_res else self.pad_value_y
            pad_value_255 = int(round(pad_value * 255.0))

            return np.pad(
                arr,
                ((top, bottom), (left, right)),
                mode="constant",
                constant_values=pad_value_255,
            )

        if layout == "HWC":
            return np.pad(
                arr,
                ((top, bottom), (left, right), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        return np.pad(
            arr,
            ((0, 0), (top, bottom), (left, right)),
            mode="constant",
            constant_values=0.0,
        )

    @staticmethod
    def _crop(arr: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
        if arr.ndim == 2:
            return arr[top:top + height, left:left + width]

        if arr.ndim == 3:
            if arr.shape[0] in (1, 2, 3, 4) and arr.shape[0] <= arr.shape[-1]:
                return arr[:, top:top + height, left:left + width]
            return arr[top:top + height, left:left + width, :]

        return arr

    @staticmethod
    def _random_crop_coords(h: int, w: int, crop_h: int, crop_w: int):
        top = 0 if h == crop_h else random.randint(0, h - crop_h)
        left = 0 if w == crop_w else random.randint(0, w - crop_w)
        return top, left, crop_h, crop_w

    def _do_aug(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply the same augmentation to all tensors.

        This keeps the previous augmentation behavior except that unused
        reference tensors are no longer carried through the pipeline.
        """
        if self.use_flip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[2]) for t in tensors]

        if self.use_flip and random.random() < 0.5:
            tensors = [torch.flip(t, dims=[1]) for t in tensors]

        if self.use_rot:
            k = random.randint(0, 3)
            if k:
                tensors = [torch.rot90(t, k, dims=[1, 2]) for t in tensors]

        return tensors

    def _get_dirs(self, video_name: str):
        lq_dir = os.path.join(self.lq_root, video_name, self.lq_dir_name)
        mv_dir = os.path.join(self.lq_root, video_name, "mv")
        res_dir = os.path.join(self.lq_root, video_name, self.res_dir_name)

        if self.gt_subdir_name is None:
            gt_dir = os.path.join(self.gt_root, video_name)
        else:
            gt_dir = os.path.join(self.gt_root, video_name, self.gt_subdir_name)

        return lq_dir, mv_dir, res_dir, gt_dir

    def __getitem__(self, index: int):
        video_idx, video_name, frame_idx, num_frames = self.items[index]

        lq_dir, mv_dir, res_dir, gt_dir = self._get_dirs(video_name)

        lq_paths = sorted(glob.glob(os.path.join(lq_dir, "*.png")))

        if len(lq_paths) == 0:
            raise RuntimeError(f"No PNG frames found in {lq_dir}.")

        if frame_idx >= len(lq_paths):
            raise RuntimeError(
                f"frame_idx={frame_idx}, but only {len(lq_paths)} frames found in {lq_dir}."
            )

        lq_path = lq_paths[frame_idx]
        frame_name = os.path.splitext(os.path.basename(lq_path))[0]

        ors_idx = self._select_ors_reference_index(frame_idx, num_frames)
        ors_name = os.path.splitext(os.path.basename(lq_paths[ors_idx]))[0]

        # Current frame.
        y_lq = self._read_png_gray(lq_path)

        gt_path = os.path.join(gt_dir, frame_name + ".png")
        y_gt = self._read_png_gray(gt_path)
        gt_h, gt_w = y_gt.shape

        mv_path = os.path.join(mv_dir, frame_name + ".npy")
        mv = self._read_npy(mv_path)

        if mv.ndim == 3 and mv.shape[0] in (1, 2, 3, 4) and mv.shape[0] <= mv.shape[-1]:
            mv = np.transpose(mv, (1, 2, 0))

        # ORS reference and its residual prior.
        y_ors_lq = self._read_png_gray(
            os.path.join(lq_dir, ors_name + ".png")
        )

        ors_res_path = os.path.join(res_dir, ors_name + ".npy")
        y_ors_res = self._read_npy(ors_res_path)
        y_ors_res = self._fix_residual_shape(y_ors_res, gt_h, gt_w)

        # Padding before crop.
        if self.gt_size is not None:
            y_lq = self._ensure_min_size(
                y_lq,
                self.gt_size,
                self.gt_size,
                is_res=False
            )
            y_gt = self._ensure_min_size(
                y_gt,
                self.gt_size,
                self.gt_size,
                is_res=False
            )
            y_ors_lq = self._ensure_min_size(
                y_ors_lq,
                self.gt_size,
                self.gt_size,
                is_res=False
            )
            y_ors_res = self._ensure_min_size(
                y_ors_res,
                self.gt_size,
                self.gt_size,
                is_res=True
            )
            mv = self._ensure_min_size(
                mv,
                self.gt_size,
                self.gt_size,
                is_res=False
            )

        crop_source_h, crop_source_w = y_gt.shape

        if self.gt_size is None:
            top, left, crop_h, crop_w = 0, 0, crop_source_h, crop_source_w
        else:
            top, left, crop_h, crop_w = self._random_crop_coords(
                crop_source_h,
                crop_source_w,
                self.gt_size,
                self.gt_size
            )

        # Synchronized crop.
        y_lq = self._crop(y_lq, top, left, crop_h, crop_w)
        y_gt = self._crop(y_gt, top, left, crop_h, crop_w)
        y_ors_lq = self._crop(y_ors_lq, top, left, crop_h, crop_w)
        y_ors_res = self._crop(y_ors_res, top, left, crop_h, crop_w)
        mv = self._crop(mv, top, left, crop_h, crop_w)

        # Tensor conversion.
        lq = self._to_chw01(y_lq)
        gt = self._to_chw01(y_gt)
        ors_lq = self._to_chw01(y_ors_lq)
        ors_res = torch.from_numpy(
            (y_ors_res.astype(np.float32) + 128.0) / 255.0
        ).unsqueeze(0)
        lq_mv = self._to_chw_mv(mv)

        tensors = [lq, gt, ors_lq, ors_res, lq_mv]
        lq, gt, ors_lq, ors_res, lq_mv = self._do_aug(tensors)

        sample = {
            "lq": lq,
            "lq_mv": lq_mv,
            "ors_lq": ors_lq,
            "ors_res": ors_res,
            "gt": gt,
            "video_name": video_name,
            "video_idx": video_idx,
            "frame_idx": frame_idx,
            "ors_idx": ors_idx,
        }

        return sample

class TrainORSLDBDataset(_BaseORSLDBDataset):
    """
    Training split for ORS-CVE.

    Expected directory structure:
        lq_root/
            video_name/
                compress/
                mv/
                residue/

        gt_root/
            video_name/
                HQ/
    """

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        gop: int = 8,
        gt_size: Optional[int] = 256,
        use_flip: bool = True,
        use_rot: bool = True,
        pad_value_y: float = 0.0,
        pad_value_res: float = 128.0 / 255.0,
    ):
        super().__init__(
            lq_root=lq_root,
            gt_root=gt_root,
            gop=gop,
            gt_size=gt_size,
            use_flip=use_flip,
            use_rot=use_rot,
            pad_value_y=pad_value_y,
            pad_value_res=pad_value_res,
            lq_dir_name="compress",
            res_dir_name="residue",
            gt_subdir_name="HQ",
        )

class ValORSLDBDataset(_BaseORSLDBDataset):
    """
    Validation split for ORS-CVE.

    Expected directory structure:
        lq_root/
            video_name/
                compress_y/
                mv/
                residual/

        gt_root/
            video_name/
    """

    def __init__(
        self,
        lq_root: str,
        gt_root: str,
        gop: int = 8,
        gt_size: Optional[int] = 256,
        use_flip: bool = True,
        use_rot: bool = True,
        pad_value_y: float = 0.0,
        pad_value_res: float = 128.0 / 255.0,
    ):
        super().__init__(
            lq_root=lq_root,
            gt_root=gt_root,
            gop=gop,
            gt_size=gt_size,
            use_flip=use_flip,
            use_rot=use_rot,
            pad_value_y=pad_value_y,
            pad_value_res=pad_value_res,
            lq_dir_name="compress_y",
            res_dir_name="residual",
            gt_subdir_name=None,
        )

