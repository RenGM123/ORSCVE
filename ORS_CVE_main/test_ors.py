# file: test_ors.py
import os
import argparse
import os.path as op
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from dataset.ors_ldb_dataset import ValORSLDBDataset as ValSet
#from dataset.ors_ra_dataset import ValORSRADataset as ValSet
from models.ors_cve import ORS_CVE


def _pad_to16(x: torch.Tensor):
    B, C, H, W = x.shape
    ph = (-H) % 16
    pw = (-W) % 16

    if ph == 0 and pw == 0:
        return x, (H, W)

    return F.pad(x, (0, pw, 0, ph), mode="replicate"), (H, W)


def _unpad(x: torch.Tensor, size_hw):
    H, W = size_hw
    return x[..., :H, :W]


def _gaussian_kernel(window_size=11, sigma=1.5, device="cpu", dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    return kernel_2d


@torch.no_grad()
def ssim_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size=11,
    sigma=1.5,
    C1=0.01 ** 2,
    C2=0.03 ** 2
):
    assert x.ndim == 4
    assert y.ndim == 4
    assert x.shape == y.shape
    assert x.shape[1] == 1

    device = x.device
    dtype = x.dtype

    window = _gaussian_kernel(
        window_size=window_size,
        sigma=sigma,
        device=device,
        dtype=dtype
    )

    pad = window_size // 2

    mu_x = F.conv2d(x, window, padding=pad, groups=1)
    mu_y = F.conv2d(y, window, padding=pad, groups=1)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=1) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=1) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=1) - mu_xy

    ssim_map = (
        (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    return ssim_map.mean(dim=[1, 2, 3])


def _strip_module_prefix(sd):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def _save_pred_y_png(pred01: torch.Tensor, save_path: str):
    from PIL import Image

    if pred01.ndim == 3:
        pred01 = pred01.squeeze(0)

    pred01 = pred01.clamp(0.0, 1.0)
    y_u8 = (pred01 * 255.0).round().to(torch.uint8).cpu().numpy()

    os.makedirs(op.dirname(save_path), exist_ok=True)
    Image.fromarray(y_u8, mode="L").save(save_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to ckp_xxxxx.pt"
    )

    parser.add_argument(
        "--lq_root",
        type=str,
        default="/tdx/rgm/data/VCP_dataset/test_18_data/LD/qp37"
    )

    parser.add_argument(
        "--gt_root",
        type=str,
        default="/tdx/rgm/data/VCP_dataset/test_18_data/GT"
    )

    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--save_csv",
        type=str,
        default="",
        help="optional: save per-video metrics csv"
    )

    parser.add_argument(
        "--save_y",
        action="store_true",
        help="if set, save enhanced Y as uint8 PNGs"
    )

    parser.add_argument(
        "--save_y_dir",
        type=str,
        default="./enhY_png",
        help="root folder to save enhanced Y PNGs"
    )

    parser.add_argument(
        "--save_y_pattern",
        type=str,
        default="{frame:05d}.png",
        help="filename pattern inside each video folder, e.g. {frame:05d}.png"
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")

    val_ds = ValSet(
        lq_root=args.lq_root,
        gt_root=args.gt_root,
        gop=args.gop,
        gt_size=None,
        use_flip=False,
        use_rot=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"[Info] val samples = {len(val_ds)}")

    model = ORS_CVE().to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = _strip_module_prefix(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)

    model.load_state_dict(sd, strict=True)
    model.eval()

    print(f"[Info] loaded ckpt: {args.ckpt}")

    psnr_fn = utils.PSNR()

    acc = defaultdict(lambda: {
        "n": 0,
        "psnr_lq_sum": 0.0,
        "psnr_pr_sum": 0.0,
        "ssim_lq_sum": 0.0,
        "ssim_pr_sum": 0.0,
    })

    pbar = tqdm(val_loader, ncols=120)

    for data in pbar:
        LQ = data["lq"].to(device, non_blocking=True)
        GT = data["gt"].to(device, non_blocking=True)
        ORS_LQ = data["ors_lq"].to(device, non_blocking=True)
        ORS_RES = data["ors_res"].to(device, non_blocking=True)
        LQMV = data["lq_mv"].to(device, non_blocking=True)

        video_names = data["video_name"]

        LQp, _ = _pad_to16(LQ)
        GTp, sz = _pad_to16(GT)
        ORS_LQp, _ = _pad_to16(ORS_LQ)
        ORS_RESp, _ = _pad_to16(ORS_RES)
        LQMVp, _ = _pad_to16(LQMV)

        pred_p = model(
            LQ=LQp,
            LQMV=LQMVp,
            ORS_LQ=ORS_LQp,
            ORS_RES=ORS_RESp,
            GT=GTp,
            test_mode=True,
            return_intermediates=False
        )

        if isinstance(pred_p, (tuple, list)):
            pred_p = pred_p[0]

        pred = _unpad(pred_p, sz).clamp(0.0, 1.0)

        B = LQ.shape[0]

        psnr_lq = [
            float(psnr_fn(LQ[i], GT[i]))
            for i in range(B)
        ]

        psnr_pr = [
            float(psnr_fn(pred[i], GT[i]))
            for i in range(B)
        ]

        ssim_lq = ssim_torch(
            LQ.clamp(0, 1),
            GT.clamp(0, 1)
        ).detach().cpu().numpy().tolist()

        ssim_pr = ssim_torch(
            pred.clamp(0, 1),
            GT.clamp(0, 1)
        ).detach().cpu().numpy().tolist()

        for i in range(B):
            vn = video_names[i]
            frame_idx = acc[vn]["n"]

            if args.save_y:
                out_dir = op.join(args.save_y_dir, vn)
                fname = args.save_y_pattern.format(frame=frame_idx)
                save_path = op.join(out_dir, fname)
                _save_pred_y_png(pred[i].detach(), save_path)

            acc[vn]["n"] += 1
            acc[vn]["psnr_lq_sum"] += psnr_lq[i]
            acc[vn]["psnr_pr_sum"] += psnr_pr[i]
            acc[vn]["ssim_lq_sum"] += ssim_lq[i]
            acc[vn]["ssim_pr_sum"] += ssim_pr[i]

        all_n = sum(v["n"] for v in acc.values())

        if all_n > 0:
            psnr_lq_avg = sum(v["psnr_lq_sum"] for v in acc.values()) / all_n
            psnr_pr_avg = sum(v["psnr_pr_sum"] for v in acc.values()) / all_n
            ssim_lq_avg = sum(v["ssim_lq_sum"] for v in acc.values()) / all_n
            ssim_pr_avg = sum(v["ssim_pr_sum"] for v in acc.values()) / all_n

            pbar.set_description(
                f"running ΔPSNR={psnr_pr_avg - psnr_lq_avg:.3f} | "
                f"running ΔSSIM*100={(ssim_pr_avg - ssim_lq_avg) * 100:.3f}"
            )

    print("\n<<<<<<<<<< Results >>>>>>>>>>")

    per_vid_psnr_lq = []
    per_vid_psnr_pr = []
    per_vid_ssim_lq = []
    per_vid_ssim_pr = []

    for vn in sorted(acc.keys()):
        n = acc[vn]["n"]

        psnr_lq_avg = acc[vn]["psnr_lq_sum"] / n
        psnr_pr_avg = acc[vn]["psnr_pr_sum"] / n
        dpsnr = psnr_pr_avg - psnr_lq_avg

        ssim_lq_avg = acc[vn]["ssim_lq_sum"] / n
        ssim_pr_avg = acc[vn]["ssim_pr_sum"] / n
        dssim100 = (ssim_pr_avg - ssim_lq_avg) * 100.0

        per_vid_psnr_lq.append(psnr_lq_avg)
        per_vid_psnr_pr.append(psnr_pr_avg)
        per_vid_ssim_lq.append(ssim_lq_avg)
        per_vid_ssim_pr.append(ssim_pr_avg)

        print(
            f"{vn}: [{psnr_lq_avg:.3f}] dB -> "
            f"[{psnr_pr_avg:.3f}] dB Delta:[{dpsnr:.3f}]"
        )
        print(
            f"SSIM: {vn}: [{ssim_lq_avg * 100:.3f}] dB -> "
            f"[{ssim_pr_avg * 100:.3f}] dB Delta:[{dssim100:.3f}]"
        )

    mean_psnr_lq = float(np.mean(per_vid_psnr_lq)) if len(per_vid_psnr_lq) else 0.0
    mean_psnr_pr = float(np.mean(per_vid_psnr_pr)) if len(per_vid_psnr_pr) else 0.0
    mean_dpsnr = mean_psnr_pr - mean_psnr_lq

    mean_ssim_lq = float(np.mean(per_vid_ssim_lq)) if len(per_vid_ssim_lq) else 0.0
    mean_ssim_pr = float(np.mean(per_vid_ssim_pr)) if len(per_vid_ssim_pr) else 0.0
    mean_dssim100 = (mean_ssim_pr - mean_ssim_lq) * 100.0

    print(f"> ori: [{mean_psnr_lq:.3f}] dB")
    print(f"> ave: [{mean_psnr_pr:.3f}] dB")
    print(f"> delta: [{mean_dpsnr:.3f}] dB")
    print(f"> ori: [{mean_ssim_lq * 100:.3f}] dB")
    print(f"> ave: [{mean_ssim_pr * 100:.3f}] dB")
    print(f"> delta: [{mean_dssim100:.3f}] dB")

    if args.save_csv:
        import csv

        if op.dirname(args.save_csv):
            os.makedirs(op.dirname(args.save_csv), exist_ok=True)

        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "video",
                "num_frames",
                "psnr_lq",
                "psnr_pred",
                "delta_psnr",
                "ssim_lq",
                "ssim_pred",
                "delta_ssim_x100",
            ])

            for vn in sorted(acc.keys()):
                n = acc[vn]["n"]

                psnr_lq_avg = acc[vn]["psnr_lq_sum"] / n
                psnr_pr_avg = acc[vn]["psnr_pr_sum"] / n

                ssim_lq_avg = acc[vn]["ssim_lq_sum"] / n
                ssim_pr_avg = acc[vn]["ssim_pr_sum"] / n

                writer.writerow([
                    vn,
                    n,
                    f"{psnr_lq_avg:.6f}",
                    f"{psnr_pr_avg:.6f}",
                    f"{(psnr_pr_avg - psnr_lq_avg):.6f}",
                    f"{ssim_lq_avg:.6f}",
                    f"{ssim_pr_avg:.6f}",
                    f"{(ssim_pr_avg - ssim_lq_avg) * 100.0:.6f}",
                ])

        print(f"[Info] saved csv to: {args.save_csv}")


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python test_ors.py \
#   --ckpt /tdx/rgm/ORS_CVE_main/exp/qp37_RA/ckp_ra_qp37.pt \
#   --lq_root /tdx/rgm/data/VCP_dataset/test_18_data/LD/qp37 \
#   --gt_root /tdx/rgm/data/VCP_dataset/test_18_data/GT \
#   --batch_size 1 \
#   --num_workers 8 \
#   --save_y \ #
#   --save_y_dir /tdx/rgm/对比实验/ORS_CVE