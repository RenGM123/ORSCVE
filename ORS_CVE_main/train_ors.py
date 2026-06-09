import os
import math
import time
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
from flow_vis import flow_to_image
from collections import defaultdict

import utils

from dataset.ors_ldb_dataset import (
     TrainORSLDBDataset as TrainSet,
     ValORSLDBDataset as ValSet,
 )

#from dataset.ors_ra_dataset import (
#    TrainORSRADataset as TrainSet,
 #   ValORSRADataset as ValSet,
#)

from models.ors_cve import ORS_CVE


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_uint8_img(x: torch.Tensor) -> np.ndarray:
    """
    将 (B,1,H,W), (1,H,W), (H,W), 或多通道特征图转为 uint8 灰度图。
    多通道特征图默认保存第一个通道，主要用于预览。
    """
    x = x.detach().float().cpu()

    if x.ndim == 4:
        x = x[0, 0]
    elif x.ndim == 3:
        x = x[0]

    xmin, xmax = float(x.min()), float(x.max())

    if not (xmin >= 0.0 and xmax <= 1.0):
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        else:
            x = x * 0.0

    x = x.clamp(0.0, 1.0).numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_gray_image(t: torch.Tensor, save_path: str):
    img = _to_uint8_img(t)
    Image.fromarray(img, mode="L").save(save_path)


def receive_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt_path",
        type=str,
        default="config.yml",
        help="Path to option YAML file."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Distributed launcher requires."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt) to resume from."
    )

    args = parser.parse_args()

    with open(args.opt_path, "r") as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict["opt_path"] = args.opt_path
    opts_dict["train"]["rank"] = args.local_rank
    opts_dict["train"]["resume"] = args.resume or opts_dict["train"].get("resume", "")

    if opts_dict["train"]["exp_name"] is None:
        opts_dict["train"]["exp_name"] = utils.get_timestr()

    opts_dict["train"]["log_path"] = op.join(
        "exp",
        opts_dict["train"]["exp_name"],
        "log.log"
    )
    opts_dict["train"]["checkpoint_save_path_pre"] = op.join(
        "exp",
        opts_dict["train"]["exp_name"],
        "ckp_"
    )

    opts_dict["train"]["num_gpu"] = torch.cuda.device_count()
    opts_dict["train"]["is_dist"] = opts_dict["train"]["num_gpu"] > 1

    opts_dict["test"]["restore_iter"] = int(opts_dict["test"]["restore_iter"])

    td = opts_dict["train"]
    td.setdefault("lambda_rec", 1.0)


    td.setdefault("lambda_msa", 0.02)

    return opts_dict


def _pad_to16(x: torch.Tensor):
    B, C, H, W = x.shape
    ph = (-H) % 16
    pw = (-W) % 16

    if ph == 0 and pw == 0:
        return x, (H, W)

    return torch.nn.functional.pad(
        x,
        (0, pw, 0, ph),
        mode="replicate"
    ), (H, W)


def _unpad(x: torch.Tensor, size_hw):
    H, W = size_hw
    return x[..., :H, :W]


def _strip_module_prefix(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def main_danka():
    opts_dict = receive_arg()

    rank = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    opts_dict["train"]["num_gpu"] = 1
    opts_dict["train"]["is_dist"] = False

    log_dir = op.join("exp", opts_dict["train"]["exp_name"])
    utils.mkdir(log_dir)

    log_fp = open(opts_dict["train"]["log_path"], "w")

    msg = (
        f"{'<' * 10} Hello {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict)}"
    )

    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

    seed = opts_dict["train"]["random_seed"]
    utils.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True

    print("[DBG] will build train_ds", flush=True)

    train_ds = TrainSet(
        lq_root="/tdx/rgm/data/VCP_dataset/LQ_Priors/LD/qp37",
        gt_root="/tdx/rgm/data/VCP_dataset/GT",
        #lq_root="/tdx/rgm/data/VCP_dataset/LQ_Priors/LD/encoder_randomaccess_main_gop8_qp37",
        #gt_root="/tdx/rgm/data/VCP_dataset/GT_MFQE",
        gop=8,
        gt_size=opts_dict["dataset"]["train"].get("gt_size", 256),
        use_flip=opts_dict["dataset"]["train"].get("use_flip", True),
        use_rot=opts_dict["dataset"]["train"].get("use_rot", True),
    )

    print("[DBG] built train_ds; will build val_ds", flush=True)

    val_ds = ValSet(
        lq_root="/tdx/rgm/data/VCP_dataset/test_18_data/LD/qp37",
        gt_root="/tdx/rgm/data/VCP_dataset/test_18_data/GT",
        #lq_root="/tdx/rgm/data/VCP_dataset/test_18_data/LD/RA_qp37_gop8_mfqe",
        #gt_root="/tdx/rgm/data/VCP_dataset/test_18_data/GT_MFQE",
        gop=8,
        gt_size=None,
        use_flip=False,
        use_rot=False,
    )

    train_sampler = utils.DistSampler(
        dataset=train_ds,
        num_replicas=opts_dict["train"]["num_gpu"],
        rank=rank,
        ratio=opts_dict["dataset"]["train"]["enlarge_ratio"]
    )

    print("[DBG] built val_ds; will build train_loader", flush=True)

    train_loader = utils.create_dataloader(
        dataset=train_ds,
        opts_dict=opts_dict,
        sampler=train_sampler,
        phase="train",
        seed=opts_dict["train"]["random_seed"]
    )

    print("[DBG] built train_loader; will build val_loader", flush=True)

    val_loader = utils.create_dataloader(
        dataset=val_ds,
        opts_dict=opts_dict,
        sampler=None,
        phase="val"
    )

    assert train_loader is not None

    batch_size = opts_dict["dataset"]["train"]["batch_size_per_gpu"]
    num_iter = int(opts_dict["train"]["num_iter"])

    num_iter_per_epoch = math.ceil(
        len(train_ds)
        * opts_dict["dataset"]["train"]["enlarge_ratio"]
        / batch_size
    )

    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)

    print("[DBG] built loaders; will create prefetchers", flush=True)

    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    print("[DBG] prefetchers ready; will build model", flush=True)

    model = ORS_CVE().to(device)

    print("[DBG] model to(device) done", flush=True)

    assert opts_dict["train"]["loss"].pop("type") == "CharbonnierLoss", \
        "Only CharbonnierLoss is wired here."

    assert opts_dict["train"]["optim"].pop("type") == "Adam", \
        "Only Adam is wired here."
    optimizer = optim.Adam(model.parameters(), **opts_dict["train"]["optim"])

    def set_lr(optimizer_, lr):
        for param_group in optimizer_.param_groups:
            param_group["lr"] = lr

    scheduler = None

    if opts_dict["train"]["scheduler"]["is_on"]:
        assert opts_dict["train"]["scheduler"].pop("type") == "CosineAnnealingRestartLR", \
            "Only CosineAnnealingRestartLR."

        sc = opts_dict["train"]["scheduler"]

        periods = sc.get("periods", sc.get("T_periods", num_iter))

        if isinstance(periods, int):
            periods = [periods]
        elif not isinstance(periods, (list, tuple)):
            periods = [num_iter]

        periods = list(map(int, periods))

        total_T = sum(periods)
        if total_T < num_iter:
            periods[-1] += num_iter - total_T

        restart_weights = sc.get("restart_weights", [1.0] * len(periods))

        if (
            not isinstance(restart_weights, (list, tuple))
            or len(restart_weights) != len(periods)
        ):
            restart_weights = [1.0] * len(periods)

        sc["periods"] = periods
        sc["restart_weights"] = list(map(float, restart_weights))
        sc.pop("T_periods", None)

        scheduler = utils.CosineAnnealingRestartLR(optimizer, **sc)
        opts_dict["train"]["scheduler"]["is_on"] = True

    assert opts_dict["train"]["criterion"].pop("type") == "PSNR", \
        "Only PSNR is wired here."
    criterion = utils.PSNR()

    start_iter = 0
    start_epoch = 0
    resume_path = opts_dict["train"].get("resume", "")

    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        state_dict = _strip_module_prefix(ckpt["state_dict"])

        model.load_state_dict(state_dict, strict=True)

        optimizer.load_state_dict(ckpt["optimizer"])

        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device, non_blocking=True)

        new_lr = float(opts_dict["train"]["optim"]["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr

        print(f"[Resume] Override lr to {new_lr}")

        start_iter = int(ckpt.get("num_iter_accum", 0))
        start_epoch = start_iter // num_iter_per_epoch

        if scheduler is not None:
            try:
                scheduler.last_epoch = start_iter
                lrs = scheduler.get_lr()
                for pg, lr in zip(optimizer.param_groups, lrs):
                    pg["lr"] = lr
            except Exception:
                pass

        print(
            f"[Resume] Loaded: {resume_path}, "
            f"continue from iter={start_iter}, epoch={start_epoch}"
        )

    msg = (
        f"\n{'<' * 10} Dataloader {'>' * 10}\n"
        f"total iters: [{num_iter}]\n"
        f"total epochs: [{num_epoch}]\n"
        f"iter per epoch: [{num_iter_per_epoch}]\n"
        f"val samples: [{val_num}]\n"
        f"start from iter: [{start_iter}]\n"
        f"start from epoch: [{start_epoch}]"
    )

    print(msg)
    log_fp.write(msg + "\n")
    log_fp.flush()

    if opts_dict["train"]["pre-val"]:
        msg = f"\n{'<' * 10} Pre-evaluation {'>' * 10}"
        print(msg)
        log_fp.write(msg + "\n")

        pbar = tqdm(total=val_num, ncols=opts_dict["train"]["pbar_len"])

        val_prefetcher.reset()
        val_data = val_prefetcher.next()
        psnr_list = []

        while val_data is not None:
            LQ = val_data["lq"].to(device)
            GT = val_data["gt"].to(device)

            b = LQ.size(0)
            batch_perf = np.mean([
                criterion(LQ[i], GT[i])
                for i in range(b)
            ])

            psnr_list.append(batch_perf)

            pbar.set_description(
                "Pre-PSNR: {:.3f} {}".format(
                    batch_perf,
                    opts_dict["train"]["criterion"]["unit"]
                )
            )

            pbar.update(b)
            val_data = val_prefetcher.next()

        pbar.close()

        ave_p = float(np.mean(psnr_list)) if len(psnr_list) > 0 else 0.0

        msg = "> ori performance: [{:.3f}] {}".format(
            ave_p,
            opts_dict["train"]["criterion"]["unit"]
        )

        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

    msg = f"\n{'<' * 10} Training {'>' * 10}"
    print(msg)
    log_fp.write(msg + "\n")

    total_timer = utils.Timer()

    model.train()

    num_iter_accum = start_iter
    unit = opts_dict["train"]["criterion"]["unit"]

    for current_epoch in range(start_epoch, num_epoch + 1):
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        while train_data is not None:
            num_iter_accum += 1

            if num_iter_accum > num_iter:
                break

            LQ = train_data["lq"].to(device)
            GT = train_data["gt"].to(device)
            ORS_LQ = train_data["ors_lq"].to(device)
            ORS_RES = train_data["ors_res"].to(device)
            LQMV = train_data["lq_mv"].to(device)

            LQp, _ = _pad_to16(LQ)
            GTp, _ = _pad_to16(GT)
            ORS_LQp, _ = _pad_to16(ORS_LQ)
            ORS_RESp, _ = _pad_to16(ORS_RES)
            LQMVp, _ = _pad_to16(LQMV)

            enhanced_p, loss_rec, loss_flow = model(
                LQ=LQp,
                LQMV=LQMVp,
                ORS_LQ=ORS_LQp,
                ORS_RES=ORS_RESp,
                GT=GTp,
                test_mode=False,
            )

            lam_rec = float(opts_dict["train"]["lambda_rec"])
            lam_flow = float(opts_dict["train"]["lambda_msa"])

            loss = lam_rec * loss_rec + lam_flow * loss_flow

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if scheduler is not None and opts_dict["train"]["scheduler"]["is_on"]:
                scheduler.step()
            else:
                if num_iter_accum == 300000:
                    set_lr(optimizer, 5e-5)
                    print(f"[LR Scheduler] Iter {num_iter_accum}: set lr to 5e-5")
                elif num_iter_accum == 560000:
                    set_lr(optimizer, 2.5e-5)
                    print(f"[LR Scheduler] Iter {num_iter_accum}: set lr to 2.5e-5")
                elif num_iter_accum == 600000:
                    set_lr(optimizer, 0.125e-5)
                    print(f"[LR Scheduler] Iter {num_iter_accum}: set lr to 0.125e-5")

            if num_iter_accum % int(opts_dict["train"]["interval_print"]) == 0:
                lr = optimizer.param_groups[0]["lr"]

                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    f"lr: [{lr / 1e-4:.3f}]x1e-4, "
                    f"loss: {loss.item():.6f}, "
                    f"rec_loss: {loss_rec.item():.6f}, "
                    f"flow_loss: {loss_flow.item():.6f}"
                )

                print(msg)
                log_fp.write(msg + "\n")
                log_fp.flush()

            if (
                num_iter_accum % int(opts_dict["train"]["interval_val"]) == 0
                or num_iter_accum == num_iter
            ):
                ckpt_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}.pt"
                )

                state = {
                    "num_iter_accum": num_iter_accum,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                if scheduler is not None and opts_dict["train"]["scheduler"]["is_on"]:
                    state["scheduler"] = scheduler.state_dict()

                torch.save(state, ckpt_path)

                with torch.no_grad():
                    pbar = tqdm(
                        total=val_num,
                        ncols=opts_dict["train"]["pbar_len"]
                    )

                    model.eval()
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()

                    acc = defaultdict(lambda: {"n": 0, "psnr_sum": 0.0})

                    preview_dir = os.path.join(
                        "exp",
                        opts_dict["train"]["exp_name"],
                        "preview",
                        f"iter_{num_iter_accum}"
                    )

                    try:
                        total_batches = len(val_loader)
                    except Exception:
                        total_batches = max(1, val_num)

                    save_batch_idx = random.randint(0, max(0, total_batches - 1))
                    cur_batch_idx = 0
                    saved_preview = False

                    all_n = 0
                    all_psnr_sum = 0.0

                    while val_data is not None:
                        LQv = val_data["lq"].to(device)
                        GTv = val_data["gt"].to(device)
                        ORS_LQv = val_data["ors_lq"].to(device)
                        ORS_RESv = val_data["ors_res"].to(device)
                        LQMVv = val_data["lq_mv"].to(device)

                        video_names = val_data.get("video_name", None)

                        if video_names is None:
                            raise KeyError(
                                "val_data has no key 'video_name'. "
                                "Please check ValORSLDBDataset output dict."
                            )

                        LQvp, _ = _pad_to16(LQv)
                        GTvp, szv = _pad_to16(GTv)
                        ORS_LQvp, _ = _pad_to16(ORS_LQv)
                        ORS_RESvp, _ = _pad_to16(ORS_RESv)
                        LQMVvp, _ = _pad_to16(LQMVv)

                        enhanced_p, intermediates_p = model(
                            LQ=LQvp,
                            LQMV=LQMVvp,
                            ORS_LQ=ORS_LQvp,
                            ORS_RES=ORS_RESvp,
                            GT=GTvp,
                            test_mode=True,
                            return_intermediates=True
                        )

                        enhanced_v = _unpad(enhanced_p, szv).clamp(0.0, 1.0)

                        intermediates_v = {
                            k: _unpad(v, szv)
                            for k, v in intermediates_p.items()
                            if isinstance(v, torch.Tensor)
                        }

                        b = LQv.size(0)

                        for i in range(b):
                            vn = video_names[i]
                            ps = float(utils.PSNR()(enhanced_v[i], GTv[i]))

                            acc[vn]["n"] += 1
                            acc[vn]["psnr_sum"] += ps

                            all_n += 1
                            all_psnr_sum += ps

                        if all_n > 0:
                            running_psnr = all_psnr_sum / all_n
                            pbar.set_description(
                                "Val-PSNR(frame-weighted): {:.3f} {}".format(
                                    running_psnr,
                                    unit
                                )
                            )

                        pbar.update(b)

                        if (not saved_preview) and (cur_batch_idx == save_batch_idx):
                            _ensure_dir(preview_dir)

                            _save_gray_image(
                                LQv,
                                os.path.join(preview_dir, "LQ.png")
                            )
                            _save_gray_image(
                                ORS_LQv,
                                os.path.join(preview_dir, "ORS_LQ.png")
                            )
                            _save_gray_image(
                                ORS_RESv,
                                os.path.join(preview_dir, "ORS_RES.png")
                            )
                            _save_gray_image(
                                enhanced_v,
                                os.path.join(preview_dir, "enhanced.png")
                            )
                            _save_gray_image(
                                GTv,
                                os.path.join(preview_dir, "GT.png")
                            )

                            if "ORS_warp" in intermediates_v:
                                _save_gray_image(
                                    intermediates_v["ORS_warp"],
                                    os.path.join(preview_dir, "ORS_warp.png")
                                )

                            if "ORS_RES_warp" in intermediates_v:
                                _save_gray_image(
                                    intermediates_v["ORS_RES_warp"],
                                    os.path.join(preview_dir, "ORS_RES_warp.png")
                                )

                            if "F_Fus" in intermediates_v:
                                _save_gray_image(
                                    intermediates_v["F_Fus"],
                                    os.path.join(preview_dir, "F_Fus_ch0.png")
                                )

                            flow_np = LQMVv[0].detach().cpu().numpy()

                            if flow_np.ndim == 3 and flow_np.shape[0] == 2:
                                flow_np = np.transpose(flow_np, (1, 2, 0))

                            flow_img = flow_to_image(
                                flow_np,
                                clip_flow=None,
                                convert_to_bgr=False
                            )

                            Image.fromarray(flow_img).save(
                                os.path.join(preview_dir, "LQMV.png")
                            )

                            if "flow" in intermediates_v:
                                refined_flow = intermediates_v["flow"][0].detach().cpu().numpy()

                                if refined_flow.ndim == 3 and refined_flow.shape[0] == 2:
                                    refined_flow = np.transpose(refined_flow, (1, 2, 0))

                                refined_flow_img = flow_to_image(
                                    refined_flow,
                                    clip_flow=None,
                                    convert_to_bgr=False
                                )

                                Image.fromarray(refined_flow_img).save(
                                    os.path.join(preview_dir, "refined_flow.png")
                                )

                            saved_preview = True

                        cur_batch_idx += 1
                        val_data = val_prefetcher.next()

                    pbar.close()
                    model.train()

                    per_vid = []

                    for vn in sorted(acc.keys()):
                        n_ = acc[vn]["n"]
                        psnr_avg = acc[vn]["psnr_sum"] / max(1, n_)
                        per_vid.append(psnr_avg)

                        print(
                            f"[Val] {vn}: PSNR={psnr_avg:.3f} {unit} "
                            f"(n={n_})"
                        )

                    ave_per = float(np.mean(per_vid)) if len(per_vid) > 0 else 0.0

                    msg = (
                        "> model saved at {}\n"
                        "> ave val per(video-wise): [{:.3f}] {}"
                    ).format(ckpt_path, ave_per, unit)

                    print(msg)

                    ave_log_path = os.path.join(
                        "exp",
                        opts_dict["train"]["exp_name"],
                        "val_metric_log.txt"
                    )

                    with open(ave_log_path, "a") as f:
                        f.write(f"{num_iter_accum},{ave_per:.4f}\n")

                    log_fp.write(msg + "\n")
                    log_fp.flush()

            train_data = tra_prefetcher.next()

    total_time = total_timer.get_interval() / 3600

    msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
    print(msg)
    log_fp.write(msg + "\n")

    msg = f"\n{'<' * 10} Goodbye {'>' * 10}\nTimestamp: [{utils.get_timestr()}]"
    print(msg)
    log_fp.write(msg + "\n")
    log_fp.close()


if __name__ == "__main__":
    main_danka()

#CUDA_VISIBLE_DEVICES=0 python train_ors.py --opt_path config.yml