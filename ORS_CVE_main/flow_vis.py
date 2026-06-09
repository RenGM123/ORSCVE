import os
import glob
import numpy as np
from PIL import Image

# ===== 你的颜色编码工具 =====
def make_colorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    和原来的功能一样，只是更鲁棒：
    - 输入 u, v 里如果还有 NaN/Inf，直接当 0 处理
    """
    # 先把 NaN / Inf 干掉
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    rad = np.sqrt(u**2 + v**2)
    # 再保险一下
    rad = np.nan_to_num(rad, nan=0.0, posinf=0.0, neginf=0.0)

    a = np.arctan2(-v, -u) / np.pi  # [-1,1]
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    fk = (a + 1) / 2 * (ncols - 1)  # [0, ncols-1]
    fk = np.nan_to_num(fk, nan=0.0, posinf=0.0, neginf=0.0)

    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k0 = np.clip(k0, 0, ncols - 1)
    k1 = np.clip(k1, 0, ncols - 1)

    f = fk - k0
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75

        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * np.clip(col, 0, 1))

    return flow_image

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    强制模式 + NaN 清理版
    - 支持任意有一个维度为2的三维数组：这维视为通道
    - 把 NaN/Inf 先变成 0，再做归一化和颜色编码
    """
    if flow_uv.ndim != 3:
        raise ValueError(f"input flow must have 3 dims, got {flow_uv.shape}")

    shape = flow_uv.shape
    axes_with_2 = [i for i, s in enumerate(shape) if s == 2]
    if len(axes_with_2) != 1:
        raise ValueError(f"cannot determine channel dim (size=2) from shape {shape}")

    ch_axis = axes_with_2[0]
    if ch_axis != 2:
        flow_uv = np.moveaxis(flow_uv, ch_axis, 2)  # -> (H, W, 2)

    # 现在一定是 (H, W, 2)
    assert flow_uv.shape[2] == 2

    # 把 NaN / Inf 统统干成 0
    flow_uv = np.nan_to_num(flow_uv, nan=0.0, posinf=0.0, neginf=0.0)

    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)

    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]

    rad = np.sqrt(u**2 + v**2)
    rad = np.nan_to_num(rad, nan=0.0, posinf=0.0, neginf=0.0)

    rad_max = np.max(rad)
    epsilon = 1e-5
    if rad_max < epsilon:
        # 全零流的情况，直接返回全黑图 / 全某个固定颜色
        return np.zeros((flow_uv.shape[0], flow_uv.shape[1], 3), np.uint8)

    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)

def visualize_flow_folder(npy_dir, out_dir, clip_flow=None):
    os.makedirs(out_dir, exist_ok=True)
    npy_list = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    print(f"Found {len(npy_list)} flow files")

    for idx, npy_path in enumerate(npy_list):
        flow = np.load(npy_path)

        try:
            flow_img = flow_to_image(flow, clip_flow=clip_flow, convert_to_bgr=False)
        except Exception as e:
            print(f"Skip {npy_path}, reason: {e}")
            continue

        out_name = f"{idx:04d}.png"  # 从 0 开始：0000.png, 0001.png, ...
        out_path = os.path.join(out_dir, out_name)
        Image.fromarray(flow_img).save(out_path)

        if idx % 10 == 0:
            print(f"[{idx}/{len(npy_list)}] saved {out_path}")

    print("Done!")

if __name__ == "__main__":
    # 把这两个路径改成你的实际路径
    npy_dir  = "E:\下载\VCP_DATASET\LQ_Priors\LD\qp37\\7monkey4_1920x1080\mv"
    out_dir  = "E:\diff\data\MFQEv2_dataset\运动矢量可视化"

    # clip_flow 可以不写（为 None），如果光流值特别大可以设一个上限，例如 20.0
    visualize_flow_folder(npy_dir, out_dir, clip_flow=None)