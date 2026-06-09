import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .dcn import ModulatedDeformConvPack
from .loss import Charbonnier_L1, Ternary, aignloss


def resize(x, scale_factor):
    return F.interpolate(
        x,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=False
    )

def warp(img, flow):
    B, _, H, W = flow.shape
    xx = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(B, -1, H, -1) #
    yy = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(B, -1, -1, W)
    grid = torch.cat([xx, yy], 1).to(img)
    flow_ = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0), flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)
    grid_ = (grid + flow_).permute(0, 2, 3, 1)
    output = F.grid_sample(input=img, grid=grid_, mode='bilinear', padding_mode='border', align_corners=True)
    return output


def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias),
        nn.PReLU(out_channels)
    )

class SecondOrderDeformableAlignment2(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        super(SecondOrderDeformableAlignment2, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True), # #

            nn.Conv2d(self.out_channels, 27 * self.deformable_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        _constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, extra_feat, flow):
        extra_feat = torch.cat([extra_feat, flow], dim=1)

        offset_mask = self.conv_offset(extra_feat)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)

        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((offset_x, offset_y), dim=1)
        )

        offset = offset + flow.flip(1).repeat(
            1,
            offset.size(1) // 2,
            1,
            1
        )

        mask = torch.sigmoid(mask)

        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask
        )

class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()

        self.side_channels = side_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            ),
            nn.PReLU(in_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            ),
            nn.PReLU(side_channels)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            ),
            nn.PReLU(in_channels)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias
            ),
            nn.PReLU(side_channels)
        )

        self.conv5 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias
        )

        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        out[:, -self.side_channels:, :, :] = self.conv2(
            out[:, -self.side_channels:, :, :].clone()
        )

        out = self.conv3(out)

        out[:, -self.side_channels:, :, :] = self.conv4(
            out[:, -self.side_channels:, :, :].clone()
        )

        out = self.prelu(x + self.conv5(out))

        return out

class SharedEncoder(nn.Module):
    """
    Shared-weight multi-scale encoder for the current LQ frame
    and the ORS reference frame.
    """
    def __init__(self):
        super(SharedEncoder, self).__init__()

        self.scale1 = nn.Sequential(
            convrelu(1, 32, 3, 2, 1),
            convrelu(32, 32, 3, 1, 1)
        )

        self.scale2 = nn.Sequential(
            convrelu(32, 48, 3, 2, 1),
            convrelu(48, 48, 3, 1, 1)
        )

        self.scale3 = nn.Sequential(
            convrelu(48, 72, 3, 2, 1),
            convrelu(72, 72, 3, 1, 1)
        )

        self.scale4 = nn.Sequential(
            convrelu(72, 96, 3, 2, 1),
            convrelu(96, 96, 3, 1, 1)
        )

    def forward(self, frame):
        F_s1 = self.scale1(frame)
        F_s2 = self.scale2(F_s1)
        F_s3 = self.scale3(F_s2)
        F_s4 = self.scale4(F_s3)

        return {
            's1': F_s1,
            's2': F_s2,
            's3': F_s3,
            's4': F_s4,
        }

class MVG_CS_OFP4(nn.Module):
    """
    Coarsest MVG-CS-OFP layer.
    Initializes coarse optical flow and propagated fused feature.
    """
    def __init__(self):
        super(MVG_CS_OFP4, self).__init__()

        self.refine = nn.Sequential(
            convrelu(192, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 74, 4, 2, 1, bias=True)
        )

    def forward(self, F_Re, F_LQ):
        fusion_input = torch.cat([F_Re, F_LQ], dim=1)
        output = self.refine(fusion_input)

        flow_s3 = output[:, 0:2]
        F_Fus_s3 = output[:, 2:]

        return flow_s3, F_Fus_s3

class MVG_CS_OFP3(nn.Module):
    """
    MVG-CS-OFP layer from scale 3 to scale 2.
    """
    def __init__(self):
        super(MVG_CS_OFP3, self).__init__()

        self.align = SecondOrderDeformableAlignment2(
            in_channels=72,
            out_channels=72,
            kernel_size=3,
            stride=1,
            padding=1,
            deformable_groups=2
        )

        self.motion_gate = nn.Sequential(
            convrelu(4, 16, 3, 1, 1),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            convrelu(220, 216),
            ResBlock(216, 32),
            nn.ConvTranspose2d(216, 50, 4, 2, 1, bias=True)
        )

    def forward(self, F_Fus, F_Re, F_LQ, flow, MV):
        align_condition = torch.cat([F_Fus, F_Re, F_LQ], dim=1)
        F_Re_to_LQ = self.align(F_Re, align_condition, flow)

        gate = self.motion_gate(torch.cat([flow, MV], dim=1))
        F_Re_to_LQ = gate * F_Re_to_LQ + (1.0 - gate) * F_LQ

        refine_input = torch.cat(
            [F_Fus, F_LQ, F_Re_to_LQ, flow, MV],
            dim=1
        )

        output = self.refine(refine_input)

        flow_residual = output[:, 0:2]
        F_Fus_next = output[:, 2:]

        return flow_residual, F_Fus_next

class MVG_CS_OFP2(nn.Module):
    """
    MVG-CS-OFP layer from scale 2 to scale 1.
    """
    def __init__(self):
        super(MVG_CS_OFP2, self).__init__()

        self.align = SecondOrderDeformableAlignment2(
            in_channels=48,
            out_channels=48,
            kernel_size=3,
            stride=1,
            padding=1,
            deformable_groups=2
        )

        self.motion_gate = nn.Sequential(
            convrelu(4, 16, 3, 1, 1),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            convrelu(148, 144),
            ResBlock(144, 32),
            nn.ConvTranspose2d(144, 34, 4, 2, 1, bias=True)
        )

    def forward(self, F_Fus, F_Re, F_LQ, flow, MV):
        align_condition = torch.cat([F_Fus, F_Re, F_LQ], dim=1)
        F_Re_to_LQ = self.align(F_Re, align_condition, flow)

        gate = self.motion_gate(torch.cat([flow, MV], dim=1))
        F_Re_to_LQ = gate * F_Re_to_LQ + (1.0 - gate) * F_LQ

        refine_input = torch.cat(
            [F_Fus, F_LQ, F_Re_to_LQ, flow, MV],
            dim=1
        )

        output = self.refine(refine_input)

        flow_residual = output[:, 0:2]
        F_Fus_next = output[:, 2:]

        return flow_residual, F_Fus_next

class MVG_CS_OFP1(nn.Module):
    """
    Finest MVG-CS-OFP layer.
    Produces full-resolution optical flow and propagated fused feature.
    """
    def __init__(self):
        super(MVG_CS_OFP1, self).__init__()

        self.align = SecondOrderDeformableAlignment2(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            deformable_groups=2
        )

        self.motion_gate = nn.Sequential(
            convrelu(4, 16, 3, 1, 1),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            convrelu(100, 96),
            ResBlock(96, 32),
            nn.ConvTranspose2d(96, 34, 4, 2, 1, bias=True)
        )

    def forward(self, F_Fus, F_Re, F_LQ, flow, MV):
        align_condition = torch.cat([F_Fus, F_Re, F_LQ], dim=1)
        F_Re_to_LQ = self.align(F_Re, align_condition, flow)

        gate = self.motion_gate(torch.cat([flow, MV], dim=1))
        F_Re_to_LQ = gate * F_Re_to_LQ + (1.0 - gate) * F_LQ

        refine_input = torch.cat(
            [F_Fus, F_LQ, F_Re_to_LQ, flow, MV],
            dim=1
        )

        output = self.refine(refine_input)

        flow_residual = output[:, 0:2]
        F_Fus_full = output[:, 2:]

        return flow_residual, F_Fus_full

class MV_HFR(nn.Module):
    """
    MV-HFR: Motion-Vector-Guided Hierarchical Optical Flow Refinement.

    Implemented by stacked MVG-CS-OFP layers.
    """
    def __init__(self):
        super(MV_HFR, self).__init__()

        self.stage4 = MVG_CS_OFP4()
        self.stage3 = MVG_CS_OFP3()
        self.stage2 = MVG_CS_OFP2()
        self.stage1 = MVG_CS_OFP1()

    def forward(self, F_Re, F_LQ, MV):
        MV_s1 = resize(MV, scale_factor=0.5)
        MV_s2 = resize(MV, scale_factor=0.25)
        MV_s3 = resize(MV, scale_factor=0.125)

        flow_s3, F_Fus_s3 = self.stage4(
            F_Re=F_Re['s4'],
            F_LQ=F_LQ['s4']
        )

        flow_s2_residual, F_Fus_s2 = self.stage3(
            F_Fus=F_Fus_s3,
            F_Re=F_Re['s3'],
            F_LQ=F_LQ['s3'],
            flow=flow_s3,
            MV=MV_s3
        )
        flow_s2 = flow_s2_residual + 2.0 * resize(flow_s3, scale_factor=2.0)

        flow_s1_residual, F_Fus_s1 = self.stage2(
            F_Fus=F_Fus_s2,
            F_Re=F_Re['s2'],
            F_LQ=F_LQ['s2'],
            flow=flow_s2,
            MV=MV_s2
        )
        flow_s1 = flow_s1_residual + 2.0 * resize(flow_s2, scale_factor=2.0)

        flow_full_residual, F_Fus_full = self.stage1(
            F_Fus=F_Fus_s1,
            F_Re=F_Re['s1'],
            F_LQ=F_LQ['s1'],
            flow=flow_s1,
            MV=MV_s1
        )
        flow_full = flow_full_residual + 2.0 * resize(flow_s1, scale_factor=2.0)

        flow_pyramid = {
            'full': flow_full,
            's1': flow_s1,
            's2': flow_s2,
            's3': flow_s3,
        }

        return F_Fus_full, flow_pyramid

class LQ_RGQE(nn.Module):
    """
    LQ-RGQE: Low-QP Residual-Guided Quality Enhancement.
    """
    def __init__(self, base_ch: int = 64, fus_ch: int = 32):
        super(LQ_RGQE, self).__init__()

        total_in_ch = 1 + 1 + 1 + fus_ch

        self.conv_in = nn.Sequential(
            nn.Conv2d(total_in_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.ca_fc1 = nn.Linear(base_ch * 4, base_ch)
        self.ca_fc2 = nn.Linear(base_ch, base_ch * 4)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.sa = nn.Sequential(
            nn.Conv2d(base_ch + 1, base_ch // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, F_Fus, LQ, ORS_warp, ORS_RES_warp):
        neutral_residual = 128.0 / 255.0
        degradation = torch.abs(ORS_RES_warp - neutral_residual)

        reconstruction_input = torch.cat(
            [LQ, ORS_warp, degradation, F_Fus],
            dim=1
        )

        F_0 = self.conv_in(reconstruction_input)
        F_1 = self.down1(F_0)
        F_bot = self.down2(F_1)

        b, c, _, _ = F_bot.shape

        channel_weight = F.adaptive_avg_pool2d(F_bot, 1).view(b, c)
        channel_weight = F.relu(self.ca_fc1(channel_weight), inplace=True)
        channel_weight = torch.sigmoid(self.ca_fc2(channel_weight)).view(
            b,
            c,
            1,
            1
        )

        F_bot = F_bot * channel_weight

        F_rec = self.up1(F_bot) + F_1
        F_rec = self.up2(F_rec) + F_0

        spatial_weight = self.sa(torch.cat([F_rec, degradation], dim=1))
        F_rec = F_rec * spatial_weight

        enhanced = self.conv_out(F_rec)

        return enhanced

class ORS_CVE(nn.Module):
    """
    ORS-CVE: Compressed Video Quality Enhancement Based on
    Optimal Reference Selection.
    """
    def __init__(self, local_rank=-1, lr=1e-4):
        super(ORS_CVE, self).__init__()

        self.encoder = SharedEncoder()
        self.mv_hfr = MV_HFR()
        self.lq_rgqe = LQ_RGQE(base_ch=64, fus_ch=32)

        self.l1_loss = Charbonnier_L1()
        self.tr_loss = Ternary(7)

    def forward(
        self,
        LQ,
        LQMV,
        ORS_LQ,
        ORS_RES,
        GT=None,
        test_mode=False,
        return_intermediates=False
    ):
        F_LQ = self.encoder(LQ)
        F_Re = self.encoder(ORS_LQ)

        F_Fus, flow_pyramid = self.mv_hfr(
            F_Re=F_Re,
            F_LQ=F_LQ,
            MV=LQMV
        )

        flow_full = flow_pyramid['full']
        flow_s1 = flow_pyramid['s1']
        flow_s2 = flow_pyramid['s2']
        flow_s3 = flow_pyramid['s3']

        ORS_warp = warp(ORS_LQ, flow_full)
        ORS_RES_warp = warp(ORS_RES, flow_full)

        high_thr = 128.5 / 255.0
        low_thr = 127.5 / 255.0

        degradation_mask = (
            (ORS_RES_warp > high_thr) |
            (ORS_RES_warp < low_thr)
        ).float()

        enhanced = self.lq_rgqe(
            F_Fus=F_Fus,
            LQ=LQ,
            ORS_warp=ORS_warp,
            ORS_RES_warp=ORS_RES_warp
        )

        enhanced = torch.clamp(enhanced, 0.0, 1.0)

        if not test_mode:
            if GT is None:
                raise ValueError('GT must be provided when test_mode=False.')

            alpha, beta = 2.0, 1.0
            pixel_weight = alpha * degradation_mask + beta * (1.0 - degradation_mask)

            loss_mask_rec = self.l1_loss(enhanced - GT, pixel_weight)
            loss_rec = loss_mask_rec + self.tr_loss(enhanced, GT)

            loss_flow = (
                aignloss(LQ, ORS_LQ, GT, flow_full)
                + aignloss(LQ, ORS_LQ, GT, 2.0 * resize(flow_s1, 2.0))
                + aignloss(LQ, ORS_LQ, GT, 4.0 * resize(flow_s2, 4.0))
                + aignloss(LQ, ORS_LQ, GT, 8.0 * resize(flow_s3, 8.0))
            )

            return enhanced, loss_rec, loss_flow

        if return_intermediates:
            intermediates = {
                'enhanced': enhanced,
                'ORS_warp': ORS_warp,
                'ORS_RES_warp': ORS_RES_warp,
                'flow': flow_full,
                'F_Fus': F_Fus,
            }

            return enhanced, intermediates

        return enhanced

if __name__ == "__main__":
    import time

    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError(
            "Please install thop first: pip install thop"
        ) from exc

    class ORSCVEWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ORS_CVE()

        def forward(self, LQ, LQMV, ORS_LQ, ORS_RES, GT):
            return self.model(
                LQ=LQ,
                LQMV=LQMV,
                ORS_LQ=ORS_LQ,
                ORS_RES=ORS_RES,
                GT=GT,
                test_mode=True,
                return_intermediates=False
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ORSCVEWrapper().to(device)
    model.eval()

    B, H, W = 1, 720, 1280

    LQ = torch.randn(B, 1, H, W, device=device)
    LQMV = torch.randn(B, 2, H, W, device=device)
    ORS_LQ = torch.randn(B, 1, H, W, device=device)
    ORS_RES = torch.randn(B, 1, H, W, device=device)
    GT = torch.randn(B, 1, H, W, device=device)

    with torch.no_grad():
        macs, params = profile(
            model,
            inputs=(LQ, LQMV, ORS_LQ, ORS_RES, GT)
        )

    print("\n========== FLOPs / Params for ORS_CVE ==========")
    print(f"Params          : {params / 1e6:.3f} M")
    print(f"GFLOs            : {macs / 1e9:.3f} GMac")

    print("=================================================\n")

    LQ_t = torch.randn(B, 1, H, W, device=device)
    LQMV_t = torch.randn(B, 2, H, W, device=device)
    ORS_LQ_t = torch.randn(B, 1, H, W, device=device)
    ORS_RES_t = torch.randn(B, 1, H, W, device=device)
    GT_t = torch.randn(B, 1, H, W, device=device)

    warmup_iters = 10
    test_iters = 50

    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(LQ_t, LQMV_t, ORS_LQ_t, ORS_RES_t, GT_t)

        if device.type == "cuda":
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            for _ in range(test_iters):
                _ = model(LQ_t, LQMV_t, ORS_LQ_t, ORS_RES_t, GT_t)

            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            avg_ms_per_forward = elapsed_ms / test_iters

        else:
            t0 = time.perf_counter()

            for _ in range(test_iters):
                _ = model(LQ_t, LQMV_t, ORS_LQ_t, ORS_RES_t, GT_t)

            t1 = time.perf_counter()
            avg_ms_per_forward = (t1 - t0) * 1000.0 / test_iters

        fps = 1000.0 / avg_ms_per_forward

    print("========== Runtime @ 720p (1280×720) ==========")
    print(f"Avg forward time : {avg_ms_per_forward:.3f} ms")
    print(f"FPS              : {fps:.2f}")
    print("================================================\n")


#CUDA_VISIBLE_DEVICES=7 python -m models.ors_cve