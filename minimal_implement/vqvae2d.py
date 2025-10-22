import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Channel-wise LayerNorm for CNN
# -------------------------------
class ChannelLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, H, W)
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        return x


# -------------------------------
# Vector Quantizer (ST Estimator)
# -------------------------------
class VectorQuantizer2D(nn.Module):
    def __init__(self, n_codes=512, d=8, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, d)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.n_codes = n_codes
        self.d = d
        self.beta = beta

    def forward(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        cb = self.codebook.weight                      # (K, D)

        # squared L2 distances
        z2 = (z_flat ** 2).sum(dim=1, keepdim=True)
        e2 = (cb ** 2).sum(dim=1).unsqueeze(0)
        ze = z_flat @ cb.t()
        dists = z2 + e2 - 2 * ze

        idx = torch.argmin(dists, dim=1)
        z_q = self.codebook(idx).view(B, H, W, D).permute(0, 3, 1, 2)

        commit = F.mse_loss(z_q.detach(), z)
        codebk = F.mse_loss(z_q, z.detach())
        vq_loss = codebk + self.beta * commit

        z_q_st = z + (z_q - z).detach()
        return z_q_st, vq_loss, idx.view(B, H, W)


# -------------------------------
# Conv / Deconv with LayerNorm
# -------------------------------
def conv2(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        ChannelLayerNorm(out_ch),
        nn.SiLU()
    )

def deconv2(in_ch, out_ch, k=4, s=2, p=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        ChannelLayerNorm(out_ch),
        nn.SiLU()
    )


# -------------------------------
# Encoder Heads
# -------------------------------
class DepthHeadEnc2D(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            conv2(1, base),
            conv2(base, base)
        )
    def forward(self, x):
        return self.stem(x)


class SemHeadEnc2D(nn.Module):
    def __init__(self, num_classes=20, emb_dim=16, base=32):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.conv = nn.Sequential(
            conv2(emb_dim, base),
            conv2(base, base)
        )
    def forward(self, x):
        B,H,W = x.shape
        x = self.emb(x.view(B*H*W)).view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        return self.conv(x)


# -------------------------------
# Decoder Heads
# -------------------------------
class DepthHeadDec2D(nn.Module):
    def __init__(self, in_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            conv2(in_ch, in_ch),
            conv2(in_ch, in_ch),
            nn.Conv2d(in_ch, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


class SemHeadDec2D(nn.Module):
    def __init__(self, in_ch=32, num_classes=20):
        super().__init__()
        self.net = nn.Sequential(
            conv2(in_ch, in_ch),
            conv2(in_ch, in_ch),
            nn.Conv2d(in_ch, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------
# VQ-VAE 2D
# -------------------------------
class VQVAE2D(nn.Module):
    """
    2D VQ-VAE for single frames
    Downsample: H / 4, W / 8
    """
    def __init__(self,
                 num_classes=20,
                 codebook_size=512,
                 beta=0.25,
                 base=32,
                 latent_dim=2):
        super().__init__()
        D = latent_dim
        self.num_classes = num_classes

        # ---------------- Encoders ----------------
        self.enc_depth = DepthHeadEnc2D(base=base)
        self.enc_sem   = SemHeadEnc2D(num_classes=num_classes, emb_dim=32, base=base)

        # ↓↓↓ Changed strides here ↓↓↓
        self.enc = nn.Sequential(
            conv2(2*base, 32, s=(2,2)),  # 64x1024 -> 32x512  (H/2, W/2)
            conv2(32, 64, s=(2,2)),      # 32x512 -> 16x256  (H/4, W/4)
            conv2(64, 96, s=(1,2)),      # 16x256 -> 16x128  (H/4, W/8)
            conv2(96, D)
        )

        self.vq = VectorQuantizer2D(n_codes=codebook_size, d=D, beta=beta)

        # ---------------- Decoder ----------------
        self.dec = nn.Sequential(
            conv2(D, 96),
            deconv2(96, 64, s=(1,2)),   # W*2 : 16x128 -> 16x256
            conv2(64, 64),
            deconv2(64, 32, s=(2,2)),   # H*2,W*2 : 16x256 -> 32x512
            conv2(32, 32),
            deconv2(32, 32, s=(2,2)),   # 32x512 -> 64x1024
            conv2(32, 32)
        )

        self.dec_depth = DepthHeadDec2D(in_ch=32)
        self.dec_sem   = SemHeadDec2D(in_ch=32, num_classes=num_classes)

    # -------- rest is unchanged --------
    def encode(self, depth, sem):
        d = self.enc_depth(depth)
        s = self.enc_sem(sem)
        x = torch.cat([d, s], dim=1)
        z = self.enc(x)
        z_q, vq_loss, idx = self.vq(z)
        return z, z_q, vq_loss, idx

    def decode(self, z_q):
        feat = self.dec(z_q)
        # crop in case of off-by-one
        feat = feat[:, :, :64, :1024]
        return self.dec_depth(feat), self.dec_sem(feat)

    def forward(self, depth, sem, depth_mask=None):
        if depth_mask is None:
            depth_mask = (depth > 0) & (depth != -1)
        depth_enc = depth.clone()
        depth_enc[~depth_mask] = 0.0

        sem = sem.clamp(min=0, max=self.num_classes-1)

        z, z_q, vq_loss, idx = self.encode(depth_enc, sem)
        depth_hat, sem_logits = self.decode(z_q)

        # L2 for depth
        if depth_mask.any():
            l2 = (depth_hat - depth) ** 2
            depth_l2 = (l2 * depth_mask.float()).sum() / (depth_mask.float().sum() + 1e-6)
        else:
            depth_l2 = (depth_hat ** 2).mean()
        if depth_mask.any():
            l1 = (depth_hat - depth).abs() 
            depth_l1 = (l1 * depth_mask.float()).sum() / (depth_mask.float().sum() + 1e-6) 
        else: 
            depth_l1 = depth_hat.abs().mean()
        sem_loss = F.cross_entropy(
            sem_logits.permute(0,2,3,1).reshape(-1, sem_logits.size(1)),
            sem.reshape(-1),
            ignore_index=0
        )

        loss = 0.1*depth_l2 + depth_l1+ sem_loss + vq_loss
        return {
            "depth_hat": depth_hat,
            "sem_logits": sem_logits,
            "loss": loss,
            "loss_depth": depth_l2,
            "loss_sem": sem_loss,
            "loss_vq": vq_loss,
            "codes": idx
        }