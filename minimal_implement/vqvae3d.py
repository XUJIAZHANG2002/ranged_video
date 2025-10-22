import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# Vector Quantizer (ST estimator)
# -------------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, n_codes=512, d=8, beta=0.25):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, d)
        self.codebook.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.beta = beta
        self.n_codes = n_codes
        self.d = d

    def forward(self, z):
        # z: (B, D, T, H, W)
        B, D, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, D)  # (B*T*H*W, D)

        # distances to codebook
        cb = self.codebook.weight  # (K, D)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e
        z2 = (z_flat**2).sum(dim=1, keepdim=True)               # (N,1)
        e2 = (cb**2).sum(dim=1).unsqueeze(0)                    # (1,K)
        ze = z_flat @ cb.t()                                    # (N,K)
        dists = z2 + e2 - 2 * ze

        # nearest codes
        idx = torch.argmin(dists, dim=1)                        # (N,)
        z_q = self.codebook(idx).view(B, T, H, W, D).permute(0, 4, 1, 2, 3)

        # losses (straight-through)
        # commit: ||z.detach() - z_q||^2 ; codebook: ||z - z_q.detach()||^2
        commit = F.mse_loss(z_q.detach(), z, reduction='mean')
        codebk = F.mse_loss(z_q, z.detach(), reduction='mean')
        vq_loss = codebk + self.beta * commit

        # straight-through estimator
        z_q_st = z + (z_q - z).detach()  # gradients pass to z

        return z_q_st, vq_loss, idx.view(B, T, H, W)


# -------------------------------
# Small 3D building blocks
# -------------------------------
def conv3(in_ch, out_ch, k=3, s=(1,1,1), p=1):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

# def deconv3(in_ch, out_ch, k=4, s=(1,2,2), p=1):
#     # transpose conv to upsample H/W by (2,2), keep T
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
#         nn.GroupNorm(8, out_ch),
#         nn.SiLU()
#     )
def deconv3(in_ch, out_ch, k=4, s=(1,2,2), p=1):
    # transpose conv to upsample H/W by (2,2), keep T
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch,
                           kernel_size=(1,4,4),
                           stride=s,
                           padding=(0,1,1)),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

# -------------------------------
# Heads
# -------------------------------
class DepthHeadEnc(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            conv3(1, base),
            conv3(base, base)
        )
    def forward(self, x):  # x: (B,1,T,H,W)
        return self.stem(x)

class SemHeadEnc(nn.Module):
    def __init__(self, num_classes, emb_dim=16, base=32):
        super().__init__()
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.conv = nn.Sequential(
            conv3(emb_dim, base),
            conv3(base, base)
        )
    def forward(self, x):  # x: (B,T,H,W) int64
        B,T,H,W = x.shape
        x = self.emb(x.view(B*T*H*W)).view(B, T, H, W, -1)  # (..., E)
        x = x.permute(0, 4, 1, 2, 3)  # (B,E,T,H,W)
        return self.conv(x)

class DepthHeadDec(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.out = nn.Conv3d(in_ch, 1, kernel_size=1)
    def forward(self, x):  # -> (B,1,T,H,W)
        return self.out(x)

class SemHeadDec(nn.Module):
    def __init__(self, in_ch=64, num_classes=20):
        super().__init__()
        self.out = nn.Conv3d(in_ch, num_classes, kernel_size=1)
    def forward(self, x):  # -> (B,C,T,H,W)
        return self.out(x)

# -------------------------------
# Minimal VQ-VAE 3D (video spatiotemporal)
# -------------------------------
class VQVAE3D(nn.Module):
    """
    Input:
      depth: (B, 1, T=16, H=64, W=1024)  float
      sem:   (B, T=16, H=64, W=1024)     long
    Latent target spatial: (T=16, H=16, W=128), channels D = 4*C, with C=2 -> D=8
    """
    def __init__(self,
                 num_classes=20,
                 C=2,
                 codebook_size=512,
                 beta=0.25,
                 base=32):
        super().__init__()
        D = 4 * C  # embedding dim in latent (default 8)
        self.num_classes = num_classes

        # modality-specific encoders
        self.enc_depth = DepthHeadEnc(base=base)
        self.enc_sem   = SemHeadEnc(num_classes=num_classes, emb_dim=16, base=base)

        # shared encoder: downsample (64x1024) -> (16x128), keep T
        # Do: (1) stride (1,2,2) 64x1024 -> 32x512
        #     (2) stride (1,2,2) 32x512  -> 16x256
        #     (3) stride (1,1,2) 16x256  -> 16x128
        self.enc = nn.Sequential(
            conv3(2*base, 32, s=(1,2,2)),
            conv3(32, 64, s=(1,2,2)),
            conv3(64, 96, s=(1,1,2)),
            conv3(96, D)
        )
        # vector quantizer over channel dim D
        self.vq = VectorQuantizer(n_codes=codebook_size, d=D, beta=beta)

        # shared decoder upsampling back to 64x1024
        self.dec = nn.Sequential(
            conv3(D, 96),
            deconv3(96, 64, s=(1,1,2)),
            conv3(64, 64),
            deconv3(64, 32, s=(1,2,2)),
            conv3(32, 32),
            deconv3(32, 32, s=(1,2,2)),
            conv3(32, 32)
        )
        
        # two decoder heads
        self.dec_depth = DepthHeadDec(in_ch=32)
        self.dec_sem   = SemHeadDec(in_ch=32, num_classes=num_classes)

    def encode(self, depth, sem):
        d = self.enc_depth(depth)                 # (B,base,T,H,W)
        s = self.enc_sem(sem)                     # (B,base,T,H,W)
        x = torch.cat([d, s], dim=1)              # (B,2*base,T,H,W)
        z = self.enc(x)                           # (B,D,T,16,128)
        z_q, vq_loss, idx = self.vq(z)            # quantized
        return z, z_q, vq_loss, idx

    # def decode(self, z_q):
    #     feat = self.dec(z_q)                      # (B,64,T,64,1024)
    #     depth_hat = self.dec_depth(feat)          # (B,1,T,64,1024)
    #     sem_logits = self.dec_sem(feat)           # (B,C,T,64,1024)
    #     return depth_hat, sem_logits
    def decode(self, z_q):
        feat = self.dec(z_q)  # may end up slightly larger than (64,1024)
        # Crop to target size (64,1024)
        _, _, T, H, W = z_q.shape
        target_H, target_W = 64, 1024
        feat = feat[:, :, :, :target_H, :target_W]

        depth_hat = self.dec_depth(feat)
        sem_logits = self.dec_sem(feat)
        return depth_hat, sem_logits

    def forward(self, depth, sem, depth_mask=None):
        """
        depth : (B,1,T,64,1024) float, put -1 where invalid (or provide depth_mask)
        sem   : (B,T,64,1024)   long
        depth_mask: (B,1,T,64,1024) bool, True where valid; if None, infer from depth!=-1
        """
        if depth_mask is None:
            depth_mask = (depth > 0) & (depth != -1)

        # ----- 2. Replace -1 with 0 before encoding -----
        depth_enc = depth.clone()
        depth_enc[~depth_mask] = 0.0
        z, z_q, vq_loss, idx = self.encode(depth, sem)
        depth_hat, sem_logits = self.decode(z_q)

        # losses (minimal)
        if depth_mask is None:
            depth_mask = (depth > 0) & (depth != -1)

        # masked L1 for depth
        if depth_mask.any():
            l1 = (depth_hat - depth).abs()
            depth_l1 = (l1 * depth_mask.float()).sum() / (depth_mask.float().sum() + 1e-6)
        else:
            depth_l1 = depth_hat.abs().mean()  # fallback

        # CE for semantics
        sem_loss = F.cross_entropy(
            sem_logits.permute(0,2,3,4,1).reshape(-1, sem_logits.size(1)),
            sem.reshape(-1),
            ignore_index=0  # change if you use a different "void" id
        )

        loss = depth_l1 + sem_loss + vq_loss

        return {
            "depth_hat": depth_hat,
            "sem_logits": sem_logits,
            "loss": loss,
            "loss_depth": depth_l1,
            "loss_sem": sem_loss,
            "loss_vq": vq_loss,
            "codes": idx
        }
