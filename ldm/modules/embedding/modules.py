import torch
import torch.nn as nn
from einops import rearrange
from ldm.modules.diffusionmodules.model import Normalize, nonlinearity

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv_shortcut=False, num_groups=32, dropout=0, spatial_dims=3):
        super().__init__()

        Conv = nn.Conv3d if spatial_dims==3 else nn.Conv2d

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = Normalize(in_channels, normal_type='GN', num_groups=num_groups)
        self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)

        self.norm2 = Normalize(out_channels, normal_type='GN', num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = Conv(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class DoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv_shortcut=False, num_groups=32, dropout=0, spatial_dims=3):
        super().__init__()

        Conv = nn.Conv3d if spatial_dims==3 else nn.Conv2d

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = Normalize(in_channels, normal_type='GN', num_groups=num_groups)
        self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)

        self.norm2 = Normalize(out_channels, normal_type='GN', num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h

class Mlp(nn.Module):
    """ Multilayer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 2, 2, 0)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class AttnBlock(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels, num_groups=num_groups)
        self.q = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hwd,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c, hw (hwd of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)
        h_ = self.proj_out(h_)
        return x+h_

def make_attn(in_channels, attn_type="vanilla", num_groups=32):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    # print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        None

class Encoder(nn.Module):
    def __init__(self, *, ch, resolution, attn_resolutions, ch_mult=(1,2,4),
                 num_blocks=2, z_channels=16, attn_type="vanilla", dropout=0, num_groups=32):
        super(Encoder, self).__init__()
        self.init_conv = torch.nn.Conv2d(1, ch, kernel_size=3, stride=1, padding=1)
        ch_mult = ch_mult
        in_ch_mult = (1,) + tuple(ch_mult)
        curr_res = resolution
        self.num_blocks = num_blocks
        self.num_resolutions = len(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in, out_channels=block_out, num_groups=num_groups, dropout=dropout, spatial_dims=2))
                block_in = block_out
                if curr_res in attn_resolutions:
                   attn.append(make_attn(block_in, attn_type=attn_type, num_groups=num_groups))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # mid
        self.mid = nn.Module()
        self.mid.block = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       num_groups=num_groups,
                                       dropout=dropout,
                                       spatial_dims=2
                                       )
        self.mid.attn = make_attn(block_in, attn_type=attn_type)
        # 2d convert to 3d
        self.mid.norm = Normalize(32, normal_type='LN')
        self.mid.mlp = Mlp(in_features=32)
        self.mid.norm2 = Normalize(32, normal_type='LN')
        self.mid.mlp2 = Mlp(in_features=32)
        self.mid.act = nn.GELU()
        self.conv_out = DoubleBlock(in_channels=16, out_channels=z_channels, num_groups=4)

    def forward(self, x):
        hs = [self.init_conv(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = self.mid.block(hs[-1])
        h = self.mid.attn(h)
        B, C, H, W = h.shape
        h = h.reshape(B, -1, H, H, W)
        h = rearrange(h, 'b c d h w  -> b c h w d')
        h = self.mid.mlp(self.mid.norm(h))
        h = self.mid.act(h)
        h = self.mid.mlp2(self.mid.norm2(h))
        h = rearrange(h, 'b c h w d  -> b c h d w')
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, ch, resolution, attn_resolutions, ch_mult=(1,2,4),
                 num_blocks=2, z_channels=16, attn_type="vanilla", dropout=0, num_groups=32):
        super().__init__()
        # middle
        self.conv = nn.Sequential(
            torch.nn.Conv3d(z_channels, ch, 3, 1, 1),
            DoubleBlock(in_channels=ch, out_channels=ch // 2, num_groups=num_groups, dropout=0),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            DoubleBlock(in_channels=ch // 2, out_channels=ch // 4, num_groups=num_groups // 2, dropout=0),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(ch // 4, 1, 3, 1, 1)
        )

    def forward(self, z):
        # z to block_in
        h = self.conv(z)
        return h
