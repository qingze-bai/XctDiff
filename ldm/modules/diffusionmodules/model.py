import torch
import torch.nn as nn

def nonlinearity(x):
    return x*torch.sigmoid(x)

def Normalize(in_channels, normal_type='GN', num_groups=32, eps=1e-6, affine=True):
    if normal_type == 'GN':
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps, affine=affine)
    elif normal_type == 'LN':
        return torch.nn.LayerNorm(normalized_shape=in_channels, eps=eps, elementwise_affine=affine)

class AttnBlock(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels, num_groups=num_groups)
        self.q = torch.nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.k = torch.nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.v = torch.nn.Conv3d(in_channels, in_channels, 1, 1, 0)
        self.proj_out = torch.nn.Conv3d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w,d = q.shape
        q = q.reshape(b,c,h*w*d)
        q = q.permute(0,2,1)   # b,hwd,c
        k = k.reshape(b,c,h*w*d) # b,c,hwd
        w_ = torch.bmm(q,k)     # b,hwd,hwd    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w*d)
        w_ = w_.permute(0,2,1)   # b,hwd,hwd (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hwd (hwd of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w,d)
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
        print("暂时未实现")
        None

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels, in_channels, 2, 2, 0)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = ResnetBlock(in_channels, in_channels, num_groups=8)
            # self.conv = torch.nn.Conv3d(in_channels, in_channels,3, 1, 1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="trilinear")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv_shortcut=False, num_groups=32, dropout=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = Normalize(in_channels, normal_type='GN', num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1)

        self.norm2 = Normalize(out_channels, normal_type='GN', num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, 1, 1, 0)

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

class ConvnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, use_conv_shortcut=False, dropout=0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut

        self.dwconv = nn.Conv3d(in_channels, out_channels, 5, 2, groups=in_channels)
        self.norm1 = Normalize(out_channels, normal_type='LN')
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = x
        h = self.dwconv(h)
        h = h.permute(0, 2, 3, 4, 1)
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        h = nonlinearity(h)
        h = h.permute(0, 4, 1, 2, 3)
        h = self.dropout(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h

class Encoder(nn.Module):
    def __init__(self, *, in_channels, out_channels, ch, resolution,
                 attn_resolutions, ch_mult=(1,2,4,8), num_blocks=2,
                 z_channels=256, block_type='ResnetBlock',
                 attn_type="vanilla", dropout=0, resamp_with_conv=True, num_groups=32):
        super().__init__()

        assert block_type in ['ResnetBlock', 'ConvnetBlock', None], 'block_type only in ResnetBlock, ConvnetBlock and None'
        if block_type == None: block_type = 'ResnetBlock'

        self.num_resolutions = len(ch_mult)
        self.num_blocks = num_blocks

        self.init_conv = torch.nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        conv_block = ResnetBlock if block_type == 'ResnetBlock' else ConvnetBlock

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_blocks):

                block.append(conv_block(in_channels=block_in, out_channels=block_out, num_groups=num_groups, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, num_groups=num_groups))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

            self.norm = Normalize(block_in, normal_type='GN', num_groups=num_groups)
            self.conv_out = torch.nn.Conv3d(block_in, z_channels, 3, 1, 1)

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

        # end
        h = hs[-1]
        h = self.norm(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, *, in_channels, out_channels, ch, resolution,
                 attn_resolutions, ch_mult=(1,2,4,8), num_blocks=2,
                 z_channels=256, block_type='ResnetBlock',
                 attn_type="vanilla", dropout=0, resamp_with_conv=True, num_groups=32):
        super().__init__()
        assert block_type in ['ResnetBlock', 'ConvnetBlock', None], 'block_type only in ResnetBlock, ConvnetBlock and None'
        if block_type == None: block_type = 'ResnetBlock'

        self.num_resolutions = len(ch_mult)
        self.num_blocks = num_blocks

        conv_block = ResnetBlock if block_type == 'ResnetBlock' else ConvnetBlock

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = torch.nn.Conv3d(z_channels, block_in, 3, 1, 1)

        # middle
        self.mid_norm = Normalize(block_in, normal_type='GN', num_groups=num_groups)
        self.mid_block = torch.nn.Conv3d(block_in, block_in, 3, 1, 1)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_blocks + 1):
                block.append(conv_block(in_channels=block_in, out_channels=block_out, num_groups=num_groups, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type, num_groups=num_groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in, normal_type='GN', num_groups=num_groups)
        self.conv_out = torch.nn.Conv3d(block_in, out_channels, 3, 1, 1)

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)
        # middle
        h = self.mid_norm(h)
        h = nonlinearity(h)
        h = self.mid_block(h)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_blocks+1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h