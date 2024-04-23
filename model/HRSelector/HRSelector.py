from  torchvision import models
import sys
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from misc.utils import *
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 2, patch_size: int = 4, emb_size: int = 256, img_size: int = 224):
        self.patch_size = patch_size
        super(PatchEmbedding, self).__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Conv2d(emb_size, emb_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(emb_size),
            nn.ReLU(),
            nn.Conv2d(emb_size, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        # 位置编码信息，一共有(img_size // patch_size)**2 + 1(cls token)个位置向量
        self.positions = nn.Parameter(torch.randn((img_size // 4 // patch_size)**2, emb_size))
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        x = self.positions + x
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super(MultiHeadAttention, self).__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads

        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)

        
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)

        att = F.softmax(energy, dim=-1) / scaling

        att = self.att_drop(att)
 
        
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
  
        out = rearrange(out, "b h n d -> b n (h d)")
    
        out = self.projection(out)

        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x = x + res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super(FeedForwardBlock, self).__init__()
        self.FFN = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )
    
    def forward(self, inport):
        return self.FFN(inport)


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 emb_size: int = 256,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super(TransformerEncoderBlock, self).__init__()
        self.ViTBlock = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            )
        )
    def forward(self, inport):
        return self.ViTBlock(inport)

class ReformResolution(nn.Module):
    def __init__(self, inport_dim=256):
        super(ReformResolution, self).__init__()
        self.Decode_2X = nn.Sequential(
            nn.Linear(inport_dim, 2*inport_dim, bias=False),
            nn.ReLU()
        )
        self.norm_2 = nn.LayerNorm(inport_dim // 2)
        self.ViTDecode_2 = TransformerEncoderBlock(emb_size=inport_dim//2)

        self.Decode_4X = nn.Sequential(
            nn.Linear(inport_dim // 2, inport_dim, bias=False),
            nn.ReLU()
        )
        self.norm_4 = nn.LayerNorm(inport_dim // 4)
        self.ViTDecode_4 = TransformerEncoderBlock(emb_size=inport_dim//4)
        self.outlayer = nn.Sequential(
            nn.Conv2d(in_channels=inport_dim//4, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 2, 1, 1, 0),
            nn.Softmax(dim=1)
        )
        
    def forward(self, inport):
        
        X2_inport = self.Decode_2X(inport)
        B, L, C = X2_inport.size()
        H, W = int(L**0.5), int(L**0.5)
        X2_inport = X2_inport.view(B, H, W, C)
        X2_inport = rearrange(X2_inport, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        X2_inport = X2_inport.view(B,-1,C//4)
        X2_inport = self.norm_2(X2_inport)
        X2_outport = self.ViTDecode_2(X2_inport)
        
        X4_inport = X2_outport
        # print(X4_inport.size())
        X4_inport = self.Decode_4X(X4_inport)
        B, L, C = X4_inport.size()
        H, W = int(L**0.5), int(L**0.5)

        X4_inport = X4_inport.view(B, H, W, C)
        X4_inport = rearrange(X4_inport, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        X4_inport = X4_inport.view(B,-1,C//4)
        X4_inport = self.norm_4(X4_inport)

        X4_outport = self.ViTDecode_4(X4_inport)

        B, L, C = X4_outport.size()
        H, W = int(L**0.5), int(L**0.5)

        resolution_map = X4_outport.view(B, H, W, C).permute([0, 3, 1, 2])
   
        return self.outlayer(resolution_map)






mode = 'Vgg_bn'
class HRSelector(nn.Module):
    def __init__(self, img_size):
        super(HRSelector, self).__init__()
     
        self.encoder = nn.Sequential(
            PatchEmbedding(img_size=img_size),
            TransformerEncoderBlock()
        )
        self.decoder = ReformResolution()
        # initialize_weights(self.encoder())
        # initialize_weights(self.decoder())
        self.scale_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(27648, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),

        )


    def forward(self, x):
        feat = self.encoder(x)
        

        res = self.decoder(feat)
        feat = F.adaptive_avg_pool1d(feat, (48))
        feat = feat.view(feat.shape[0], -1)
        # print(feat.size())
        pred_scale = self.scale_head(feat).clamp(min=1.5, max=2.5)
        return res, pred_scale



class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,in_channels,out_channels,num_outs,start_level=0,end_level=-1,
                extra_convs_on_inputs=True,bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2d( in_channels[i], out_channels,1,bn=bn, bias=not bn,same_padding=True)

            fpn_conv = Conv2d( out_channels, out_channels,3,bn=bn, bias=not bn,same_padding=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)


    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = F.interpolate(laterals[i], size=prev_shape, mode='nearest') + laterals[i - 1]

        # build outputs
        # part 1: from original levels
        outs = [ self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) ]


        return tuple(outs)



class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=True, bias=True):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU(inplace=False)
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

if __name__ == "__main__":


    net = VGG16_FPN(pretrained=False).cuda()
    print(net)
    # summary(net,(3,64 ,64 ),batch_size=4)
    net(torch.rand(1,3,64,64).cuda())
