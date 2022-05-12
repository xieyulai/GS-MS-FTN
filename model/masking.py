import torch
import torch.nn as nn


def multi_make_masks(feature_stacks, captions, pad_idx):
    masks = {}

    aud_fea = feature_stacks['audio']
    vid_fea = feature_stacks['rgb'] + feature_stacks['flow']
    tex_fea = feature_stacks['text']
    if captions is None:
        masks['A_mask'] = mask(aud_fea[:, :, 0], None, pad_idx)
        masks['V_mask'] = mask(vid_fea[:, :, 0], None, pad_idx)
        masks['T_mask'] = mask(tex_fea[:, :, 0], None, pad_idx)
    else:
        masks['V_mask'], masks['C_mask'] = mask(vid_fea[:, :, 0], captions, pad_idx)
        masks['A_mask'] = mask(aud_fea[:, :, 0], None, pad_idx)
        masks['T_mask'] = mask(tex_fea[:, :, 0], None, pad_idx)

    return masks


def subsequent_mask(size):
    '''
    in: size               Sc
    out: (1, size, size)   (1,Sc,Sc)
    '''
    mask = torch.ones(1, size, size)
    mask = torch.tril(mask, 0)    # 返回矩阵下三角部分，其余部分定义为0

    return mask.byte()


def mask(src, trg, pad_idx):
    if src is not None and trg is None:
        # masking the padding. src shape: (B, Sv) -> (B, 1, Sv)
        src_mask = (src != pad_idx).unsqueeze(1)
        return src_mask
    if src is None and trg is not None:
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).bool()
        return trg_mask
    if src is not None and trg is not None:
        src_mask = (src != pad_idx).unsqueeze(1)
        trg_mask = (trg != pad_idx).unsqueeze(-2) & subsequent_mask(trg.size(-1)).type_as(src_mask.data)
        return src_mask, trg_mask


# 上采样
def upsample(x, scale):
    x1 = x.permute(0, 2, 1)
    m = nn.Upsample(scale_factor=scale, mode='nearest')
    y = m(x1).permute(0, 2, 1)

    return y
