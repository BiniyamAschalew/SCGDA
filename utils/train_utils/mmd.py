import torch
import numpy as np

from Learn.Clean_SCGDA.utils.train_utils.sinkhorn import Sinkhorn

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = torch.as_tensor(source)
    target = torch.as_tensor(target)
    n_samples = int(source.size(0)) + int(target.size(0))
    total = torch.cat([source, target], dim=0)
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    L2_distance = torch.cdist(total, total, p=2).pow(2)
    finite_distance = torch.isfinite(L2_distance)
    if finite_distance.any():
        finite_max = torch.max(L2_distance[finite_distance]) if finite_distance.any() else torch.tensor(0.0, device=L2_distance.device)
        if finite_max.item() > 0:
            L2_distance = torch.where(finite_distance, L2_distance, finite_max)
    else:
        L2_distance = torch.zeros_like(L2_distance)
    if fix_sigma is not None:
        bandwidth = torch.as_tensor(fix_sigma, device=L2_distance.device, dtype=L2_distance.dtype)
    else:
        bandwidth = (torch.sum(L2_distance) + 1e-6) / (n_samples**2 - n_samples)

    bandwidth = torch.nan_to_num(
        bandwidth,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )
    if bandwidth.item() <= 0:
        bandwidth = torch.tensor(1.0, device=L2_distance.device, dtype=L2_distance.dtype)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    if source_num == 0 or target_num == 0:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)

    kernels = guassian_kernel(source_feat, 
                              target_feat,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    
    batch_size = min(int(source_num), int(target_num))
    if batch_size == 0:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_num = int(source_num)
    target_num = int(target_num)
    sampling_num = int(sampling_num)
    times = int(times)

    if source_num == 0 or target_num == 0 or sampling_num <= 0 or times <= 0:
        return torch.tensor(
            0.0,
            device=source_feat.device,
            dtype=source_feat.dtype,
        )

    sampling_num = min(sampling_num, source_num, target_num)

    source_sample = torch.randint(
        source_num,
        (times, sampling_num),
        device=source_feat.device,
    )
    target_sample = torch.randint(
        target_num,
        (times, sampling_num),
        device=target_feat.device,
    )

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd


def mmd_kernel(
    source_feat,
    target_feat,
    source_role,
    target_role,
    sampling_num=1000,
    times=5,
    kernel_mul=2.0,
    kernel_num=5,
    fix_sigma=None,
    role_kernel_mul=None,
    role_kernel_num=None,
    role_fix_sigma=None,
):
    if role_kernel_mul is None:
        role_kernel_mul = kernel_mul
    if role_kernel_num is None:
        role_kernel_num = kernel_num
    if role_fix_sigma is None:
        role_fix_sigma = fix_sigma

    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]
        source_sample_role = source_role[source_sample[i]]
        target_sample_role = target_role[target_sample[i]]

        feat_kernel = guassian_kernel(
            source_sample_feat,
            target_sample_feat,
            kernel_mul=kernel_mul,
            kernel_num=kernel_num,
            fix_sigma=fix_sigma,
        )
        role_kernel = guassian_kernel(
            source_sample_role,
            target_sample_role,
            kernel_mul=role_kernel_mul,
            kernel_num=role_kernel_num,
            fix_sigma=role_fix_sigma,
        )

        kernels = feat_kernel * role_kernel
        batch_size = min(int(source_sample_feat.size()[0]), int(target_sample_feat.size()[0]))

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        mmd = mmd + torch.mean(XX + YY - XY - YX)

    mmd = mmd / times
    return mmd
