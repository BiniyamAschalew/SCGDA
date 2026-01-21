import torch
import numpy as np


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = (torch.sum(L2_distance.data) + 1e-6) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    
    kernels = guassian_kernel(source_feat, 
                              target_feat,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    
    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))  
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def MMD(source_feat, target_feat, sampling_num=1000, times=5):
    
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

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
