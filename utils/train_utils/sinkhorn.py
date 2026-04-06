import torch
import torch.nn.functional as F
import warnings

from geomloss import SamplesLoss


def _normalize_transport_features(source_feat, target_feat, eps=1e-6):
    all_feat = torch.cat([source_feat, target_feat], dim=0)
    mean = all_feat.mean(dim=0, keepdim=True)
    std = all_feat.std(dim=0, keepdim=True).clamp_min(eps)

    source_feat = (source_feat - mean) / std
    target_feat = (target_feat - mean) / std

    source_feat = F.normalize(source_feat, p=2, dim=1, eps=eps)
    target_feat = F.normalize(target_feat, p=2, dim=1, eps=eps)
    return source_feat, target_feat


def _quantile_transport_1d(sorted_source, sorted_target):
    grid_size = max(int(sorted_source.numel()), int(sorted_target.numel()))
    if grid_size <= 0:
        return sorted_source.new_tensor(0.0)
    if grid_size == 1:
        return (sorted_source[0] - sorted_target[0]).abs()

    q = torch.linspace(0.0, 1.0, steps=grid_size, device=sorted_source.device, dtype=sorted_source.dtype)

    def _interp(sorted_values):
        n = int(sorted_values.numel())
        if n == 1:
            return sorted_values.expand_as(q)
        pos = q * (n - 1)
        left = torch.floor(pos).long()
        right = torch.ceil(pos).long()
        weight = pos - left.to(pos.dtype)
        return sorted_values[left] * (1.0 - weight) + sorted_values[right] * weight

    interp_source = _interp(sorted_source)
    interp_target = _interp(sorted_target)
    return (interp_source - interp_target).abs().mean()


def _sliced_w1_fallback(source_feat, target_feat, num_projections=16):
    source_feat = torch.as_tensor(source_feat)
    target_feat = torch.as_tensor(target_feat)

    if source_feat.numel() == 0 or target_feat.numel() == 0:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)

    if source_feat.dim() != 2 or target_feat.dim() != 2:
        raise ValueError("Sliced W1 fallback expects 2D tensors of shape [N, D].")

    dim = int(source_feat.size(1))
    if dim <= 0:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)

    if dim == 1:
        directions = torch.ones((1, 1), device=source_feat.device, dtype=source_feat.dtype)
    else:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        directions = torch.randn((num_projections, dim), generator=gen, dtype=source_feat.dtype)
        directions = directions.to(source_feat.device)
        directions = F.normalize(directions, p=2, dim=1, eps=1e-12)

    proj_source = source_feat @ directions.t()
    proj_target = target_feat @ directions.t()

    losses = []
    for proj_idx in range(proj_source.size(1)):
        sorted_source = torch.sort(proj_source[:, proj_idx]).values
        sorted_target = torch.sort(proj_target[:, proj_idx]).values
        losses.append(_quantile_transport_1d(sorted_source, sorted_target))

    return torch.stack(losses).mean()


def get_Sinkhorn(
    source_feat,
    target_feat,
    blur=0.05,
    p=2,
    scaling=0.9,
    debias=True,
    backend="auto",
):
    eps = 1e-8
    source_feat = torch.as_tensor(source_feat)
    target_feat = torch.as_tensor(target_feat)

    source_num = int(source_feat.size(0))
    target_num = int(target_feat.size(0))
    if source_num == 0 or target_num == 0:
        return torch.tensor(
            0.0,
            device=source_feat.device,
            dtype=source_feat.dtype,
        )

    source_feat = torch.nan_to_num(source_feat, nan=0.0, posinf=0.0, neginf=0.0)
    target_feat = torch.nan_to_num(target_feat, nan=0.0, posinf=0.0, neginf=0.0)

    if source_feat.dim() != 2 or target_feat.dim() != 2:
        raise ValueError("Sinkhorn expects 2D tensors of shape [N, D].")

    source_feat, target_feat = _normalize_transport_features(source_feat, target_feat, eps=1e-6)

    all_feat = torch.cat([source_feat, target_feat], dim=0)
    diameter = torch.cdist(all_feat, all_feat, p=2).amax()
    diameter_val = float(diameter.detach().item())
    if diameter_val <= eps:
        return torch.tensor(0.0, device=source_feat.device, dtype=source_feat.dtype)

    adaptive_blur = max(float(blur), 0.05 * max(diameter_val, 1e-3))
    scaling = min(max(float(scaling), 0.5), 0.999)

    sinkhorn_loss = SamplesLoss(
        "sinkhorn",
        p=p,
        blur=adaptive_blur,
        scaling=scaling,
        debias=debias,
        backend=backend,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            loss = sinkhorn_loss(source_feat, target_feat)
    except Exception:
        loss = _sliced_w1_fallback(source_feat, target_feat)

    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(loss):
        loss = _sliced_w1_fallback(source_feat, target_feat)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def Sinkhorn(
    source_feat,
    target_feat,
    sampling_num=1000,
    times=5,
    blur=0.05,
    p=2,
    scaling=0.9,
    debias=True,
    backend="auto",
):
    source_feat = torch.as_tensor(source_feat)
    target_feat = torch.as_tensor(target_feat)

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

    sinkhorn = source_feat.new_tensor(0.0)
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        sinkhorn = sinkhorn + get_Sinkhorn(
            source_sample_feat,
            target_sample_feat,
            blur=blur,
            p=p,
            scaling=scaling,
            debias=debias,
            backend=backend,
        )

    sinkhorn = sinkhorn / times
    return sinkhorn
