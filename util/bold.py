import torch

# corrcoef based on
# https://github.com/pytorch/pytorch/issues/1254
def corrcoef(x):
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    c = torch.clamp(c, -1.0, 1.0)
    return c


def process_fc(minibatch_timeseries: torch.Tensor, self_loop: bool = True, eps: float = 1e-8) -> torch.Tensor:
    if minibatch_timeseries.ndim != 3:
        raise ValueError(
            f"process_fc expects [B, T, N], got shape {tuple(minibatch_timeseries.shape)}"
        )

    x = minibatch_timeseries.transpose(1, 2)  # [B, N, T]
    x = x - x.mean(dim=2, keepdim=True)
    t = x.size(2)
    if t < 2:
        raise ValueError("process_fc requires T >= 2")

    cov = torch.matmul(x, x.transpose(1, 2)) / (t - 1)  # [B, N, N]
    var = torch.diagonal(cov, dim1=1, dim2=2).clamp_min(eps)  # [B, N]
    std = torch.sqrt(var)
    corr = cov / (std.unsqueeze(2) * std.unsqueeze(1))
    corr = corr.clamp(-1.0, 1.0)

    if not self_loop:
        n = corr.size(-1)
        corr = corr - torch.eye(n, device=corr.device, dtype=corr.dtype).unsqueeze(0)

    return corr


