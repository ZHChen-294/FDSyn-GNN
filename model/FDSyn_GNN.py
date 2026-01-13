import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def normalize_minmax_pm1(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    min_val = x.min(dim=1, keepdim=True).values
    max_val = x.max(dim=1, keepdim=True).values
    return 2.0 * (x - min_val) / (max_val - min_val + eps) - 1.0


class GRUEncoder(nn.Module):
    def __init__(self, node_num: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=node_num,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, node_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)              # [B, T, 2H]
        gru_out = self.proj(gru_out)          # [B, T, N]
        mean_pooled = normalize_minmax_pm1(gru_out.mean(dim=1))
        max_pooled = normalize_minmax_pm1(gru_out.max(dim=1).values)
        return torch.stack([mean_pooled, max_pooled], dim=-1)  # [B, N, 2]


class LayerGIN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 2, epsilon: bool = True, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.epsilon = nn.Parameter(torch.tensor([[0.0]])) if epsilon else 0.0

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        n = x.size(0)

        q = self.q_proj(x).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.num_heads, self.head_dim)

        scores = torch.einsum("nhd,mhd->nhm", q, k) / (self.head_dim ** 0.5)

        a = a.coalesce()
        row, col = a.indices()
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[row, :, col] = True
        scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        v_agg = torch.einsum("nhm,mhd->nhd", attn, v).reshape(n, -1)
        v_agg = self.out_proj(v_agg)

        v_agg = v_agg + (self.epsilon * x if isinstance(self.epsilon, nn.Parameter) else self.epsilon * x)
        return self.mlp(v_agg)


class ModuleSERO(nn.Module):
    def __init__(self, hidden_dim: int, input_dim: int, dropout: float = 0.1, upscale: float = 1.0):
        super().__init__()
        mid = int(round(upscale * hidden_dim))
        self.embed = nn.Sequential(nn.Linear(hidden_dim, mid), nn.BatchNorm1d(mid), nn.GELU())
        self.attend = nn.Linear(mid, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, node_axis: int = 1):
        x_readout = x.mean(node_axis)              # [B, C]
        x_embed = self.embed(x_readout)            # [B, mid]
        g_attn = torch.sigmoid(self.attend(x_embed))  # [B, input_dim]
        return (x * self.dropout(g_attn.unsqueeze(-1))).mean(node_axis), g_attn


class FDSyn_GNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        sparsity: int,
        dropout: float = 0.1,
        cls_token: str = "sum",
        readout: str = "sero",
        garo_upscale: float = 1.0,
    ):
        super().__init__()
        if cls_token not in {"sum", "mean", "param"}:
            raise ValueError("cls_token must be one of: sum, mean, param")

        self.cls_token = cls_token
        self.token_parameter = nn.Parameter(torch.randn([num_layers, 1, 1, hidden_dim])) if cls_token == "param" else None

        self.num_classes = num_classes
        self.sparsity = sparsity

        self.time_series_encoder = GRUEncoder(input_dim, hidden_dim)
        self.initial_linear = nn.Linear(input_dim + 2, hidden_dim)

        self.gnn_layers = nn.ModuleList([LayerGIN(hidden_dim, hidden_dim, hidden_dim, num_heads=num_heads) for _ in range(num_layers)])
        self.readout_modules = nn.ModuleList([ModuleSERO(hidden_dim=hidden_dim, input_dim=input_dim, dropout=0.1, upscale=garo_upscale) for _ in range(num_layers)])
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _collate_adjacency(self, a: torch.Tensor, sparsity: int) -> torch.Tensor:
        b, n, _ = a.shape
        device = a.device

        i_list, v_list = [], []
        for s in range(b):
            _a = a[s]
            thr = np.percentile(_a.detach().cpu().numpy(), 100 - sparsity)
            keep = (_a > thr).nonzero(as_tuple=False)  # [E, 2]
            if keep.numel() == 0:
                continue
            keep = keep + s * n
            i_list.append(keep)
            v_list.append(torch.ones(keep.size(0), device=device, dtype=torch.float32))

        if len(i_list) == 0:
            idx = torch.empty((2, 0), device=device, dtype=torch.long)
            val = torch.empty((0,), device=device, dtype=torch.float32)
        else:
            idx = torch.cat(i_list, dim=0).t().contiguous()
            val = torch.cat(v_list, dim=0)

        return torch.sparse_coo_tensor(idx, val, (b * n, b * n), device=device)

    def forward(self, v: torch.Tensor, a: torch.Tensor, t: torch.Tensor):
        logit = 0.0
        node_attn_list = []
        latent_list = []

        b, n = a.shape[:2]

        time_encoding = self.time_series_encoder(t)          # [B, N, 2]
        h = torch.cat([a, time_encoding], dim=2)             # [B, N, N+2]
        h = rearrange(h, "b n c -> (b n) c")                 # [B*N, N+2]
        h = self.initial_linear(h)                           # [B*N, H]

        adj = self._collate_adjacency(a, self.sparsity)      # sparse [B*N, B*N]

        for layer, (G, R, L) in enumerate(zip(self.gnn_layers, self.readout_modules, self.linear_layers)):
            h = G(h, adj)                                    # [B*N, H]
            h_bridge = rearrange(h, "(b n) c -> b n c", b=b, n=n)

            h_readout, node_attn = R(h_bridge, node_axis=1)  # [B, H], [B, input_dim]

            if self.token_parameter is not None:
                h_readout = torch.cat([h_readout, self.token_parameter[layer].expand(-1, h_readout.size(1), -1)], dim=0)

            latent = h_readout
            logit = logit + self.dropout(L(latent))

            node_attn_list.append(node_attn)
            latent_list.append(latent)

        logit = logit.squeeze(1)

        if len(node_attn_list) > 4:
            node_attn_list = node_attn_list[:4]
            latent_list = latent_list[:4]

        attention = {"node-attention": torch.stack(node_attn_list, dim=1).detach().cpu()}
        latent = torch.stack(latent_list, dim=1)

        return logit, attention, latent


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
