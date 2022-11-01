import torch
import torch.nn as nn
from einops import rearrange


class PathModel(nn.Module):
    def __init__(self, channel, dim, exp_rate = 0.5):
        super().__init__()
        inner_channel = int(channel * exp_rate)
        inner_dim = int(dim * exp_rate)
        self.chan_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(channel, inner_channel, 1),
            nn.GELU(),
            nn.Conv1d(inner_channel, channel, 1)
        )
        self.inner_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, mask):
        x = self.chan_update(x * mask) + x
        x = self.inner_update(x * mask) + x
        return self.norm(x)


class LinkModel(nn.Module):
    def __init__(self, channel, dim, exp_rate = 0.5):
        super().__init__()
        inner_channel = int(channel * exp_rate)
        inner_dim = int(dim * exp_rate)

        self.chan_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Conv1d(channel, inner_channel, 1),
            nn.GELU(),
            nn.Conv1d(inner_channel, channel, 1)
        )
        self.inner_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        
        self.norm = nn.LayerNorm(dim)
        self.merge = nn.MaxPool1d(2)

    def forward(self, l_st, link, batch):
        x = torch.cat((l_st, link), dim = 1)
        x = rearrange(x, '(b n) d -> b n d', b = batch)
        x = self.chan_update(x) + x
        x = self.inner_update(x) + x
        out = self.merge(self.norm(x))
        out = rearrange(out, 'b n d -> (b n) d')
        return out


class NewModel(nn.Module):
    def __init__(self, args, max_len, n_links):
        super().__init__()

        self.args = args
        self.max_len = max_len
        
        self.bw_emb = nn.Linear(1, args.dim)
        self.tr_emb = nn.Linear(1, args.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len + 1, args.dim))
        self.bw_pad = nn.Parameter(torch.randn(1, args.dim))

        self.path_w = nn.Linear(args.dim, 1)
        self.norm = nn.LayerNorm(args.dim)

        self.update = nn.ModuleList([])
        for _ in range(args.depth):
            self.update.append(nn.ModuleList([
                PathModel(max_len+1, args.dim, 0.5),
                LinkModel(n_links, args.dim*2, 2)
            ]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(args.dim),
            nn.Linear(args.dim, args.mlp_dim),
            nn.GELU(),
            nn.Linear(args.mlp_dim, args.mlp_dim),
            nn.GELU(),
            nn.Linear(args.mlp_dim, 1)
        )

    def forward(self, bw, tr, p_lidx, mask):
        bw = self.bw_emb(bw.unsqueeze(2))  # [B, Ne, D]
        tr = self.tr_emb(tr.unsqueeze(2))  # [B, Np, D]
        p_lidx = p_lidx.view(-1)  # [B*Np*Lp]
        batch = bw.size(0)

        mask = rearrange(mask, 'b n s -> (b n) s 1') # [B*Np, Lp+1, 1]
        # 链路状态l_st, 路径状态p_st
        p_st = rearrange(tr, 'b n d -> (b n) 1 d')  # [B*Np, 1, D]
        l_st = rearrange(bw, 'b n d -> (b n) d')  # [B*Ne, D]
        l_st = torch.cat((self.bw_pad, l_st), dim=0)  # [B*Ne+1, D]

        for path_update, link_update in self.update:
            p_bw = l_st[p_lidx]  # [B*Np*Lp, D]
            p_bw = rearrange(p_bw, '(n s) d -> n s d', s = self.max_len)  # [B*Np, Lp, D]
            # path = torch.cat((p_st, p_bw), dim=1) + self.pos_emb 
            path = torch.cat((p_st, p_bw), dim=1)  # [B*Np, Lp+1, D]

            path = path_update(path, mask)
            p_st, p_bw = torch.split(path, [1, self.max_len], dim=1)
            p_w = self.path_w(p_st)  # [B*Np, 1, 1]
            p_w = torch.sigmoid(p_w)
            p_bw = p_w * p_bw  # [B*Np, Lp, D]
            p_bw = rearrange(p_bw, 'n s d -> (n s) d')  # [B*Np*Lp, D]

            link = torch.zeros_like(l_st, dtype=p_bw.dtype)
            link.index_add_(0, p_lidx, p_bw)  # link此时输入很大，会使得p_bw梯度变小
            link = self.norm(link[1:])  # 做LayerNorm降低数值
            link = link_update(l_st[1:], link, batch)  # [B*Ne, D]
            l_st = torch.cat((self.bw_pad, link), dim=0)   
        
        target = self.mlp_head(p_st.squeeze(1)).squeeze(1)
        return target