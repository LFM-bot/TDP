import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from transformers import BertModel, AutoTokenizer
from positional_encodings.torch_encodings import PositionalEncoding1D

from src.dataset.dataset import chinese_datasets
from src.model.abstract_detector import AbstractDetector
from src.model.vision_module import mae_vit
from src.utils.utils import HyperParamDict

TDP_TYPE_DESCRIPTIONS = {
    0: "real news with truthful and consistent multimodal content, where the image and the text are both genuine and mutually corroborate each other",
    1: "image fabrication, where the image has been manipulated, doctored, photoshopped, or synthetically generated to depict a scene that did not actually occur",
    2: "entity inconsistency, where the people, objects, or places shown in the image do not match the entities mentioned in the accompanying text",
    3: "event inconsistency, where the event depicted in the image and the event described in the text are two different events",
    4: "time and space inconsistency, where the image and the text conflict on the temporal or spatial context, such as a wrong location, season, or time period",
    5: "ineffective visual information, where the image provides no evidential support for the claim in the text and is irrelevant or non-informative",
}

TDP_UNK_DESCRIPTION = "unknown deception type outside the source deception taxonomy, a fake post whose deception pattern does not fit any known type"

TDP_SOURCE_CLASSES = [0, 1, 2, 3, 4, 5]
TDP_FAKE_SOURCE = [1, 2, 3, 4, 5]
TDP_FAKE_TARGET = [1, 2, 3, 4, 5, 6]

def spherical_kmeans(embeddings, k, niter=10, chunk_size=4096):
    n = embeddings.size(0)
    k = min(k, n)
    device = embeddings.device
    perm = torch.randperm(n, device=device)[:k]
    centroids = embeddings[perm].clone()

    idx = None
    for _ in range(niter):
        idx_parts = []
        for start in range(0, n, chunk_size):
            chunk = embeddings[start:start + chunk_size]
            idx_parts.append((chunk @ centroids.t()).argmax(dim=-1))
        idx = torch.cat(idx_parts, dim=0)
        new_centroids = centroids.clone()
        for c in range(k):
            mask = idx == c
            if mask.any():
                cc = embeddings[mask].mean(dim=0)
                new_centroids[c] = cc / cc.norm().clamp_min(1e-12)
        if torch.allclose(centroids, new_centroids, atol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    idx_parts = []
    for start in range(0, n, chunk_size):
        chunk = embeddings[start:start + chunk_size]
        idx_parts.append((chunk @ centroids.t()).argmax(dim=-1))
    idx = torch.cat(idx_parts, dim=0)
    return idx, centroids

class BottleneckAdapter(nn.Module):

    def __init__(self, dim, rank, dropout=0.):
        super().__init__()
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.up(self.drop(F.gelu(self.down(x))))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

class TransformerExpert(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.block = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, proj_drop=drop)
        self._zero_residual()

    def _zero_residual(self):
        nn.init.zeros_(self.block.attn.proj.weight)
        nn.init.zeros_(self.block.attn.proj.bias)
        nn.init.zeros_(self.block.mlp.fc2.weight)
        nn.init.zeros_(self.block.mlp.fc2.bias)

    def forward(self, x):
        return self.block(x)

    def reset_parameters(self):
        self._zero_residual()

class TDP_FND(AbstractDetector):
    def __init__(self, config, additional_data_dict):
        super(TDP_FND, self).__init__(config)
        self.dataset = config.dataset
        self.unified_dim = 768
        self.image_token_len = 197
        self.text_token_len = config.max_text_len
        self.fusion_seq_len = self.image_token_len + self.text_token_len

        self.tdp_stage = config.tdp_stage
        self.use_unk = self.tdp_stage == 'target'
        self.tdp_source_loss = getattr(config, 'tdp_source_loss', 'multiclass')
        self.num_experts = len(TDP_SOURCE_CLASSES)
        self.num_prototypes = 7 if self.use_unk else 6

        self.adapter_rank = config.tdp_adapter_rank
        self.fusion_layers = config.tdp_fusion_layers
        self.fusion_heads = config.tdp_fusion_heads
        self.moe_rank = config.tdp_moe_rank
        self.num_patterns = config.tdp_num_patterns
        self.cluster_interval = config.tdp_cluster_interval
        self.cluster_chunk_size = config.tdp_cluster_chunk_size
        self.temperature = config.tdp_temperature
        self.lambda_pcl = config.lambda_pcl
        self.lambda_moe = config.lambda_moe
        self.eta_moe = config.eta_moe
        self.beta_div = config.beta_div
        self.use_llm_weak = getattr(config, 'use_llm_weak', False)
        self.eta_w = getattr(config, 'eta_w', 0.1)
        self.llm_conf_threshold = getattr(config, 'llm_conf_threshold', 0.5)

        model_size = 'base'
        self.image_model = mae_vit.__dict__[f"mae_vit_{model_size}_patch16"](norm_pix_loss=False)
        checkpoint = torch.load(
            f'/mnt1/userhome/tangpang/shichenglong/proj/LLMs/mae_checkpoint/mae_pretrain_vit_{model_size}.pth',
            map_location='cpu',
        )
        self.image_model.load_state_dict(checkpoint['model'], strict=False)

        bert_path = (
            '/mnt1/userhome/tangpang/shichenglong/proj/LLMs/bert-base-chinese'
            if self.dataset in chinese_datasets
            else '/mnt1/userhome/tangpang/shichenglong/proj/LLMs/bert-base-uncased'
        )
        print(f'BERT: using {bert_path}')
        self.text_model = BertModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)

        self.text_adapter = BottleneckAdapter(self.unified_dim, self.adapter_rank)
        self.image_adapter = BottleneckAdapter(self.unified_dim, self.adapter_rank)
        self.input_dropout = nn.Dropout(getattr(config, 'tdp_input_dropout', 0.0))

        self.fusion_drop = getattr(config, 'tdp_fusion_drop', 0.0)
        self.fusion_blocks = nn.ModuleList([
            Block(dim=self.unified_dim, num_heads=self.fusion_heads, proj_drop=self.fusion_drop)
            for _ in range(self.fusion_layers)
        ])
        self.fusion_norm = nn.LayerNorm(self.unified_dim)
        self.register_buffer(
            'fusion_pos_embed',
            self._build_sinusoidal_pos(self.fusion_seq_len, self.unified_dim),
        )

        self.expert_type = getattr(config, 'tdp_expert_type', 'adapter')
        self.router = nn.Linear(self.unified_dim, self.num_experts)
        if self.expert_type == 'transformer':
            self.experts = nn.ModuleList([
                TransformerExpert(self.unified_dim, self.fusion_heads, drop=self.fusion_drop)
                for _ in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                BottleneckAdapter(self.unified_dim, self.moe_rank) for _ in range(self.num_experts)
            ])

        self.pattern_attn = nn.MultiheadAttention(
            self.unified_dim, num_heads=self.fusion_heads, batch_first=True
        )
        self.pattern_norm = nn.LayerNorm(self.unified_dim)
        self.pattern_ffn = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim * 2),
            nn.GELU(),
            nn.Linear(self.unified_dim * 2, self.unified_dim),
        )
        self.pattern_ffn_norm = nn.LayerNorm(self.unified_dim)

        self.proto_logits = nn.Parameter(torch.zeros(self.num_prototypes, self.num_patterns))
        self.text_proj = nn.Linear(self.unified_dim, self.unified_dim)

        self.register_buffer('pattern_centers', torch.zeros(self.num_patterns, self.unified_dim))
        self.register_buffer('pattern_dom', torch.full((self.num_patterns,), -1, dtype=torch.long))
        self.register_buffer('patterns_ready', torch.tensor(False))

        self._build_type_descriptions()

        self.text_anchor_mode = config.tdp_text_anchor_mode
        if self.text_anchor_mode == 'frozen':
            with torch.no_grad():
                self.register_buffer('text_anchors_buf', self._compute_text_anchors())
        else:
            self.text_anchors_buf = None

        self._init_weights()
        self._last_loss_dict = {}

    @staticmethod
    def _build_sinusoidal_pos(seq_len, dim):
        pe = PositionalEncoding1D(dim)
        dummy = torch.rand(1, seq_len, dim)
        return pe(dummy).detach()

    def _build_type_descriptions(self):
        descriptions = [TDP_TYPE_DESCRIPTIONS[c] for c in TDP_SOURCE_CLASSES]
        if self.use_unk:
            descriptions.append(TDP_UNK_DESCRIPTION)
        enc = self.tokenizer.batch_encode_plus(
            descriptions,
            padding='max_length',
            truncation=True,
            max_length=48,
            return_tensors='pt',
        )
        self.register_buffer('desc_input_ids', enc['input_ids'])
        self.register_buffer('desc_attention_mask', enc['attention_mask'])

    def _init_weights(self):
        for adapter in [self.text_adapter, self.image_adapter] + list(self.experts):
            adapter.reset_parameters()
        nn.init.zeros_(self.proto_logits)

    def _encode(self, data_dict):
        image = data_dict['image']
        input_ids = data_dict['input_ids']
        attention_mask = data_dict['attention_mask']
        token_type_ids = data_dict['token_type_ids']
        h_v = self.image_model.forward_ying(image)
        h_t = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )[0]
        h_v = h_v + self.image_adapter(h_v)
        h_t = h_t + self.text_adapter(h_t)
        return h_v, h_t

    def _fuse(self, h_v, h_t):
        h = torch.cat([h_v, h_t], dim=1)
        h = h + self.fusion_pos_embed
        h = self.input_dropout(h)
        for blk in self.fusion_blocks:
            h = blk(h)
        return self.fusion_norm(h)

    def _pooled_feature(self, h_f):
        return h_f.mean(dim=1)

    def _encoder_feature(self, h_v, h_t):
        v = h_v.mean(dim=1)
        t = h_t.mean(dim=1)
        return torch.cat([v, t], dim=-1)

    def _moe_fuse(self, h_f):
        pooled = h_f.mean(dim=1)
        router_logits = self.router(pooled)
        rho = F.softmax(router_logits, dim=-1)
        weighted = 0
        for c, expert in enumerate(self.experts):
            weighted = weighted + rho[:, c].unsqueeze(-1).unsqueeze(-1) * expert(h_f)
        h_tilde = h_f + weighted
        z_tilde = h_tilde.mean(dim=1)
        return rho, router_logits, z_tilde

    def _pattern_rep(self, z_tilde):
        if not bool(self.patterns_ready):
            return z_tilde
        m = self.pattern_centers
        kv = m.unsqueeze(0).expand(z_tilde.size(0), -1, -1)
        q = z_tilde.unsqueeze(1)
        attn_out, _ = self.pattern_attn(q, kv, kv)
        h = self.pattern_norm(z_tilde + attn_out.squeeze(1))
        g = self.pattern_ffn_norm(h + self.pattern_ffn(h))
        return g

    def _compute_text_anchors(self):
        out = self.text_model(
            input_ids=self.desc_input_ids,
            attention_mask=self.desc_attention_mask,
        )[0]
        mask = self.desc_attention_mask.unsqueeze(-1).float()
        return (out * mask).sum(1) / mask.sum(1).clamp_min(1.0)

    def _text_anchors(self):
        if self.text_anchor_mode == 'frozen':
            return self.text_anchors_buf
        with torch.no_grad():
            return self._compute_text_anchors()

    def _prototypes(self):
        e = self._text_anchors()
        alpha = F.softmax(self.proto_logits, dim=-1)
        p = alpha @ self.pattern_centers + self.text_proj(e)
        return p

    def _scores(self, g, p):
        g_n = F.normalize(g, dim=-1)
        p_n = F.normalize(p, dim=-1)
        return (g_n @ p_n.t()) / self.temperature

    def _binary_logits(self, sim):
        s_real = sim[:, 0]
        fake_idx = TDP_FAKE_TARGET if self.use_unk else TDP_FAKE_SOURCE
        s_fake = torch.logsumexp(sim[:, fake_idx], dim=-1)
        return torch.stack([s_real, s_fake], dim=-1)

    def inner_forward(self, data_dict):
        h_v, h_t = self._encode(data_dict)
        h_f = self._fuse(h_v, h_t)
        pooled = self._pooled_feature(h_f)
        encoder = self._encoder_feature(h_v, h_t)
        rho, router_logits, z_tilde = self._moe_fuse(h_f)
        g = self._pattern_rep(z_tilde)
        p = self._prototypes()
        sim = self._scores(g, p)
        bin_logits = self._binary_logits(sim)
        return {
            'sim': sim,
            'bin_logits': bin_logits,
            'rho': rho,
            'router_logits': router_logits,
            'g': g,
            'p': p,
            'z_tilde': z_tilde,
            'pooled': pooled,
            'encoder': encoder,
        }

    def forward(self, data_dict):
        out = self.inner_forward(data_dict)
        prob_fake = F.softmax(out['bin_logits'], dim=-1)[:, 1:2]
        return prob_fake

    def get_news_emb(self, data_dict):
        return self.inner_forward(data_dict)['g']

    @staticmethod
    def _load_balance(rho, num_classes):
        rho_bar = rho.mean(dim=0)
        return num_classes * (rho_bar ** 2).sum()

    @staticmethod
    def _decorrelation(p):
        p_n = F.normalize(p, dim=-1)
        gram = p_n @ p_n.t()
        eye = torch.eye(p_n.size(0), device=p_n.device)
        return ((gram - eye) ** 2).sum()

    def _pcl_loss(self, g, fg_label):
        if not bool(self.patterns_ready):
            return g.new_tensor(0.0)
        m = self.pattern_centers
        dom = self.pattern_dom
        g_n = F.normalize(g, dim=-1)
        sim_gm = g_n @ m.t()
        pos_mask = (dom.unsqueeze(0) == fg_label.unsqueeze(1)) & (dom >= 0).unsqueeze(0)
        has_pos = pos_mask.any(dim=-1)
        if not has_pos.any():
            return g.new_tensor(0.0)
        sim_valid = sim_gm[has_pos]
        pos_mask_valid = pos_mask[has_pos]
        fg_valid = fg_label[has_pos]
        neg_mask_valid = (dom.unsqueeze(0) != fg_valid.unsqueeze(1)) & (dom >= 0).unsqueeze(0)

        pos_sim = sim_valid.masked_fill(~pos_mask_valid, float('-inf'))
        pos_logit = pos_sim.max(dim=-1).values
        neg_logits = sim_valid.masked_fill(~neg_mask_valid, float('-inf'))
        denom = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
        loss = -pos_logit + torch.logsumexp(denom, dim=-1)
        return loss.mean()

    def _source_loss(self, data_dict, out):
        fg_label = data_dict['fg_label'].long() if 'fg_label' in data_dict else data_dict['label'].long()
        fg_label = fg_label.clamp(0, 5)
        sim = out['sim']
        rho = out['rho']
        l_route = F.cross_entropy(out['router_logits'], fg_label)
        l_bal = self._load_balance(rho, self.num_experts)
        l_pcl = self._pcl_loss(out['g'], fg_label)
        r_div = self._decorrelation(out['p'])
        if getattr(self, 'tdp_source_loss', 'multiclass') == 'binary':
            bin_label = (fg_label > 0).long()
            l_cls = F.cross_entropy(out['bin_logits'], bin_label)
        else:
            l_cls = F.cross_entropy(sim, fg_label)
        loss = (l_cls + self.lambda_pcl * l_pcl
                + self.lambda_moe * (l_route + l_bal)
                + self.beta_div * r_div)
        self._last_loss_dict = {
            'l_cls': float(l_cls.detach()),
            'l_pcl': float(l_pcl.detach()),
            'l_route': float(l_route.detach()),
            'l_bal': float(l_bal.detach()),
            'r_div': float(r_div.detach()),
        }
        return loss

    def _weak_losses(self, data_dict, out):
        zero = out['sim'].new_tensor(0.0)
        if (not self.use_llm_weak) or ('confidence' not in data_dict):
            return zero, zero
        label = data_dict['label'].long()
        fg = data_dict['fg_label'].long()
        conf = data_dict['confidence'].float()
        consistent = ((label == 0) & (fg == 0)) | ((label == 1) & (fg > 0))
        retained = (conf >= self.llm_conf_threshold) & consistent \
                   & (fg >= 0) & (fg < self.num_prototypes)
        if not retained.any():
            return zero, zero

        sim_r = out['sim'][retained]
        fg_r = fg[retained]
        conf_r = conf[retained]
        ce = F.cross_entropy(sim_r, fg_r, reduction='none')
        l_weak = (ce * conf_r).mean()

        known = fg_r < (self.num_prototypes - 1)
        if known.any():
            router_r = out['router_logits'][retained][known]
            ce_route = F.cross_entropy(router_r, fg_r[known], reduction='none')
            l_route_t = (ce_route * conf_r[known]).mean()
        else:
            l_route_t = zero
        return l_weak, l_route_t

    def _target_loss(self, data_dict, out):
        label = data_dict['label'].long()
        l_bin = F.cross_entropy(out['bin_logits'], label)
        l_bal = self._load_balance(out['rho'], self.num_experts)
        r_div = self._decorrelation(out['p'])
        loss = l_bin + self.eta_moe * l_bal + self.beta_div * r_div
        l_weak, l_route_t = self._weak_losses(data_dict, out)
        loss = loss + self.eta_w * l_weak + self.eta_moe * l_route_t
        self._last_loss_dict = {
            'l_bin': float(l_bin.detach()),
            'l_bal': float(l_bal.detach()),
            'r_div': float(r_div.detach()),
            'l_weak': float(l_weak.detach()),
            'l_route_t': float(l_route_t.detach()),
        }
        return loss

    def calc_loss(self, data_dict: dict):
        out = self.inner_forward(data_dict)
        if self.tdp_stage == 'source':
            return self._source_loss(data_dict, out)
        return self._target_loss(data_dict, out)

    @torch.no_grad()
    def discover_patterns(self, train_loader, device, epoch=0):
        if self.cluster_interval > 1 and epoch % self.cluster_interval != 0:
            return
        was_training = self.training
        self.eval()
        z_list, label_list = [], []
        for batch_dict in train_loader:
            for key, obj in list(batch_dict.items()):
                if torch.is_tensor(obj):
                    batch_dict[key] = obj.to(device)
            out = self.inner_forward(batch_dict)
            z_list.append(out['z_tilde'].detach())
            label_key = 'fg_label' if 'fg_label' in batch_dict else 'label'
            label_list.append(batch_dict[label_key].detach())

        z = torch.cat(z_list, dim=0)
        z_n = F.normalize(z, dim=-1)
        idx, centroids = spherical_kmeans(
            z_n, self.num_patterns, chunk_size=self.cluster_chunk_size
        )
        self.pattern_centers.copy_(centroids)

        labels = torch.cat(label_list, dim=0)
        dom = torch.full((self.num_patterns,), -1, dtype=torch.long, device=device)
        if self.tdp_stage == 'source':
            labels = labels.clamp(0, 5)
            for c in range(self.num_patterns):
                members = labels[idx == c]
                if members.numel() > 0:
                    dom[c] = torch.bincount(members, minlength=6).argmax()
        self.pattern_dom.copy_(dom)
        self.patterns_ready.fill_(True)
        logging.info(f'[TDP] refreshed {centroids.size(0)} pattern centers '
                     f'(stage={self.tdp_stage}, epoch={epoch}).')
        if was_training:
            self.train()

def TDP_FND_config():
    config = HyperParamDict('TDP_FND default hyper-parameters')
    config.add_argument('--model', default='TDP_FND', type=str)
    config.add_argument('--model_type', default='General', type=str)
    config.add_argument('--tdp_stage', default='source', type=str,
                        help='source: fine-grained 6-way; target: binary + p_unk')
    config.add_argument('--tdp_source_loss', default='multiclass', choices=['multiclass', 'binary'], type=str,
                        help='source-stage loss: multiclass=6-way CE on prototypes; binary=CE on bin_logits')
    config.add_argument('--tdp_adapter_rank', default=64, type=int)
    config.add_argument('--tdp_fusion_layers', default=2, type=int)
    config.add_argument('--tdp_fusion_heads', default=8, type=int)
    config.add_argument('--tdp_input_dropout', default=0.0, type=float,
                        help='dropout on the encoder token outputs before the fusion transformer')
    config.add_argument('--tdp_fusion_drop', default=0.0, type=float,
                        help='dropout inside each fusion transformer block (residual + FFN)')
    config.add_argument('--tdp_moe_rank', default=64, type=int)
    config.add_argument('--tdp_expert_type', default='adapter', choices=['adapter', 'transformer'], type=str,
                        help="MoE expert type: 'adapter' = bottleneck adapter (default); "
                             "'transformer' = a timm Block per expert (self-attn + FFN, ~70x params, "
                             'zero-init residual so it starts as identity) -- the parameter-scaling variant.')
    config.add_argument('--tdp_num_patterns', default=64, type=int)
    config.add_argument('--tdp_cluster_interval', default=1, type=int,
                        help='refresh pattern centers every N epochs')
    config.add_argument('--tdp_cluster_chunk_size', default=4096, type=int)
    config.add_argument('--tdp_temperature', default=0.1, type=float)
    config.add_argument('--tdp_text_anchor_mode', default='live', choices=['live', 'frozen'], type=str,
                        help='live: recompute type-text anchors each forward with the current text encoder; '
                             'frozen: compute once from the pretrained encoder and cache')
    config.add_argument('--lambda_pcl', default=0.5, type=float)
    config.add_argument('--lambda_moe', default=0.1, type=float)
    config.add_argument('--eta_moe', default=0.1, type=float)
    config.add_argument('--beta_div', default=0.01, type=float)
    config.add_argument('--use_llm_weak', action='store_true',
                        help='enable confidence-weighted L_weak + L_route^t on LLM-annotated target posts')
    config.add_argument('--eta_w', default=0.1, type=float,
                        help='weight for L_weak (confidence-weighted fine-grained classification)')
    config.add_argument('--llm_conf_threshold', default=0.5, type=float,
                        help='gamma: retain LLM labels with confidence >= gamma')
    config.add_argument('--warmup_epochs', default=0, type=int)
    config.add_argument('--num_expert', default=2, type=int)
    config.add_argument('--beta', default=0.7, type=float)
    return config

if __name__ == '__main__':
    print(TDP_FND_config())
