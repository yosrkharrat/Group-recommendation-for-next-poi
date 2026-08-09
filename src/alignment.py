"""
Curvature-aware alignment for POI embeddings.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

EPS = 1e-8
_HYP_EPS = 1e-15


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Möbius addition on the Poincaré ball. x, y: [..., d]; c: scalar or [..., 1]."""
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c * c * x2 * y2
    return num / den.clamp(min=_HYP_EPS)


def geodesic_distance(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Geodesic distance between two points on the Poincaré ball, curvature c."""
    sqrt_c = c.sqrt()
    add = mobius_add(-x, y, c)
    add_norm = add.norm(dim=-1, keepdim=True).clamp(min=_HYP_EPS)
    arg = (sqrt_c * add_norm).clamp(max=1.0 - _HYP_EPS)
    return ((2.0 / sqrt_c) * torch.arctanh(arg)).squeeze(-1)


class ManifoldAwareAdapter(nn.Module):
    """Tangent-space vector -> transformer hidden space, via a geometry-aware MLP stack."""

    def __init__(self,
                 hyperbolic_dim: int,
                 transformer_hidden_dim: int,
                 hidden_dim: int = 1024,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.geom_layers = nn.ModuleList()
        input_dim = hyperbolic_dim
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else transformer_hidden_dim
            layer = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.geom_layers.append(layer)
            input_dim = output_dim

    def forward(self, tangent_emb: torch.Tensor) -> torch.Tensor:
        """tangent_emb: [batch, hyperbolic_dim] (already log-mapped). Returns [batch, transformer_hidden_dim]."""
        out = tangent_emb
        for layer in self.geom_layers:
            out = layer(out)
        return out


class RadiusPredictor(nn.Module):
    """Predicts the original hyperbolic radius (hierarchy depth) from the aligned embedding."""

    def __init__(self, transformer_hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(transformer_hidden_dim // 2, 1),
        )

    def forward(self, aligned_emb: torch.Tensor) -> torch.Tensor:
        return self.net(aligned_emb).squeeze(-1)


class RelationScorer(nn.Module):
    """TransE-style scoring in the aligned space, to keep KG triples geometrically sensible."""

    def __init__(self, num_relations: int, transformer_hidden_dim: int):
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, transformer_hidden_dim)

    def score(self, h: torch.Tensor, r_id: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r = self.rel_emb(r_id)
        return -torch.norm(h + r - t, dim=-1)


def _eigenvalue_loss(K_hyp: torch.Tensor, K_proj: torch.Tensor, top_m: int = 32) -> torch.Tensor:
    """Normalized top-m eigenvalue MSE between two real symmetric [B, B] kernel matrices."""
    evals_h = torch.linalg.eigvalsh(K_hyp)[-top_m:]
    evals_p = torch.linalg.eigvalsh(K_proj)[-top_m:]
    evals_h = evals_h / evals_h.sum().detach().clamp_min(EPS)
    evals_p = evals_p / evals_p.sum().detach().clamp_min(EPS)
    return torch.mean((evals_h - evals_p) ** 2)


class GeometryPreservingLoss(nn.Module):
    """
    Components:
    1. Neighborhood ranking: preserves hyperbolic nearest neighbors, hinge loss with
       aligned-space hard-negative mining.
    2. Radius preservation: RadiusPredictor recovers the hyperbolic norm from the aligned vector.
    3. Triple preservation: TransE margin loss via RelationScorer (only computed when triples
       and num_entities are both passed in).
    4. Spectral (eigenvalue): top-m RBF-kernel eigenvalue matching, opt-in via use_spectral=True.
    """

    def __init__(self,
                 radius_predictor: RadiusPredictor,
                 relation_scorer: Optional[RelationScorer] = None,
                 weight_ranking: float = 1.0,
                 weight_radius: float = 0.2,
                 weight_triple: float = 0.5,
                 margin: float = 0.2,
                 weight_spectral: float = 0.01,
                 top_m_spectral: int = 32):
        super().__init__()
        self.radius_predictor = radius_predictor
        self.relation_scorer = relation_scorer
        self.weight_ranking = weight_ranking
        self.weight_radius = weight_radius
        self.weight_triple = weight_triple
        self.margin = margin
        self.weight_spectral = weight_spectral
        self.top_m_spectral = top_m_spectral

    def forward(self,
                hyp_emb: torch.Tensor,
                aligned_emb: torch.Tensor,
                triples: Optional[torch.Tensor] = None,
                num_entities: Optional[int] = None,
                curvature: float = 1.0,
                use_spectral: bool = False) -> Tuple[torch.Tensor, dict]:
        losses = {}
        batch_size = hyp_emb.shape[0]

        # 1. Neighborhood ranking loss. D_h is computed once and reused by the spectral loss.
        if batch_size > 1:
            with torch.no_grad():
                c_t = torch.tensor([curvature], device=hyp_emb.device)
                D_h = geodesic_distance(hyp_emb.unsqueeze(1), hyp_emb.unsqueeze(0), c_t)
                hyp_dist = D_h.clone()
                hyp_dist.fill_diagonal_(float('inf'))
                positive_idx = hyp_dist.argmin(dim=1)

            anchor = aligned_emb
            positive = aligned_emb[positive_idx]
            pos_dist = torch.norm(anchor - positive, dim=-1)

            with torch.no_grad():
                aligned_dist = torch.cdist(aligned_emb, aligned_emb, p=2)
                exclude = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=hyp_emb.device)
                exclude.scatter_(1, positive_idx.unsqueeze(1), True)
                exclude.fill_diagonal_(True)
                aligned_dist_masked = aligned_dist.masked_fill(exclude, float('inf'))
                neg_idx = aligned_dist_masked.argmin(dim=1)

            negative = aligned_emb[neg_idx]
            neg_dist = torch.norm(anchor - negative, dim=-1)
            ranking_loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        else:
            ranking_loss = torch.tensor(0.0, device=hyp_emb.device)
        losses['ranking'] = ranking_loss

        # 2. Radius (hierarchy) preservation loss.
        with torch.no_grad():
            hyp_radius = torch.norm(hyp_emb, dim=-1)
        pred_radius = self.radius_predictor(aligned_emb)
        hyp_radius_norm = hyp_radius / hyp_radius.mean().detach().clamp_min(1e-5)
        pred_radius_norm = pred_radius / pred_radius.mean().detach().clamp_min(1e-5)
        radius_loss = F.smooth_l1_loss(pred_radius_norm, hyp_radius_norm)
        losses['radius'] = radius_loss

        # 3. Triple (relation) preservation loss.
        triple_loss = torch.tensor(0.0, device=hyp_emb.device)
        if self.relation_scorer is not None and triples is not None and num_entities is not None:
            h_id, r_id, t_id = triples[:, 0], triples[:, 1], triples[:, 2]
            try:
                h = aligned_emb[h_id]
                t = aligned_emb[t_id]
                pos_score = self.relation_scorer.score(h, r_id, t)
                neg_t_id = torch.randint(0, batch_size, (len(t_id),), device=hyp_emb.device)
                neg_t = aligned_emb[neg_t_id]
                neg_score = self.relation_scorer.score(h, r_id, neg_t)
                triple_loss = F.relu(1.0 - pos_score + neg_score).mean()
            except IndexError:
                pass
        losses['triple'] = triple_loss

        # 4. Spectral eigenvalue loss (opt-in). Shares sigma between both kernels for a fair
        # comparison; K_h is constant, K_p is differentiable through aligned_emb.
        spectral_loss = torch.tensor(0.0, device=hyp_emb.device)
        if use_spectral and self.weight_spectral > 0.0 and batch_size > 1:
            with torch.no_grad():
                nonzero = D_h[D_h > 0]
                sigma = nonzero.median().clamp_min(EPS) if nonzero.numel() > 0 else D_h.new_tensor(1.0)
                K_h = torch.exp(-(D_h ** 2) / (sigma ** 2))
            D_p = torch.cdist(aligned_emb, aligned_emb, p=2)
            K_p = torch.exp(-(D_p ** 2) / (sigma.detach() ** 2))
            spectral_loss = _eigenvalue_loss(K_h, K_p, top_m=self.top_m_spectral)
        losses['spectral'] = spectral_loss

        total_loss = (self.weight_ranking * ranking_loss +
                      self.weight_radius * radius_loss +
                      self.weight_triple * triple_loss +
                      self.weight_spectral * spectral_loss)

        return total_loss, losses


def match_llm_embedding_distribution(projected_emb: torch.Tensor,
                                      llm_embedding_weight: torch.Tensor) -> torch.Tensor:
    """Shift/rescale projected_emb to match the target LLM's native embedding mean/std, per-dim."""
    target_mean = llm_embedding_weight.mean(dim=0, keepdim=True)
    target_std = llm_embedding_weight.std(dim=0, keepdim=True).clamp_min(1e-5)
    proj_mean = projected_emb.mean(dim=0, keepdim=True)
    proj_std = projected_emb.std(dim=0, keepdim=True).clamp_min(1e-5)
    projected_normed = (projected_emb - proj_mean) / proj_std
    return projected_normed * target_std + target_mean
