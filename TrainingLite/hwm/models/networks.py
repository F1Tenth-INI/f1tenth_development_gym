"""Model definitions for the HWM agent.

``build_models`` is the single place that decides which modules exist and how
they are sized. It runs identically on the planner (client) and the learner
(server) so weights can be exchanged by module name. The actor and the dynamics
are placeholders sized from ``utilities/Settings.py``; replace them freely as long
as ``build_models`` keeps returning a ``ModelBundle``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from TrainingLite.hwm.models.bundle import ModelBundle
from TrainingLite.hwm.shared.observation import (
    actor_super_state_dim,
    lld_delta_scale,
    lld_direct_delta,
    lld_dynamic_indices,
    lld_displacement_channels,
    lld_kinematic_residual,
    lld_reflect,
    average_velocity_displacement,
    world_frame_displacement,
    lld_super_flat_dim,
)
from utilities.Settings import Settings
from utilities.state_utilities import (
    ANGULAR_VEL_Z_IDX,
    LINEAR_VEL_X_IDX,
    LINEAR_VEL_Y_IDX,
    POSE_THETA_COS_IDX,
    POSE_THETA_IDX,
    POSE_THETA_SIN_IDX,
    POSE_X_IDX,
    POSE_Y_IDX,
    SLIP_ANGLE_IDX,
    STEERING_ANGLE_IDX,
)


def _mlp(in_dim: int, out_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Current superstate and wall points -> a policy.

    Every wall point ``[x, y, side]`` is encoded by a small MLP into a key, which also
    serves as its value token. The superstate is encoded into a query; scaled
    dot-product softmax attention over the wall tokens gives one context vector, which
    is concatenated with the superstate.

    The policy is a tanh-squashed full-covariance Gaussian over the action.
    ``log_prob`` includes the Jacobian, so Gaussian targets must lie strictly inside
    ``(-1, 1)``. ``deterministic_action`` is ``tanh`` of the Gaussian mean. Placeholder:
    the actor is not trained at the moment.
    """

    def __init__(self, hidden: int, wall_hidden: int = 64, embed_dim: int = 32):
        super().__init__()
        super_dim = actor_super_state_dim()
        action_dim = int(Settings.HWM_ACTION_DIM)
        self.action_dim = action_dim
        self.wall_encoder = nn.Sequential(
            nn.Linear(3, wall_hidden),
            nn.ReLU(),
            nn.Linear(wall_hidden, embed_dim),
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(super_dim, wall_hidden),
            nn.ReLU(),
            nn.Linear(wall_hidden, embed_dim),
        )
        n_tril = action_dim * (action_dim + 1) // 2
        self.head = _mlp(embed_dim + super_dim, action_dim + n_tril, hidden)

    def forward(self, actor_observation: dict[str, torch.Tensor]) -> torch.distributions.TransformedDistribution:
        super_state = actor_observation["current_super_state"]
        tokens = self.wall_encoder(actor_observation["wall_points"])
        query = self.query_encoder(super_state)
        scores = torch.einsum("bpe,be->bp", tokens, query) / math.sqrt(tokens.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bp,bpe->be", weights, tokens)
        out = self.head(torch.cat((context, super_state), dim=-1))

        d = self.action_dim
        mu, sigma = out[:, :d], out[:, d:]
        rows, cols = torch.tril_indices(d, d, device=mu.device)
        L = mu.new_zeros(mu.shape[:-1] + (d, d))
        L[..., rows, cols] = sigma
        raw_diag = torch.diagonal(L, dim1=-2, dim2=-1)
        L = L - torch.diag_embed(raw_diag) + torch.diag_embed(torch.nn.functional.softplus(raw_diag) + 1e-4)
        base = torch.distributions.MultivariateNormal(loc=mu, scale_tril=L)
        return torch.distributions.TransformedDistribution(
            base, [torch.distributions.transforms.TanhTransform(cache_size=1)]
        )

    @staticmethod
    def deterministic_action(dist: torch.distributions.TransformedDistribution) -> torch.Tensor:
        """``tanh`` of the Gaussian mean (the squashed distribution has no closed-form mean)."""
        return torch.tanh(dist.base_dist.mean)


class Dynamics(nn.Module):
    """(state, action) -> predicted state delta (placeholder, deterministic)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = _mlp(state_dim + action_dim, state_dim, hidden)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((state, action), dim=-1))


class MultivariateStudentT(torch.distributions.MultivariateNormal):
    """Multivariate Student-t with location ``loc``, scale matrix ``scale_tril scale_trilᵀ`` and ``df`` ``(B,)``.

    It subclasses ``MultivariateNormal`` so ``mean`` (``loc``, defined for df > 1) and
    ``scale_tril`` keep their meaning for the code that reads them. ``covariance_matrix``
    is the *scale* matrix, not the covariance (that is ``df / (df - 2)`` times it).
    """

    def __init__(self, loc: torch.Tensor, scale_tril: torch.Tensor, df: torch.Tensor):
        super().__init__(loc=loc, scale_tril=scale_tril, validate_args=False)
        self.df = df

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        d = int(self.loc.shape[-1])
        nu = self.df
        diff = (value - self.loc).unsqueeze(-1)
        z = torch.linalg.solve_triangular(self.scale_tril, diff, upper=False).squeeze(-1)
        maha = z.pow(2).sum(dim=-1)
        half_log_det = torch.log(torch.diagonal(self.scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        return (
            torch.lgamma(0.5 * (nu + d))
            - torch.lgamma(0.5 * nu)
            - 0.5 * d * torch.log(nu * math.pi)
            - half_log_det
            - 0.5 * (nu + d) * torch.log1p(maha / nu)
        )

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """``loc + L z / sqrt(u)``, ``z ~ N(0, I)``, ``u ~ χ²(df) / df`` (Gaussian scale mixture)."""
        shape = self._extended_shape(sample_shape)
        z = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        chi2 = torch.distributions.Gamma(0.5 * self.df, torch.full_like(self.df, 0.5)).rsample(sample_shape)
        u = (chi2 / self.df).unsqueeze(-1)
        return self.loc + torch.matmul(self.scale_tril, z.unsqueeze(-1)).squeeze(-1) / torch.sqrt(u)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)


def _attention_encoder(in_dim: int) -> nn.Sequential:
    """Superstate encoder: used for queries and keys alike."""
    return nn.Sequential(
        nn.Linear(in_dim, 15),
        nn.ReLU(),
        nn.Linear(15, int(Settings.HWM_LLD_ENCODER_DIM)),
    )


class LLD(nn.Module):
    """Low-Level dynamics model.

    The next-state delta is an attention-weighted average over every key of the memory bank, with a
    residual net ``g`` of the superstate correcting the mean:
    ``mean = g(q) + sum_j w_j (delta_j - g(k_j))``. A key's weight is ``exp(-distance / temperature)``
    divided by that key's own n_soft (density), then normalised over the keys.
    """

    def __init__(self):
        super().__init__()
        self.attention_temperature = float(Settings.HWM_LLD_ATTENTION_TEMPERATURE)
        if self.attention_temperature <= 0:
            raise ValueError(
                "HWM_LLD_ATTENTION_TEMPERATURE must be positive, "
                f"got {self.attention_temperature}"
            )
        super_state_dim = lld_super_flat_dim()
        # HWM_LLD_STANDALONE_ENCODER: one encoder, trained on its own likelihood (see Settings). It overrides
        # the covariance head.
        self.standalone = bool(getattr(Settings, "HWM_LLD_STANDALONE_ENCODER", False))
        self.encoder_dist: Optional[torch.distributions.MultivariateNormal] = None
        self.attention_norm_projector = _attention_encoder(super_state_dim)
        # HWM_LLD_DIRECT_DELTA: every state channel. Otherwise yaw rate, body velocity, steering, heading.
        prediction_dim = len(lld_dynamic_indices())
        self.prediction_dim = prediction_dim
        # The residual net g: a small network on the flat superstate. The attention average models stored
        # deltas minus g's output at each key. Its last layer starts at zero, so training begins at the plain
        # attention model.
        hidden = int(getattr(Settings, "HWM_LLD_CORRECTION_HIDDEN", 32))
        # The retrieved residual mean and/or the retrieved covariance factor (both detached) are extra
        # inputs of the network at the query. At the keys these inputs are zero, so the residual tokens
        # do not depend on them.
        self.input_residual = bool(getattr(Settings, "HWM_LLD_CORRECTION_INPUT_RESIDUAL", False))
        self.input_cov = bool(getattr(Settings, "HWM_LLD_CORRECTION_INPUT_COV", False))
        self.extra_dim = (prediction_dim if self.input_residual else 0) + (
            prediction_dim * (prediction_dim + 1) // 2 if self.input_cov else 0
        )
        self.register_buffer("_factor_tril", torch.tril_indices(prediction_dim, prediction_dim), persistent=False)
        net_in = super_state_dim + self.extra_dim
        self.correction_net = nn.Sequential(
            nn.Linear(net_in, hidden), nn.ReLU(), nn.Linear(hidden, prediction_dim)
        )
        nn.init.zeros_(self.correction_net[-1].weight)
        nn.init.zeros_(self.correction_net[-1].bias)
        self.last_n_soft = torch.zeros(())
        self.last_top_keys = torch.zeros(())
        self.last_n_soft_norm = torch.zeros(())
        # HWM_LLD_COV_HEAD: the covariance has its own attention, see Settings. The head's own
        # parameters exist only when it is on, so older checkpoints load.
        self.cov_head = (not self.standalone) and bool(getattr(Settings, "HWM_LLD_COV_HEAD", False))
        # HWM_LLD_COV_CENTER_TRANSITIONS: (a) stored transitions about g(q) + mu, or (b) residual tokens about mu.
        self.cov_center_transitions = bool(getattr(Settings, "HWM_LLD_COV_CENTER_TRANSITIONS", True))
        # HWM_LLD_COV_DETACH_TOKENS: delta_j - g(k_j) are constants everywhere.
        self.detach_tokens = bool(getattr(Settings, "HWM_LLD_COV_DETACH_TOKENS", True))
        self.last_cov_n_soft = torch.zeros(())
        self.last_cov_top_keys = torch.zeros(())
        if self.cov_head and self.input_cov:
            raise ValueError("HWM_LLD_COV_HEAD centres on g(q), so HWM_LLD_CORRECTION_INPUT_COV (g's input) would be circular")
        if self.cov_head:
            self.cov_norm_projector = _attention_encoder(super_state_dim)
        self._compile_rollout_step(super_state_dim)

    def _covariance_override(
        self,
        stored_deltas: torch.Tensor,
        prediction: torch.Tensor,
        correction: torch.Tensor,
        chol_scale: torch.Tensor,
        current_super_state: torch.Tensor,
        similar_superstates: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Cholesky factor of the covariance of the covariance head (HWM_LLD_COV_HEAD).

        The weights are this head's own normalised kernels (no density division).
        ``mu = prediction - correction`` is the main head's weighted residual mean. With
        HWM_LLD_COV_CENTER_TRANSITIONS (a) the covariance is the weighted second moment of the stored
        transitions ``delta_j`` (``stored_deltas`` ``(N, d)``) about the
        detached final prediction ``prediction`` ``(B, d)`` = ``g(q) + mu``. Without it (b) the tokens are
        the residuals ``delta_j - g(k_j)`` (``stored_deltas`` then holds those, detached) about ``mu``.
        """
        if int(stored_deltas.shape[0]) == 0:
            return self._scaled_cholesky(
                prediction, prediction.new_zeros(int(prediction.shape[0]), self.prediction_dim, self.prediction_dim), chol_scale
            )[1]
        query = self.cov_norm_projector(current_super_state)
        keys = self.cov_norm_projector(similar_superstates)
        memory_logit = self._memory_logits(query, keys, valid)
        kernel = torch.exp(memory_logit)
        mass = kernel.sum(dim=-1, keepdim=True)
        weights = kernel / mass.clamp_min(1e-12)
        weights = torch.where(mass > 0, weights, torch.zeros_like(weights))
        self.last_cov_n_soft = mass.detach().squeeze(-1).mean()
        self.last_cov_top_keys = self._keys_for_half_weight(weights.detach()).mean()
        center = prediction.detach() if self.cov_center_transitions else (prediction - correction).detach()
        shift = stored_deltas.mean(dim=0)
        x = stored_deltas - shift
        c = center - shift
        first = weights @ x
        d = self.prediction_dim
        second = (weights @ (x.unsqueeze(-1) * x.unsqueeze(-2)).reshape(-1, d * d)).view(-1, d, d)
        total = weights.sum(dim=-1)[:, None, None]
        cov = (
            second
            - first.unsqueeze(-1) * c.unsqueeze(-2)
            - c.unsqueeze(-1) * first.unsqueeze(-2)
            + total * c.unsqueeze(-1) * c.unsqueeze(-2)
        )
        return self._scaled_cholesky(prediction, cov, chol_scale)[1]

    def _compile_rollout_step(self, super_state_dim: int) -> None:
        """Compile the rollout attention step now, so the first rollout does not pay for it.

        Batch size and key count are marked dynamic. The warmup runs one step at the bank size
        from settings.
        """
        try:
            # Dimensions that change between rollouts (batch, key count) are dynamic, so the
            # warmup graph is reused instead of compiling again on the first real call.
            static_by_default = torch._dynamo.config.assume_static_by_default
            torch._dynamo.config.assume_static_by_default = False
            compiled = torch.compile(self._mean_delta_core, dynamic=True)
            emb = int(self.attention_norm_projector[-1].out_features)
            # Two shapes: a full bank, and a short one. Dynamic sizes near either one
            # reuse that graph, so a later rollout does not compile.
            for batch, n_keys in ((32, max(int(Settings.HWM_MEMORY_BANK_SIZE), 1)), (4, 32)):
                query = torch.zeros(batch, int(super_state_dim))
                keys = torch.zeros(n_keys, emb)
                tokens = torch.zeros(n_keys, self.prediction_dim)
                density = torch.ones(n_keys)
                valid = torch.ones(n_keys, dtype=torch.bool)
                for tensor in (query, keys, tokens, density, valid):
                    torch._dynamo.mark_dynamic(tensor, 0)
                compiled(query, keys, tokens, density, valid)
        except Exception as error:
            print(f"[LLD] rollout step stays eager: {error}")
            compiled = self._mean_delta_core
        finally:
            if "static_by_default" in locals():
                torch._dynamo.config.assume_static_by_default = static_by_default
        self.mean_delta_compiled = compiled

    def forward(
        self, LLD_observation: dict[str, torch.Tensor], *, detach_correction: bool = False
    ) -> torch.distributions.MultivariateNormal:
        current_super_state = LLD_observation["current_super_state"]
        similar_superstates = LLD_observation["similar_superstates"]
        similar_superstates_diff = LLD_observation["similar_superstates_diff"]
        query = self.attention_norm_projector(current_super_state)
        batch = int(query.shape[0])
        # One bank of every key (N, D), encoded once and shared by every row.
        keys = self.attention_norm_projector(similar_superstates)
        # L1 penalty on the activations of the encoded vectors (queries and the whole key bank), for the loss.
        l1_weight = float(getattr(Settings, "HWM_LLD_ENCODER_L1", 0.0))
        self.encoder_l1_term = (
            l1_weight * 0.5 * (query.abs().sum(dim=-1).mean() + keys.abs().sum(dim=-1).mean())
            if l1_weight > 0.0 and torch.is_grad_enabled()
            else None
        )
        valid = LLD_observation["similar_valid"] > 0
        memory_logit = self._memory_logits(query, keys, valid)
        density = LLD_observation["similar_density"]
        weights, n_soft, n_soft_norm = self._key_weights(memory_logit, density)
        self.last_n_soft = n_soft.detach().mean()
        self.last_n_soft_norm = n_soft_norm.detach().mean()
        self.last_top_keys = self._keys_for_half_weight(weights).detach().mean()
        df = None
        if bool(getattr(Settings, "HWM_LLD_STUDENT_T", False)):
            # One Student-t from n samples in d dimensions: nu = max(floor, n - d). The scale matrix
            # is the weighted covariance as is: the (1 + 1/n) factor is only for the Gaussian.
            n = n_soft
            if not bool(getattr(Settings, "HWM_LLD_T_GRAD_THROUGH_N", False)):
                n = n.detach()
            floor = getattr(Settings, "HWM_LLD_T_MIN_DOF", None)
            floor = 2.0 if floor is None else float(floor)
            df = torch.clamp(n - float(self.prediction_dim), min=floor)
            chol_scale = n.new_ones(batch)
        else:
            # Covariance x (1 + 1/n) for a mean estimated from n samples.
            chol_scale = torch.sqrt(1.0 + 1.0 / n_soft.clamp_min(1.0))
        stored_deltas = similar_superstates_diff
        if self.standalone:
            # The encoder's own distribution: attention-weighted mean and covariance of the stored deltas (no
            # g, no residuals), Student-t / Gaussian as above. Only training needs it.
            self.encoder_dist = None
            if torch.is_grad_enabled():
                encoder_mu, encoder_factor = self._shared_neighbor_gaussian(stored_deltas, weights, chol_scale)
                if df is not None:
                    self.encoder_dist = MultivariateStudentT(loc=encoder_mu, scale_tril=encoder_factor, df=df)
                else:
                    self.encoder_dist = torch.distributions.MultivariateNormal(loc=encoder_mu, scale_tril=encoder_factor)
            # The rest of the model uses this encoder detached.
            weights = weights.detach()
        # Value tokens become residuals of the network, so the attention average only models what
        # the network misses. The network extrapolates where memory is sparse.
        # delta - g(key) is evaluated here, including at inference: it changes whenever g's weights
        # change and is not stored. detach_correction holds g fixed so only the retrieval encoder
        # (and the rest of the attention) is trained to help g.
        correction = self._correct(similar_superstates)
        if detach_correction or self.detach_tokens:
            correction = correction.detach()
        similar_superstates_diff = similar_superstates_diff - correction
        tokens = similar_superstates_diff
        if self.cov_head:
            # The main head only gives the mean; the covariance comes from the head's own weights, about
            # the detached final prediction g(q) + mu.
            mu = weights @ tokens
        else:
            mu, factor = self._shared_neighbor_gaussian(tokens, weights, chol_scale)
            if self.detach_tokens:
                # No covariance head: the covariance of the retrieved tokens around their weighted mean
                # is not trained at all.
                factor = factor.detach()
        extras = []
        if self.input_residual:
            extras.append(mu.detach())  # attention mean of the residuals
        if self.input_cov:
            extras.append(factor.detach()[:, self._factor_tril[0], self._factor_tril[1]])
        correction_q = self._correct(current_super_state, torch.cat(extras, dim=-1) if extras else None)
        if detach_correction:
            correction_q = correction_q.detach()
        mean = mu + correction_q
        if self.cov_head:
            # (a) stored transitions about g(q) + mu, or (b) the residual tokens (g(k_j) detached) about mu.
            cov_tokens = stored_deltas if self.cov_center_transitions else similar_superstates_diff.detach()
            factor = self._covariance_override(
                cov_tokens, mean, correction_q, chol_scale,
                current_super_state, similar_superstates, valid,
            )
        mu = mean
        if df is not None:
            return MultivariateStudentT(loc=mu, scale_tril=factor, df=df)
        return torch.distributions.MultivariateNormal(loc=mu, scale_tril=factor)

    def encoder_l1(self, like: torch.Tensor) -> torch.Tensor:
        """HWM_LLD_ENCODER_L1 times the mean L1 norm of the encoded vectors of the last forward pass; zero if off."""
        term = getattr(self, "encoder_l1_term", None)
        return like.new_zeros(()) if term is None else term

    def encoder_nll(self, target: torch.Tensor) -> torch.Tensor:
        """HWM_LLD_STANDALONE_ENCODER: negative log-likelihood of ``target`` under the encoder's own
        attention distribution of the last forward pass; zero otherwise."""
        if self.encoder_dist is None:
            return target.new_zeros(())
        return -self.encoder_dist.log_prob(target).mean()

    def alone_loss(
        self, correction: torch.Tensor, prediction: torch.distributions.MultivariateNormal, target: torch.Tensor
    ) -> torch.Tensor:
        """Loss of g(q) on its own under HWM_LLD_CORRECTION_ALONE: squared error, or with
        HWM_LLD_ALONE_LIKELIHOOD the negative log-likelihood of ``target`` with mean ``correction`` and the
        (detached) scale and df of ``prediction``."""
        if not bool(getattr(Settings, "HWM_LLD_ALONE_LIKELIHOOD", False)):
            return (correction - target).pow(2).sum(dim=-1).mean()
        scale = prediction.scale_tril.detach()
        if isinstance(prediction, MultivariateStudentT):
            alone = MultivariateStudentT(loc=correction, scale_tril=scale, df=prediction.df.detach())
        else:
            alone = torch.distributions.MultivariateNormal(loc=correction, scale_tril=scale)
        return -alone.log_prob(target).mean()

    @torch.no_grad()
    def mean_delta(
        self,
        query_super: torch.Tensor,
        key_emb: torch.Tensor,
        tokens: torch.Tensor,
        density: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean delta ``(B, d)`` plus per-row ``n_soft`` and density-normalised ``n_soft``.

        ``query_super`` is ``(B, flat)``. ``key_emb`` ``(N, emb)`` and ``tokens`` ``(N, d)`` are the
        whole bank, computed once for a rollout. ``density`` and ``valid`` are ``(N,)``. This is the
        mean ``forward`` would return when the residual net has no retrieved inputs. An empty bank
        gives ``g(q)``.
        """
        if int(key_emb.shape[0]) == 0:
            correction_q = self._correct(query_super)
            zeros = correction_q.new_zeros(int(query_super.shape[0]))
            return correction_q, zeros, zeros
        return self.mean_delta_compiled(query_super, key_emb, tokens, density, valid)

    def _mean_delta_core(
        self,
        query_super: torch.Tensor,
        key_emb: torch.Tensor,
        tokens: torch.Tensor,
        density: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``mean_delta`` for a non-empty bank. This is what gets compiled at init."""
        correction_q = self._correct(query_super)
        query = self.attention_norm_projector(query_super)
        memory_logit = self._memory_logits(query, key_emb, valid)
        weights, n_soft, n_soft_norm = self._key_weights(memory_logit, density)
        return weights @ tokens + correction_q, n_soft, n_soft_norm

    def residual_mean(self, current_super_state: torch.Tensor) -> torch.Tensor:
        """The residual net alone as the LLD: its delta ``g(q)`` (``(B, d)``), no memory at all.

        Retrieval inputs of the net, if enabled, are zero here.
        """
        return self._correct(current_super_state)

    def _correct(self, x: torch.Tensor, extras: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Correction network on ``x``; the retrieval inputs are zero where ``extras`` is None (keys)."""
        if self.extra_dim:
            if extras is None:
                extras = x.new_zeros(*x.shape[:-1], self.extra_dim)
            x = torch.cat((x, extras), dim=-1)
        return self.correction_net(x)

    def _memory_logits(
        self, query: torch.Tensor, keys: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Attention logits ``(B, N)`` of the memory keys ``(N, D)``.

        Minus L2 distance, divided by the temperature. Invalid keys get the
        dtype minimum so their kernel is zero.
        """
        logits = -torch.cdist(query, keys) / self.attention_temperature
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    @staticmethod
    def _key_weights(
        logits: torch.Tensor, density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalised key weights ``(B, N)``, ``n_soft`` ``(B,)`` and density-normalised ``n_soft``.

        ``logits`` are ``-distance / temperature``, already at the dtype minimum for invalid keys.
        A key's weight is ``exp(logit) / density``, then divided by the sum over the keys.
        ``n_soft`` is the sum of ``exp(logit)`` before that division.
        """
        kernel = torch.exp(logits)
        n_soft = kernel.sum(dim=-1)
        divided = kernel / density.to(kernel).clamp_min(1.0)
        n_soft_norm = divided.sum(dim=-1)
        weights = divided / n_soft_norm.clamp_min(1e-12).unsqueeze(-1)
        weights = torch.where(n_soft_norm.unsqueeze(-1) > 0, weights, torch.zeros_like(weights))
        return weights, n_soft, n_soft_norm

    @staticmethod
    def _keys_for_half_weight(weights: torch.Tensor) -> torch.Tensor:
        """``(B,)`` count of keys inside the top-weight prefix that first reaches mass 0.5."""
        if int(weights.shape[-1]) == 0:
            return weights.new_zeros(weights.shape[0])
        order = weights.argsort(dim=-1, descending=True)
        cumulative = weights.gather(-1, order).cumsum(-1)
        reached = cumulative >= 0.5
        first = reached.float().argmax(dim=-1)
        never = ~reached.any(dim=-1)
        return torch.where(never, torch.zeros_like(first), first + 1).to(weights.dtype)


    def _shared_neighbor_gaussian(
        self, tokens: torch.Tensor, weights: torch.Tensor, chol_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``_neighbor_gaussian`` for one token set ``(N, d)`` shared by all rows.

        Uses the weighted first and second moments, so nothing of size ``(B, N, d)`` is
        built. Tokens are shifted by their plain mean first to keep ``E[xx'] - μμ'``
        from cancelling; the covariance does not depend on the shift.
        """
        d = self.prediction_dim
        if int(tokens.shape[0]) == 0:
            mu = weights.new_zeros(int(weights.shape[0]), d)
            return self._scaled_cholesky(mu, mu.new_zeros(int(weights.shape[0]), d, d), chol_scale)
        shift = tokens.detach().mean(dim=0)
        x = tokens - shift
        mu = weights @ x
        second = (weights @ (x.unsqueeze(-1) * x.unsqueeze(-2)).reshape(-1, d * d)).view(-1, d, d)
        cov = second - mu.unsqueeze(-1) * mu.unsqueeze(-2)
        return self._scaled_cholesky(mu + shift, cov, chol_scale)

    def _scaled_cholesky(
        self, mu: torch.Tensor, cov: torch.Tensor, chol_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cholesky of the symmetrized ``cov``, multiplied by ``chol_scale`` ``(B,)``.

        ``10⁻⁴ I`` keeps the factor defined when one token takes all the weight and the
        sample covariance is zero.
        """
        d = self.prediction_dim
        eye = torch.eye(d, device=mu.device, dtype=mu.dtype)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-4 * eye
        factor, info = torch.linalg.cholesky_ex(cov)
        if bool((info != 0).any()):
            # Rounding made some covariance indefinite (one token holds almost all the weight): more jitter for those rows.
            bad = (info != 0)[:, None, None]
            scale = cov.diagonal(dim1=-2, dim2=-1).abs().amax(dim=-1).clamp_min(1.0)[:, None, None]
            jitter = 1e-4 * scale * eye
            for _ in range(6):
                retry, retry_info = torch.linalg.cholesky_ex(cov + jitter)
                factor = torch.where(bad, retry, factor)
                bad = bad & (retry_info != 0)[:, None, None]
                if not bool(bad.any()):
                    break
                jitter = jitter * 10.0
        return mu, factor * chol_scale[:, None, None]

    @staticmethod
    def _world_velocity(vx: torch.Tensor, vy: torch.Tensor, psi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Body velocity rotated into the world frame."""
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)
        return vx * cos_psi - vy * sin_psi, vx * sin_psi + vy * cos_psi

    def reconstruct_next_state(
        self,
        state: torch.Tensor,
        delta: torch.Tensor,
        dt: float | None = None,
    ) -> torch.Tensor:
        """Full next state from the current state and this model's delta.

        ``state`` is ``(..., state_dim)`` in raw units. ``delta`` is ``(..., prediction_dim)``
        in the same normalized units as the training target, ordered as
        ``lld_dynamic_indices``.

        With ``HWM_LLD_DIRECT_DELTA`` the delta is yaw rate, ``vx``, ``vy`` and steering,
        and heading turns by the mean yaw rate over the step. Otherwise the delta also
        has the heading change. Position is the average world velocity over the step,
        unless ``HWM_LLD_PREDICT_DISPLACEMENT``: then the last two channels are the
        car-frame displacement (forward, left) and position is the current position plus
        that, rotated by the current heading. With ``HWM_LLD_KINEMATIC_RESIDUAL`` those two
        channels are a correction instead, added to the average-velocity position.
        Sin/cos of heading follow from heading, and slip is ``atan(vy / vx)``.
        """
        if dt is None:
            dt = float(Settings.TIMESTEP_CONTROL)
        state_dim = int(Settings.HWM_STATE_DIM)
        if int(state.shape[-1]) != state_dim:
            raise ValueError(f"state last dim must be {state_dim}, got {int(state.shape[-1])}")
        if int(delta.shape[-1]) != self.prediction_dim:
            raise ValueError(
                f"delta last dim must be {self.prediction_dim}, got {int(delta.shape[-1])}"
            )

        # With HWM_LLD_REFLECT the LLD speaks in the mirrored frame of this state; mirror back.
        delta = lld_reflect(delta.to(dtype=state.dtype, device=state.device), state)
        raw_delta = delta * lld_delta_scale(state)
        d_omega, d_vx, d_vy, d_steer = (raw_delta[..., i] for i in range(4))
        cursor = 4
        if lld_direct_delta():
            d_psi = None
        else:
            d_psi = raw_delta[..., cursor]
            cursor += 1
        if lld_displacement_channels():
            d_forward, d_left = raw_delta[..., cursor], raw_delta[..., cursor + 1]
        else:
            d_forward = d_left = None

        vx = state[..., LINEAR_VEL_X_IDX]
        vy = state[..., LINEAR_VEL_Y_IDX]
        psi = state[..., POSE_THETA_IDX]

        omega_n = state[..., ANGULAR_VEL_Z_IDX] + d_omega
        vx_n = vx + d_vx
        vy_n = vy + d_vy
        steer_n = state[..., STEERING_ANGLE_IDX] + d_steer

        if d_psi is None:
            # Heading is not predicted: it turns by the mean yaw rate over the step.
            d_psi = 0.5 * (state[..., ANGULAR_VEL_Z_IDX] + omega_n) * dt
        psi_n = psi + d_psi
        psi_n = torch.atan2(torch.sin(psi_n), torch.cos(psi_n))

        if d_forward is None or lld_kinematic_residual():
            # Mean body velocity over the step, rotated by the heading half way through it.
            # d_psi is the unwrapped turn, so the midpoint heading is exact across ±π.
            wx, wy = average_velocity_displacement(vx, vy, vx_n, vy_n, psi, d_psi, dt)
            x_n = state[..., POSE_X_IDX] + wx
            y_n = state[..., POSE_Y_IDX] + wy
            if d_forward is not None:
                # HWM_LLD_KINEMATIC_RESIDUAL: the model's displacement correction, in the car
                # frame at the heading at the start of the step, on top of that integral.
                cx, cy = world_frame_displacement(d_forward, d_left, psi)
                x_n = x_n + cx
                y_n = y_n + cy
        else:
            # Same heading the target was expressed in: the heading at the start of the step.
            wx, wy = world_frame_displacement(d_forward, d_left, psi)
            x_n = state[..., POSE_X_IDX] + wx
            y_n = state[..., POSE_Y_IDX] + wy

        vx_safe = torch.where(vx_n < 1.0e-3, torch.ones_like(vx_n) * 1.0e-3, vx_n)
        slip_n = torch.atan(vy_n / vx_safe)

        nxt = state.new_zeros(state.shape)
        nxt[..., ANGULAR_VEL_Z_IDX] = omega_n
        nxt[..., LINEAR_VEL_X_IDX] = vx_n
        nxt[..., LINEAR_VEL_Y_IDX] = vy_n
        nxt[..., POSE_THETA_IDX] = psi_n
        nxt[..., POSE_THETA_COS_IDX] = torch.cos(psi_n)
        nxt[..., POSE_THETA_SIN_IDX] = torch.sin(psi_n)
        nxt[..., POSE_X_IDX] = x_n
        nxt[..., POSE_Y_IDX] = y_n
        nxt[..., SLIP_ANGLE_IDX] = slip_n
        nxt[..., STEERING_ANGLE_IDX] = steer_n
        return nxt


def build_models() -> ModelBundle:
    state_dim = int(Settings.HWM_STATE_DIM)
    action_dim = int(Settings.HWM_ACTION_DIM)
    hidden = int(Settings.HWM_HIDDEN_DIM)
    return ModelBundle(
        {
            "LLD": LLD(),
            "actor": Actor(hidden),
            "dynamics": Dynamics(state_dim, action_dim, hidden),
        }
    )
