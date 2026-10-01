"""Causal residual-cache policies for the FiVE Wan-Edit A experiment.

This module deliberately contains no model-specific code.  The Wan backbone
produces a shallow hidden state and a deep residual; this file owns those
residuals and decides which of the four edit branches must refresh.
"""

from dataclasses import dataclass

import torch


BRANCHES = (
    "target_cond",
    "target_uncond",
    "source_cond",
    "source_uncond",
)


def direction_coefficients(target_scale, source_scale):
    return {
        "target_cond": float(target_scale),
        "target_uncond": 1.0 - float(target_scale),
        "source_cond": -float(source_scale),
        "source_uncond": float(source_scale) - 1.0,
    }


def match_source_noise_skip(timesteps, reference_timesteps, reference_skip):
    """Choose the fewer-step prefix whose t_prev best matches Full."""
    reference_skip = int(reference_skip)
    if reference_skip == 0:
        return 0
    if reference_skip >= len(reference_timesteps) or len(timesteps) < 2:
        raise ValueError("invalid timesteps for skip matching")
    target = reference_timesteps[reference_skip - 1].float()
    # Candidate skip k uses timesteps[k - 1] as source-noise t_prev.
    candidates = timesteps[:-1].float()
    return int(torch.argmin((candidates - target).abs()).item()) + 1


def fixed_matched_reuse_steps(total_steps, warmup_steps=2, reuse_steps=9):
    """Place a fixed number of reuse steps at uniform post-warmup bin centers."""
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    reuse_steps = int(reuse_steps)
    span = total_steps - warmup_steps
    if span <= 0 or reuse_steps <= 0 or reuse_steps > span:
        raise ValueError("invalid fixed matched schedule")
    schedule = tuple(
        warmup_steps + int((index + 0.5) * span / reuse_steps)
        for index in range(reuse_steps)
    )
    if len(set(schedule)) != reuse_steps or any(
            right - left < 2 for left, right in zip(schedule, schedule[1:])):
        raise ValueError("fixed matched schedule violates max_reuse=1")
    return schedule


def _relative_l2(delta, reference, eps=1e-8):
    return _error_metrics(delta, reference, eps)["relative"]


def _rms(value):
    return float(value.float().square().mean().sqrt().item())


def _error_metrics(delta, reference, scale_floor=1e-8):
    absolute_rms = _rms(delta)
    reference_rms = _rms(reference)
    denominator = max(reference_rms, float(scale_floor), 1e-8)
    return {
        "relative": absolute_rms / denominator,
        "absolute_rms": absolute_rms,
        "reference_rms": reference_rms,
        "scale_floor": float(scale_floor),
        "denominator": denominator,
    }


def _quantile(values, fraction, fallback):
    if not values:
        return float(fallback)
    ordered = sorted(float(v) for v in values)
    position = max(0, min(len(ordered) - 1,
                          round(float(fraction) * (len(ordered) - 1))))
    return ordered[position]


@dataclass
class ResidualCacheEntry:
    residual: torch.Tensor
    shallow: torch.Tensor
    refresh_step: int
    reuse_count: int = 0


class ResidualCacheBank:
    """One task-local deep-residual entry for each edit branch."""

    def __init__(self):
        self.task_key = None
        self.entries = {}

    def begin_task(self, task_key):
        if task_key != self.task_key:
            self.task_key = task_key
            self.entries.clear()

    def has(self, branch):
        return branch in self.entries

    def residual(self, branch):
        return self.entries[branch].residual

    def max_reuse(self, branches=BRANCHES):
        return max((self.entries[b].reuse_count for b in branches
                    if b in self.entries), default=0)

    def feature_values(self, shallow_by_branch, coefficients):
        drift = {}
        for branch in BRANCHES:
            current = shallow_by_branch[branch]
            entry = self.entries.get(branch)
            drift[branch] = (0.0 if entry is None else _relative_l2(
                current - entry.shallow, entry.shallow))

        if all(branch in self.entries for branch in BRANCHES):
            current_joint = sum(coefficients[b] * shallow_by_branch[b]
                                for b in BRANCHES)
            cached_joint = sum(coefficients[b] * self.entries[b].shallow
                               for b in BRANCHES)
            weighted_branch_scale = (
                sum(abs(coefficients[b]) * _rms(self.entries[b].shallow)
                    for b in BRANCHES) /
                max(1e-8, sum(abs(coefficients[b]) for b in BRANCHES)))
            joint_scale_floor = 0.1 * weighted_branch_scale
            joint_metrics = _error_metrics(current_joint - cached_joint,
                                           cached_joint,
                                           joint_scale_floor)
            joint_drift = joint_metrics["relative"]
        else:
            joint_drift = 0.0
            joint_metrics = {
                "relative": 0.0,
                "absolute_rms": 0.0,
                "reference_rms": 0.0,
                "scale_floor": 0.0,
                "denominator": 1e-8,
            }

        values = [drift[b] for b in BRANCHES]
        return {
            "branch_drift": drift,
            "joint_drift": joint_drift,
            "joint_drift_metrics": joint_metrics,
            "max_branch_drift": max(values),
            "mean_branch_drift": sum(values) / len(values),
        }

    def refresh(self, branch, shallow, residual, step):
        self.entries[branch] = ResidualCacheEntry(
            residual=residual.detach(),
            shallow=shallow.detach(),
            refresh_step=int(step),
            reuse_count=0,
        )

    def reused(self, branch):
        self.entries[branch].reuse_count += 1


class CausalRidge:
    """Tiny online ridge regressor; observations are past refreshes only."""

    def __init__(self, feature_dim, ridge=1e-3):
        self.feature_dim = int(feature_dim)
        self.xtx = torch.eye(feature_dim, dtype=torch.float64) * float(ridge)
        self.xty = torch.zeros(feature_dim, dtype=torch.float64)
        self.count = 0

    def predict(self, features):
        x = torch.as_tensor(features, dtype=torch.float64)
        if x.numel() != self.feature_dim:
            raise ValueError("feature dimension mismatch")
        if self.count == 0:
            return None
        beta = torch.linalg.solve(self.xtx, self.xty)
        return max(0.0, float(torch.dot(x, beta).item()))

    def observe(self, features, target):
        x = torch.as_tensor(features, dtype=torch.float64)
        y = float(target)
        self.xtx.add_(torch.outer(x, x))
        self.xty.add_(x * y)
        self.count += 1

    def state_dict(self):
        return {
            "feature_dim": self.feature_dim,
            "xtx": self.xtx.tolist(),
            "xty": self.xty.tolist(),
            "count": self.count,
        }

    def load_state_dict(self, state):
        if int(state["feature_dim"]) != self.feature_dim:
            raise ValueError("calibration feature dimension mismatch")
        self.xtx = torch.as_tensor(state["xtx"], dtype=torch.float64)
        self.xty = torch.as_tensor(state["xty"], dtype=torch.float64)
        self.count = int(state["count"])


class CachePolicy:
    """Refresh decisions made before current-step deep outputs are available."""

    MODES = {"full", "full_native", "fewer_steps", "fixed_refresh",
             "fixed_matched", "dev_time_schedule", "independent",
             "aggregate_sync", "source_only", "joint"}

    def __init__(self, mode, threshold=0.08, fixed_interval=2,
                 warmup_steps=2, max_reuse=1,
                 aggregate_sync_threshold_multiplier=1.0,
                 dev_time_reuse_steps=()):
        if mode not in self.MODES:
            raise ValueError(f"unknown cache mode: {mode}")
        if fixed_interval < 1 or max_reuse < 1 or warmup_steps < 1:
            raise ValueError("interval, warmup, and max_reuse must be positive")
        self.mode = mode
        self.threshold = float(threshold)
        self.fixed_interval = int(fixed_interval)
        self.warmup_steps = int(warmup_steps)
        self.max_reuse = int(max_reuse)
        self.aggregate_sync_threshold_multiplier = float(
            aggregate_sync_threshold_multiplier)
        if self.aggregate_sync_threshold_multiplier <= 0:
            raise ValueError(
                "aggregate_sync_threshold_multiplier must be positive")
        self.dev_time_reuse_steps = tuple(
            int(step) for step in dev_time_reuse_steps)
        if len(set(self.dev_time_reuse_steps)) != len(
                self.dev_time_reuse_steps):
            raise ValueError("dev_time_reuse_steps must be unique")
        if tuple(sorted(self.dev_time_reuse_steps)) != (
                self.dev_time_reuse_steps):
            raise ValueError("dev_time_reuse_steps must be sorted")
        self.independent_models = {b: CausalRidge(3) for b in BRANCHES}
        self.joint_model = CausalRidge(8)
        self.branch_thresholds = {b: self.threshold for b in BRANCHES}
        self.joint_threshold = self.threshold
        self.branch_observations = {b: [] for b in BRANCHES}
        self.joint_observations = []

    def calibrate_thresholds(self, target_reuse_fraction=0.5):
        """Match Independent and Joint to the same dev error quantile."""
        for branch in BRANCHES:
            self.branch_thresholds[branch] = _quantile(
                self.branch_observations[branch], target_reuse_fraction,
                self.threshold)
        self.joint_threshold = _quantile(
            self.joint_observations, target_reuse_fraction, self.threshold)

    def state_dict(self):
        self.calibrate_thresholds(target_reuse_fraction=0.5)
        return {
            "version": 1,
            "independent": {
                b: self.independent_models[b].state_dict() for b in BRANCHES
            },
            "joint": self.joint_model.state_dict(),
            "branch_thresholds": self.branch_thresholds,
            "joint_threshold": self.joint_threshold,
            "branch_observations": self.branch_observations,
            "joint_observations": self.joint_observations,
            "threshold_calibration": "matched_dev_error_quantile_0.5",
        }

    def load_state_dict(self, state):
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported calibration state version")
        for branch in BRANCHES:
            self.independent_models[branch].load_state_dict(
                state["independent"][branch])
        self.joint_model.load_state_dict(state["joint"])
        self.branch_thresholds = {
            b: float(state.get("branch_thresholds", {}).get(b, self.threshold))
            for b in BRANCHES
        }
        self.joint_threshold = float(
            state.get("joint_threshold", self.threshold))
        self.branch_observations = {
            b: [float(v) for v in state.get("branch_observations", {}).get(b, [])]
            for b in BRANCHES
        }
        self.joint_observations = [
            float(v) for v in state.get("joint_observations", [])]

    @staticmethod
    def independent_features(step_fraction, drift):
        return [1.0, float(step_fraction), float(drift)]

    @staticmethod
    def joint_features(step_fraction, features):
        return [
            1.0,
            float(step_fraction),
            *[float(features["branch_drift"][b]) for b in BRANCHES],
            float(features["joint_drift"]),
            float(features["max_branch_drift"]),
        ]

    def decide(self, step, total_steps, bank, features):
        fraction = float(step) / max(1, int(total_steps) - 1)
        missing = {b: not bank.has(b) for b in BRANCHES}
        predicted = {b: None for b in BRANCHES}
        predicted_joint = None
        predicted_aggregate_score = None

        if self.mode in {"full", "full_native", "fewer_steps"}:
            decisions = {b: True for b in BRANCHES}
        elif self.mode == "fixed_refresh":
            refresh = (step < self.warmup_steps or any(missing.values()) or
                       step % self.fixed_interval == 0 or
                       bank.max_reuse() >= self.max_reuse)
            decisions = {b: refresh for b in BRANCHES}
        elif self.mode == "fixed_matched":
            planned_reuse = step in fixed_matched_reuse_steps(
                total_steps, self.warmup_steps, reuse_steps=9)
            refresh = (step < self.warmup_steps or any(missing.values()) or
                       bank.max_reuse() >= self.max_reuse or
                       not planned_reuse)
            decisions = {b: refresh for b in BRANCHES}
        elif self.mode == "dev_time_schedule":
            if not self.dev_time_reuse_steps:
                raise ValueError(
                    "dev_time_schedule requires frozen reuse steps")
            if any(reuse_step < self.warmup_steps or
                   reuse_step >= total_steps
                   for reuse_step in self.dev_time_reuse_steps):
                raise ValueError(
                    "dev_time_reuse_steps outside active-step range")
            if any(right - left < 2 for left, right in zip(
                    self.dev_time_reuse_steps,
                    self.dev_time_reuse_steps[1:])):
                raise ValueError(
                    "dev_time_reuse_steps violate max_reuse=1")
            planned_reuse = step in self.dev_time_reuse_steps
            refresh = (step < self.warmup_steps or any(missing.values()) or
                       bank.max_reuse() >= self.max_reuse or
                       not planned_reuse)
            decisions = {b: refresh for b in BRANCHES}
        elif self.mode == "source_only":
            source = (step < self.warmup_steps or
                      missing["source_cond"] or missing["source_uncond"] or
                      step % self.fixed_interval == 0 or
                      bank.max_reuse(("source_cond", "source_uncond")) >=
                      self.max_reuse)
            decisions = {
                "target_cond": True,
                "target_uncond": True,
                "source_cond": source,
                "source_uncond": source,
            }
        elif self.mode == "independent":
            decisions = {}
            for branch in BRANCHES:
                x = self.independent_features(
                    fraction, features["branch_drift"][branch])
                predicted[branch] = self.independent_models[branch].predict(x)
                entry = bank.entries.get(branch)
                decisions[branch] = (
                    step < self.warmup_steps or missing[branch] or
                    (entry is not None and entry.reuse_count >= self.max_reuse) or
                    predicted[branch] is None or
                    predicted[branch] > self.branch_thresholds[branch])
        elif self.mode == "aggregate_sync":
            normalized = {}
            for branch in BRANCHES:
                x = self.independent_features(
                    fraction, features["branch_drift"][branch])
                predicted[branch] = self.independent_models[branch].predict(x)
                threshold = self.branch_thresholds[branch]
                if threshold <= 0:
                    raise ValueError("branch thresholds must be positive")
                normalized[branch] = (
                    None if predicted[branch] is None else
                    predicted[branch] / threshold)
            available = [value for value in normalized.values()
                         if value is not None]
            predicted_aggregate_score = (
                max(available) if len(available) == len(BRANCHES) else None)
            refresh = (
                step < self.warmup_steps or any(missing.values()) or
                bank.max_reuse() >= self.max_reuse or
                predicted_aggregate_score is None or
                predicted_aggregate_score >
                self.aggregate_sync_threshold_multiplier)
            decisions = {b: refresh for b in BRANCHES}
        else:
            x = self.joint_features(fraction, features)
            predicted_joint = self.joint_model.predict(x)
            refresh = (step < self.warmup_steps or any(missing.values()) or
                       bank.max_reuse() >= self.max_reuse or
                       predicted_joint is None or
                       predicted_joint > self.joint_threshold)
            decisions = {b: refresh for b in BRANCHES}

        return decisions, {
            "predicted_branch_error": predicted,
            "predicted_joint_error": predicted_joint,
            "predicted_aggregate_score": predicted_aggregate_score,
            "step_fraction": fraction,
        }

    def observe_refresh(self, step, total_steps, features, exact, shadow,
                        coefficients, update_models=False):
        """Measure post-decision truth and optionally fit development models."""
        fraction = float(step) / max(1, int(total_steps) - 1)
        branch_metrics = {}
        for branch in BRANCHES:
            if branch not in exact or branch not in shadow:
                continue
            metrics = _error_metrics(shadow[branch] - exact[branch],
                                     exact[branch])
            branch_metrics[branch] = metrics
            if update_models:
                self.independent_models[branch].observe(
                    self.independent_features(
                        fraction, features["branch_drift"][branch]),
                    metrics["relative"])
                self.branch_observations[branch].append(metrics["relative"])

        joint_metrics = None
        if all(b in exact and b in shadow for b in BRANCHES):
            exact_direction = sum(coefficients[b] * exact[b] for b in BRANCHES)
            shadow_direction = sum(coefficients[b] * shadow[b]
                                   for b in BRANCHES)
            weighted_branch_scale = (
                sum(abs(coefficients[b]) * _rms(exact[b]) for b in BRANCHES) /
                max(1e-8, sum(abs(coefficients[b]) for b in BRANCHES)))
            joint_metrics = _error_metrics(
                shadow_direction - exact_direction,
                exact_direction,
                scale_floor=0.1 * weighted_branch_scale)
            if update_models:
                self.joint_model.observe(
                    self.joint_features(fraction, features),
                    joint_metrics["relative"])
                self.joint_observations.append(joint_metrics["relative"])
        return branch_metrics, joint_metrics


class ReplayNoise:
    """Production noise stream isolated from schedulers and diagnostics."""

    def __init__(self, seed, device):
        self.seed = int(seed)
        self.device = device
        self.draws = 0
        self.step_ids = []

    def randn_for_step(self, step_id, shape, dtype, device):
        generator = torch.Generator(device=self.device)
        # The large odd stride makes each original scheduler step an explicit,
        # reproducible substream.  Diagnostics cannot advance it.
        step_seed = (self.seed + int(step_id) * 1_000_003) % (2**63 - 1)
        generator.manual_seed(step_seed)
        value = torch.randn(*shape, dtype=dtype, device=device,
                            generator=generator)
        self.draws += 1
        self.step_ids.append(int(step_id))
        return value
