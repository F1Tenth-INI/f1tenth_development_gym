"""Offline search over LLD settings: best top-5% yaw-rate-change correlation after 3 minutes of sim time.

Emulation of a fresh run at simulator speed 1. Data arrives at ``1 / TIMESTEP_CONTROL`` rows per
second from a recorded stream (two stored memories, oldest first), so after 180 s the LLD has
seen at most 4500 rows. The LLD trains on independent random transitions of the rows that
exist at that moment, at ``--updates-per-second`` gradient updates per sim second
(``HWM_LLD_BATCH_SIZE`` transitions each). After 180 s it is scored on a *different* stored run:
open-loop rollouts with the recorded actions from every second start, using the stream as the
only memory. Score: correlation between predicted and recorded change of the yaw rate over
``H`` control steps, on the 5% of starts with the largest recorded change.

    python TrainingLite/hwm/experiments/lld_search.py --trials 60 --updates-per-second 20 --out /tmp/search_20.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
from TrainingLite.hwm.shared.observation import lld_correction_alone
from torch.distributions import MultivariateNormal

MODELS = Path(__file__).resolve().parents[1] / "models"
STREAM = [("hwm_adaptive_noise_v30", 3000), ("hwm_adaptive_noise_v34", 1500)]
EVAL = "hwm_adaptive_noise_v33"
SIM_SECONDS = 180.0
SPACE = {
    "temperature": [1 / 300, 1 / 150, 1 / 60, 1 / 30, 1 / 10, 1 / 3, 1.0],
    "correction": ["none", "additive", "residual"],
    "hidden": [20, 64],
    "student_t": [True, False],
    "separate": [False, True],
    "super_state": [2, 3, 4],
    "pose_free": [False, True],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch": [32, 64, 128, 256],
    "margin": [None, 25],
    "density": [False],
    "net_only": [False],
    "dummy": [2.0],
    "in_res": [False],
    "in_cov": [False],
}
BASELINE = dict(
    temperature=1 / 150, correction="additive", hidden=20, student_t=True, separate=False,
    super_state=3, pose_free=False, lr=3e-4, batch=32, margin=None, density=False, net_only=False, in_res=False, in_cov=False, dummy=2.0,
)


def apply(cfg: dict) -> None:
    from utilities.Settings import Settings

    Settings.HWM_LLD_ATTENTION_TEMPERATURE = float(cfg["temperature"])
    Settings.HWM_LLD_CORRECTION_NET = cfg["correction"] != "none"
    Settings.HWM_LLD_CORRECTION_RESIDUAL = cfg["correction"] == "residual"
    Settings.HWM_LLD_CORRECTION_HIDDEN = int(cfg["hidden"])
    Settings.HWM_LLD_CORRECTION_DEEP = bool(cfg.get("deep", False))
    Settings.HWM_LLD_COV_HEAD = bool(cfg.get("cov_head", False))
    Settings.HWM_LLD_COV_CENTER_TRANSITIONS = bool(cfg.get("cov_center_transitions", True))
    Settings.HWM_LLD_STANDALONE_ENCODER = bool(cfg.get("standalone", False))
    Settings.HWM_LLD_ALONE_LIKELIHOOD = bool(cfg.get("alone_ll", False))
    Settings.HWM_LLD_CORRECTION_ALONE = bool(cfg.get("alone", getattr(Settings, "HWM_LLD_CORRECTION_ALONE", False)))
    Settings.HWM_LLD_COV_DETACH_TOKENS = bool(cfg.get("cov_detach", True))
    Settings.HWM_LLD_STUDENT_T = bool(cfg["student_t"])
    Settings.HWM_LLD_SEPARATE_ENCODERS = bool(cfg["separate"])
    Settings.HWM_SUPER_STATE_SIZE = int(cfg["super_state"])
    Settings.HWM_LLD_DIRECT_DELTA = bool(cfg.get("direct", cfg["pose_free"]))
    Settings.HWM_LLD_PREDICT_DISPLACEMENT = bool(cfg.get("disp", False))
    Settings.HWM_LLD_KINEMATIC_RESIDUAL = bool(cfg.get("kin", False))
    Settings.HWM_LLD_REFLECT = False
    Settings.HWM_LLD_COSINE_ATTENTION = False
    Settings.HWM_LLD_WHOLE_MEMORY_ATTENTION = True
    Settings.HWM_LLD_ROLLOUT_WINDOW = None
    Settings.HWM_LLD_SKIP_START_ROWS = 10
    Settings.HWM_LLD_BATCH_SIZE = int(cfg["batch"])
    Settings.HWM_LLD_DENSITY_WEIGHTING = bool(cfg.get("density", False))
    Settings.HWM_LLD_REFLECT = bool(cfg.get("reflect", False))
    Settings.HWM_LLD_DUMMY_DISTANCE_INIT = float(cfg.get("dummy", 2.0))
    Settings.HWM_LLD_CORRECTION_INPUT_RESIDUAL = bool(cfg.get("in_res", False))
    Settings.HWM_LLD_CORRECTION_INPUT_COV = bool(cfg.get("in_cov", False))


def load_stream() -> dict:
    rows, run = [], []
    for name, take in STREAM:
        payload = torch.load(MODELS / name / "memory.pt", weights_only=True)
        rows.append(payload["rows"][:take])
        run.append(payload["run_len"].reshape(-1)[:take])
    return {"rows": torch.cat(rows), "run_len": torch.cat(run)}


def load_eval() -> dict:
    payload = torch.load(MODELS / EVAL / "memory.pt", weights_only=True)
    return {"rows": payload["rows"], "run_len": payload["run_len"].reshape(-1)}


def run_trial(cfg: dict, seed: int, updates_per_second: float, stream: dict, evaluation: dict) -> dict:
    from utilities.Settings import Settings

    apply(cfg)
    from TrainingLite.hwm.models.networks import build_models
    from TrainingLite.hwm.shared.memory_manager import MemoryManager
    from TrainingLite.hwm.shared.observation import (
        build_LLD_observation_inference,
        lld_delta,
        retrieve_LLD_neighbors,
    )
    from TrainingLite.hwm.shared.state_indices import ANGULAR_VEL_Z_IDX

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    started = time.time()
    lld = build_models()["LLD"]
    capacity = int(Settings.HWM_MEMORY_MAX_ROWS)
    # The whole stream, every row at the step equal to its index, to build the training observations
    # from; the ring buffer of the real learner is emulated below by hiding what it would have dropped.
    memory = MemoryManager(capacity=len(stream["rows"]), overlap=bool(Settings.HWM_MEMORY_OVERLAP_TRAIN), bank=False)
    memory.attach_projector(copy.deepcopy(lld.key_encoder()), copy.deepcopy(lld.query_encoder()))
    memory.import_rows(stream)
    exported = memory.export_rows()
    assert len(exported["rows"]) == len(stream["rows"]) and int(memory._steps.min()) == 0
    rows, run = exported["rows"], exported["run_len"]
    state_dim, width = int(Settings.HWM_STATE_DIM), int(Settings.HWM_SUPER_STATE_SIZE)
    margin = cfg["margin"]
    skip = int(Settings.HWM_LLD_SKIP_START_ROWS)
    ids = torch.tensor([i for i in range(len(rows) - 1) if run[i + 1] != 1 and run[i] - 1 >= skip])

    # Training observations for every possible sample, built once from the final stream.
    queries, valid, targets = [], [], []
    with torch.no_grad():
        for chunk in ids.split(256):
            past, _ = memory.preceding(chunk, width - 1)
            current, nxt = rows[chunk], rows[chunk + 1]
            obs = build_LLD_observation_inference(
                current[:, :state_dim], current[:, state_dim:], memory, 0,
                past_states=past, before_steps=chunk, future_margin=margin,
            )
            queries.append(obs["current_super_state"])
            valid.append(obs["similar_valid"] > 0)
            targets.append(lld_delta(current[:, :state_dim], nxt[:, :state_dim]))
            bank, diff = obs["similar_superstates"], obs["similar_superstates_diff"]
    queries, valid, targets = torch.cat(queries), torch.cat(valid), torch.cat(targets)
    key_slots = torch.nonzero(memory._match_mask()).squeeze(1)
    key_first = memory._steps[key_slots]
    key_next = key_first + 1  # a key only exists once its next row is stored

    # Training, with the data arriving over simulated time.
    control_rate = 1.0 / float(Settings.TIMESTEP_CONTROL)
    total_steps = int(round(SIM_SECONDS * updates_per_second))
    net_only = bool(cfg.get("net_only"))
    if net_only:
        # No memory: a net maps the superstate to the mean and Cholesky factor of the delta.
        d = int(targets.shape[1])
        net_in = int(queries.shape[1])
        hidden = int(cfg["hidden"] or 20)
        net = torch.nn.Sequential(
            torch.nn.Linear(net_in, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d + d * (d + 1) // 2)
        )
        q_mean, q_std = queries.mean(0), queries.std(0).clamp_min(1e-6)
        t_std = targets.std(0).clamp_min(1e-6)  # the net works in standardised delta units
        tril = torch.tril_indices(d, d)

        def net_dist(query: torch.Tensor) -> MultivariateNormal:
            out = net((query - q_mean) / q_std)
            mean, raw = out[:, :d], out[:, d:]
            scale = torch.zeros(len(query), d, d)
            scale[:, tril[0], tril[1]] = raw
            diag = torch.arange(d)
            scale[:, diag, diag] = torch.nn.functional.softplus(scale[:, diag, diag]) + 1e-3
            return MultivariateNormal(mean, scale_tril=scale)

        parameters = list(net.parameters())
    else:
        parameters = list(lld.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(cfg["lr"]))
    batch = int(cfg["batch"])
    last_loss = float("nan")
    density = None

    def key_density(keys: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        """Kernel mass of every key among the available keys, from the current key encoder."""
        with torch.no_grad():
            emb = lld.encode_key(keys)
            ref = emb[available]
            out = torch.empty(len(emb))
            for chunk in torch.arange(len(emb)).split(1024):
                out[chunk] = torch.exp(-torch.cdist(emb[chunk], ref) / float(cfg["temperature"])).sum(dim=1)
            return out.clamp_min(1.0)

    for step in range(total_steps):
        stored = min(len(rows), int(control_rate * step / updates_per_second) + 1)
        available = int(torch.searchsorted(ids, torch.tensor(stored - 1)))  # samples whose next row is stored
        oldest = max(0, stored - capacity)  # the ring buffer has overwritten everything before
        first = int(torch.searchsorted(ids, torch.tensor(oldest + width)))
        if available - first < 2:
            continue
        pick = first + torch.randint(available - first, (batch,))
        if net_only:
            loss = -net_dist(queries[pick]).log_prob(targets[pick] / t_std).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(Settings.HWM_LLD_GRAD_CLIP))
            optimizer.step()
            last_loss = float(loss.detach())
            continue
        in_memory = (key_next < stored) & (key_first >= oldest)
        mask = valid[pick] & in_memory[None, :]
        batch_obs = {
            "current_super_state": queries[pick],
            "similar_superstates": bank,
            "similar_superstates_diff": diff,
            "similar_valid": mask,
        }
        if cfg.get("density"):
            if density is None or step % 8 == 0:  # as the learner: refreshed now and then, not every step
                density = key_density(bank, torch.nonzero(in_memory).squeeze(1))
            batch_obs["similar_density"] = density
        if lld_correction_alone():
            # As the learner: g matches the delta on its own, the encoder is trained with g held fixed.
            dist = lld(batch_obs, detach_correction=True)
            loss = lld.alone_loss(lld._correct(queries[pick]), dist, targets[pick]) - dist.log_prob(targets[pick]).mean()
        else:
            loss = -lld(batch_obs).log_prob(targets[pick]).mean()
        loss = loss + lld.encoder_nll(targets[pick])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lld.parameters(), float(Settings.HWM_LLD_GRAD_CLIP))
        optimizer.step()
        last_loss = float(loss.detach())
    lld.eval()

    # Score on the other run, open loop with recorded actions, the stream as the only memory.
    # The memory at the end: the last `capacity` rows of the stream.
    memory = MemoryManager(overlap=bool(Settings.HWM_MEMORY_OVERLAP_INFERENCE), bank=False)
    memory.attach_projector(copy.deepcopy(lld.key_encoder()), copy.deepcopy(lld.query_encoder()))
    memory.import_rows(stream)
    ev_rows, ev_run = evaluation["rows"], evaluation["run_len"]
    scores = {}
    with torch.no_grad():
        for horizon in (10, 25):
            starts = [
                i for i in range(width, len(ev_rows) - horizon - 1, 2)
                if ev_run[i] >= width and ev_run[i + horizon] >= horizon + width
            ]
            idx = torch.tensor(starts)
            state = ev_rows[idx, :state_dim].clone()
            start_rate = state[:, ANGULAR_VEL_Z_IDX].clone()
            history = torch.stack([ev_rows[idx - width + 1 + k] for k in range(width - 1)], dim=1)
            bank_eval = None
            for h in range(horizon):
                action = ev_rows[idx + h, state_dim:]
                obs = build_LLD_observation_inference(
                    state, action, memory, 0, past_states=history, neighbors=bank_eval
                )
                if bank_eval is None:
                    bank_eval = {
                        "similar_superstates": obs["similar_superstates"],
                        "similar_superstates_diff": obs["similar_superstates_diff"],
                        "valid": obs["similar_valid"],
                    }
                    if cfg.get("density"):
                        bank_eval["density"] = key_density(obs["similar_superstates"], torch.arange(len(obs["similar_superstates"])))
                if cfg.get("density"):
                    obs["similar_density"] = bank_eval["density"]
                if net_only:
                    mean = net_dist(obs["current_super_state"]).mean * t_std
                else:
                    mean = lld(obs).mean
                nxt = lld.reconstruct_next_state(state, mean)
                history = torch.cat((history, torch.cat((state, action), dim=-1)[:, None]), dim=1)[:, 1:]
                state = nxt
            predicted = (state[:, ANGULAR_VEL_Z_IDX] - start_rate).numpy()
            truth = (ev_rows[idx + horizon, ANGULAR_VEL_Z_IDX] - ev_rows[idx, ANGULAR_VEL_Z_IDX]).numpy()
            order = np.argsort(-np.abs(truth))
            for frac, tag in ((0.05, "top5"), (1.0, "all")):
                top = order[: max(5, int(len(truth) * frac))]
                scores[f"{tag}_corr_h{horizon}"] = float(np.corrcoef(predicted[top], truth[top])[0, 1])
                scores[f"{tag}_size_h{horizon}"] = float(
                    (np.sign(truth[top]) * predicted[top]).mean() / np.abs(truth[top]).mean()
                )
    scores = {k: (0.0 if math.isnan(v) else v) for k, v in scores.items()}
    run_trial.stash = {"lld": lld, "memory": memory, "net_only": net_only}  # for diagnostics
    return {
        "cfg": cfg, "seed": seed, "updates_per_second": updates_per_second, "steps": total_steps,
        "final_loss": last_loss, "seconds": round(time.time() - started, 1), **scores,
    }


_STREAM = None
_EVAL = None


def _job(args):
    cfg, seed, rate = args
    global _STREAM, _EVAL
    torch.set_num_threads(1)
    if _STREAM is None:
        _STREAM, _EVAL = load_stream(), load_eval()
    try:
        return run_trial(cfg, seed, rate, _STREAM, _EVAL)
    except Exception as error:  # a bad combination must not stop the search
        return {"cfg": cfg, "seed": seed, "updates_per_second": rate, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--updates-per-second", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--configs", type=str, default=None, help="jsonl of configs to run instead of a random search")
    args = parser.parse_args()
    rng = random.Random(0)
    if args.configs:
        configs = [json.loads(line)["cfg"] for line in Path(args.configs).read_text().splitlines() if line.strip()]
    else:
        configs = [dict(BASELINE)] + [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(args.trials - 1)]
    jobs = [(cfg, 1000 + s, args.updates_per_second) for cfg in configs for s in range(args.seeds)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool, open(args.out, "a") as out:
        for result in pool.imap_unordered(_job, jobs):
            out.write(json.dumps(result) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
