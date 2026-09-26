"""Message schemas for planner <-> learner TCP traffic.

Framing (4-byte length + JSON, base64 blobs) is reused from
``TrainingLite/rl_racing/tcp_utilities.py``. This module only defines payloads.

client -> server
    raw_batch    {episode_id, episode_end, state: blob(N, S), action: blob(N, A)}
    clear_buffer {}
    terminate    {}

server -> client
    ack            {msg}
    handshake      {model_name, map_name}
    weights        {blob: torch.save({module_name: state_dict}), modules: [...]}
    training_info  {...}          # free-form telemetry
    terminate      {reason}
    terminate_ack  {msg}
    clear_buffer_ack {}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from TrainingLite.rl_racing.tcp_utilities import (
    blob_to_np,
    bytes_to_state_dict,
    np_to_blob,
    state_dict_to_bytes,
)

MSG_RAW_BATCH = "raw_batch"
MSG_CLEAR_BUFFER = "clear_buffer"
MSG_CLEAR_BUFFER_ACK = "clear_buffer_ack"
MSG_TERMINATE = "terminate"
MSG_TERMINATE_ACK = "terminate_ack"
MSG_ACK = "ack"
MSG_HANDSHAKE = "handshake"
MSG_WEIGHTS = "weights"
MSG_TRAINING_INFO = "training_info"
MSG_BATCH_ACK = "batch_ack"


def pack_raw_batch(states: np.ndarray, actions: np.ndarray, episode_id: int, episode_end: bool) -> dict:
    return {
        "type": MSG_RAW_BATCH,
        "data": {
            "episode_id": int(episode_id),
            "episode_end": bool(episode_end),
            "state": np_to_blob(np.ascontiguousarray(states, dtype=np.float32)),
            "action": np_to_blob(np.ascontiguousarray(actions, dtype=np.float32)),
        },
    }


def unpack_raw_batch(data: dict) -> tuple[np.ndarray, np.ndarray, int, bool]:
    states = blob_to_np(data["state"])
    actions = blob_to_np(data["action"])
    return states, actions, int(data.get("episode_id", 0)), bool(data.get("episode_end", False))


def pack_weights(state_dicts: dict[str, dict[str, Any]]) -> dict:
    return {
        "type": MSG_WEIGHTS,
        "data": {
            "blob": state_dict_to_bytes(state_dicts),
            "format": "torch_state_dicts",
            "modules": list(state_dicts.keys()),
        },
    }


def unpack_weights(data: dict) -> dict[str, dict[str, Any]]:
    return bytes_to_state_dict(data["blob"])


def pack_handshake(model_name: str, map_name: str) -> dict:
    return {
        "type": MSG_HANDSHAKE,
        "data": {"model_name": str(model_name), "map_name": str(map_name)},
    }


def pack_simple(msg_type: str, **data: Any) -> dict:
    return {"type": msg_type, "data": dict(data)}
