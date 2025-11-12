# tcp_utilities.py
from __future__ import annotations
import base64, io, json, struct
from typing import Any, Dict
import numpy as np
import torch
import asyncio

# ---- framing: 4-byte big-endian length + JSON (with base64 for bytes) ----
def pack_frame(msg: Dict[str, Any]) -> bytes:
    m = msg.copy()
    # auto-encode top-level data['blob'] (bytes) into base64 for JSON
    if "data" in m and isinstance(m["data"], dict) and isinstance(m["data"].get("blob"), (bytes, bytearray)):
        m["data"]["blob"] = base64.b64encode(m["data"]["blob"]).decode("ascii")
        m["data"]["blob_encoding"] = "base64"
    payload = json.dumps(m, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload

async def read_frame(reader: asyncio.StreamReader) -> Dict[str, Any]:
    hdr = await reader.readexactly(4)
    (n,) = struct.unpack(">I", hdr)
    data = await reader.readexactly(n)
    msg = json.loads(data.decode("utf-8"))
    # auto-decode top-level data['blob'] if base64-encoded
    if "data" in msg and isinstance(msg["data"], dict):
        if msg["data"].get("blob_encoding") == "base64" and "blob" in msg["data"]:
            msg["data"]["blob"] = base64.b64decode(msg["data"]["blob"])
            del msg["data"]["blob_encoding"]
    return msg

# ---- numpy helpers for transitions (future use) ----
def np_to_blob(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data_b64": base64.b64encode(arr.tobytes()).decode("ascii"),
    }

def blob_to_np(obj: Dict[str, Any]) -> np.ndarray:
    buf = base64.b64decode(obj["data_b64"])
    return np.frombuffer(buf, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])

# ---- torch state_dict serialization helpers ----
def state_dict_to_bytes(sd: Dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    cpu_sd = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
    torch.save(cpu_sd, buf)
    return buf.getvalue()

def bytes_to_state_dict(blob: bytes) -> Dict[str, Any]:
    bio = io.BytesIO(blob)
    return torch.load(bio, map_location="cpu", weights_only=True)
