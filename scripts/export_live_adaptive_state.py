"""Export Extreme-RGMT adaptive sampler state from a live debugpy process.

This utility is intentionally separate from the training process.  The target
must already have a debugpy server attached.  It briefly pauses the Python
threads, evaluates the export inside ``OnPolicyRunner.learn``, and resumes the
process after the sidecar has been written.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any


class DapClient:
    def __init__(self, host: str, port: int, timeout_s: float = 30.0):
        self.socket = socket.create_connection((host, port), timeout=timeout_s)
        self.socket.settimeout(timeout_s)
        self.buffer = b""
        self.sequence = 1
        self.pending_responses: dict[int, dict[str, Any]] = {}

    def close(self) -> None:
        self.socket.close()

    def send_request(self, command: str, arguments: dict[str, Any] | None = None) -> int:
        sequence = self.sequence
        self.sequence += 1
        payload = {
            "seq": sequence,
            "type": "request",
            "command": command,
            "arguments": arguments or {},
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.socket.sendall(f"Content-Length: {len(encoded)}\r\n\r\n".encode() + encoded)
        return sequence

    def receive(self) -> dict[str, Any]:
        while b"\r\n\r\n" not in self.buffer:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("debugpy closed the DAP connection")
            self.buffer += chunk
        header, remainder = self.buffer.split(b"\r\n\r\n", 1)
        length = None
        for line in header.decode("ascii").split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1].strip())
                break
        if length is None:
            raise RuntimeError(f"DAP response has no Content-Length: {header!r}")
        while len(remainder) < length:
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("debugpy closed an incomplete DAP response")
            remainder += chunk
        payload = remainder[:length]
        self.buffer = remainder[length:]
        return json.loads(payload)

    def wait_response(self, request_seq: int) -> dict[str, Any]:
        if request_seq in self.pending_responses:
            return self._validate_response(self.pending_responses.pop(request_seq))
        while True:
            message = self.receive()
            if message.get("type") != "response":
                continue
            response_seq = int(message.get("request_seq", -1))
            if response_seq != request_seq:
                self.pending_responses[response_seq] = message
                continue
            return self._validate_response(message)

    @staticmethod
    def _validate_response(message: dict[str, Any]) -> dict[str, Any]:
        if not message.get("success", False):
            raise RuntimeError(
                f"DAP {message.get('command')} failed: {message.get('message')} "
                f"{message.get('body', '')}"
            )
        return message


def export_code(output: Path, source_checkpoint: str, runner_expression: str) -> str:
    path_literal = repr(str(output))
    checkpoint_literal = repr(source_checkpoint)
    return f"""
import hashlib
import time
import torch
_rgmt_runner = {runner_expression}
_rgmt_env = _rgmt_runner.env.unwrapped
_rgmt_ds = _rgmt_env._motion_dataset
_rgmt_counts = _rgmt_ds._bin_counts.detach().cpu()
_rgmt_durations = _rgmt_ds.durations.detach().cpu()
_rgmt_weights = _rgmt_ds.sample_weights.detach().cpu()
_rgmt_groups = None if _rgmt_ds.sampling_groups is None else _rgmt_ds.sampling_groups.detach().cpu()
_rgmt_hash = hashlib.sha256()
for _rgmt_tensor in (_rgmt_counts, _rgmt_durations, _rgmt_weights):
    _rgmt_hash.update(_rgmt_tensor.contiguous().numpy().tobytes())
if _rgmt_groups is not None:
    _rgmt_hash.update(_rgmt_groups.contiguous().numpy().tobytes())
_rgmt_payload = {{
    'format_version': 1,
    'bin_scores': _rgmt_ds._bin_scores.detach().cpu(),
    'bin_counts': _rgmt_counts,
    'bin_offsets': _rgmt_ds._bin_offsets.detach().cpu(),
    'bin_duration_s': float(_rgmt_ds.adaptive_bin_duration_s),
    'ema_alpha': float(_rgmt_ds.adaptive_ema_alpha),
    'score_clip': float(_rgmt_ds.adaptive_score_clip),
    'uniform_ratio': float(_rgmt_ds.adaptive_uniform_ratio),
    'durations': _rgmt_durations,
    'sample_weights': _rgmt_weights,
    'sampling_groups': _rgmt_groups,
    'dataset_fingerprint': _rgmt_hash.hexdigest(),
    'source_checkpoint': {checkpoint_literal},
    'captured_iteration': int(_rgmt_runner.current_learning_iteration),
    'captured_unix_time': time.time(),
    'optimizer_learning_rate': float(_rgmt_runner.alg.optimizer.param_groups[0]['lr']),
    'ppo_learning_rate': float(_rgmt_runner.alg.learning_rate),
}}
torch.save(_rgmt_payload, {path_literal})
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = DapClient(args.host, args.port)
    paused_thread: int | None = None
    try:
        request = client.send_request(
            "initialize",
            {
                "clientID": "codex-rgmt-state-exporter",
                "adapterID": "python",
                "pathFormat": "path",
                "linesStartAt1": True,
                "columnsStartAt1": True,
                "supportsVariableType": True,
                "supportsRunInTerminalRequest": False,
            },
        )
        client.wait_response(request)
        # debugpy deliberately delays the ``attach`` response until the client
        # has completed configuration.  Sending both requests before waiting
        # avoids a protocol deadlock with attach-to-PID sessions.
        attach_request = client.send_request("attach", {"justMyCode": False})
        configuration_request = client.send_request("configurationDone")
        client.wait_response(configuration_request)
        client.wait_response(attach_request)

        request = client.send_request("threads")
        threads = client.wait_response(request)["body"]["threads"]
        ordered_threads = sorted(
            threads,
            key=lambda item: ("main" not in item.get("name", "").lower(), item["id"]),
        )
        frame = None
        runner_expression = None
        for thread in ordered_threads:
            thread_id = int(thread["id"])
            request = client.send_request("pause", {"threadId": thread_id})
            client.wait_response(request)
            paused_thread = thread_id
            time.sleep(0.2)
            request = client.send_request(
                "stackTrace", {"threadId": thread_id, "startFrame": 0, "levels": 100}
            )
            frames = client.wait_response(request).get("body", {}).get("stackFrames", [])
            for candidate in frames:
                source_path = candidate.get("source", {}).get("path", "").replace("\\", "/").lower()
                if candidate.get("name") == "learn" and source_path.endswith("runners/on_policy_runner.py"):
                    frame = candidate
                    runner_expression = "self"
                    break
                if candidate.get("name") == "main" and source_path.endswith("scripts/rsl_rl/train.py"):
                    frame = candidate
                    runner_expression = "runner"
                    break
            if frame is not None:
                break
            request = client.send_request("continue", {"threadId": thread_id, "singleThread": True})
            client.wait_response(request)
            paused_thread = None

        if frame is None or runner_expression is None:
            raise RuntimeError("Could not find OnPolicyRunner.learn or train.main in the live process")

        expression = "exec(" + repr(export_code(args.output, args.source_checkpoint, runner_expression)) + ")"
        request = client.send_request(
            "evaluate",
            {
                "expression": expression,
                "frameId": frame["id"],
                "context": "repl",
            },
        )
        client.wait_response(request)
        if not args.output.is_file():
            raise RuntimeError(f"Target reported success but did not create {args.output}")
        print(args.output)
    finally:
        if paused_thread is not None:
            try:
                request = client.send_request("continue", {"threadId": paused_thread})
                client.wait_response(request)
            except Exception:
                pass
        try:
            request = client.send_request("disconnect", {"terminateDebuggee": False})
            client.wait_response(request)
        except Exception:
            pass
        client.close()


if __name__ == "__main__":
    main()
