"""Local recurrent policy with fixed, signed, sparse connectome propagation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .measurement_identity import MeasurementIdentityPointer, measurement_request
from .source_request import SourceRequestPointer
from .tokenizer import CharacterTokenizer
from .tool_state import (
    CONTROL_LABELS,
    OPTION_WIDTH,
    STATE_WIDTH,
    TASK_STATE_WIDTH,
    control_features,
    option_features,
    state_features,
    task_state_features,
)

ACTION_KINDS = ("CLICK", "TYPE", "SELECT", "HOVER", "DRAG", "SCROLL", "KEYPRESS", "WAIT", "STOP")


def as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def legal_action_mask(observation: Any, device=None) -> torch.Tensor:
    permitted = {"WAIT", "STOP"}
    for control in as_dict(observation).get("controls", []):
        control = as_dict(control)
        if control.get("enabled", True):
            permitted.update(control.get("actions", ["CLICK"]))
    return torch.tensor([kind in permitted for kind in ACTION_KINDS], dtype=torch.bool, device=device)


def legal_target_mask(observation: Any, kind: str, device=None, *, width=None) -> torch.Tensor:
    controls = as_dict(observation).get("controls", [])
    width = max(1, len(controls)) if width is None else width
    if width < len(controls):
        raise ValueError("Target mask cannot truncate visible controls")
    mask = torch.zeros(width, dtype=torch.bool, device=device)
    if kind not in ("WAIT", "STOP"):
        for index, control in enumerate(controls):
            control = as_dict(control)
            mask[index] = control.get("enabled", True) and kind in control.get("actions", ["CLICK"])
    return mask


def observation_text(observation: Any) -> str:
    return json.dumps(visible_observation(observation), ensure_ascii=False, default=str)


def visible_observation(observation: Any) -> dict:
    data = as_dict(observation)
    # The target head gets controls independently; also expose all option text to the core.
    visible = {
        key: data.get(key, "")
        for key in ("instruction", "values", "feedback", "chart", "progress", "controls")
    }
    visible["controls"] = [
        {key: value for key, value in as_dict(control).items() if key != "id"}
        for control in data.get("controls", [])
    ]
    if data.get("spreadsheet"):
        visible["spreadsheet"] = data["spreadsheet"]
    if data.get("calculation"):
        visible["calculation"] = data["calculation"]
    return visible


def control_text(control: Any) -> str:
    data = as_dict(control)
    keys = ("role", "label", "value", "options")
    text = " ".join(str(data.get(key, "")) for key in keys)
    return text + (" " + data["surface"] if data.get("surface") in ("spreadsheet", "calculation") else "")


def structured_sections(state, *, prefix="calculation", max_characters=899):
    """Bound every structured reference/result independently, without dropping fields."""
    parts = []

    def visit(path, value):
        serialized = f"{path}: {json.dumps(value, sort_keys=True)}"
        if len(serialized) <= max_characters:
            parts.append(serialized)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}.{key}", child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{path}.{index}", child)
        else:
            raise ValueError("Structured observation exceeds a section token budget")

    for key, value in state.items():
        visit(f"{prefix}.{key}", value)
    return parts


def calculation_sections(state):
    return structured_sections(state)


def graph_fingerprint(graph: Any) -> str:
    digest = hashlib.sha256()
    for key in (
        "body_ids",
        "edge_src",
        "edge_dst",
        "edge_weight",
        "edge_synapse_count",
        "node_features",
        "sensory_mask",
        "intrinsic_mask",
        "efferent_mask",
    ):
        array = np.ascontiguousarray(getattr(graph, key))
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    digest.update(json.dumps(graph.metadata, sort_keys=True, default=str).encode())
    return digest.hexdigest()


@dataclass
class PolicyOutput:
    action_logits: torch.Tensor
    target_logits: torch.Tensor
    typed_value_logits: torch.Tensor
    answer_logits: torch.Tensor
    value: torch.Tensor
    pointer: torch.Tensor
    state: torch.Tensor
    pooled: torch.Tensor


class ChartEncoder(nn.Module):
    """Small randomly initialized encoder, trained only from local chart examples."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256, hidden_size),
        )

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 4 or pixels.shape[1] not in (1, 3):
            raise ValueError("Chart pixels must be [batch, 1 or 3, height, width]")
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("Chart pixels must be finite and scaled to [0, 1]")
        return self.layers(pixels.mean(dim=1, keepdim=True))


class ConnectomePolicy(nn.Module):
    def __init__(
        self,
        graph: Any,
        hidden_size: int = 32,
        propagation_steps: int = 3,
        max_answer_length: int = 20,
        tokenizer: CharacterTokenizer | None = None,
        observation_encoding: str = "pooled_text_v2",
        selection_mode: str = "characters",
        control_encoding: str = "characters",
    ):
        super().__init__()
        if hidden_size < 4 or propagation_steps < 1 or max_answer_length < 2:
            raise ValueError("Invalid model dimensions")
        self.graph = graph
        self.hidden_size = hidden_size
        self.propagation_steps = propagation_steps
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer or CharacterTokenizer()
        if observation_encoding not in (
            "pooled_text_v2",
            "structured_tool_v3",
            "structured_tool_v4",
            "structured_tool_v5",
            "structured_tool_v6",
        ):
            raise ValueError("Unknown observation encoding")
        if selection_mode not in (
            "characters",
            "option_pointer_v1",
            "option_pointer_semantic_v2",
            "measurement_identity_v1",
            "measurement_source_v2",
            "measurement_result_v3",
        ):
            raise ValueError("Unknown selection mode")
        self.observation_encoding = observation_encoding
        self.selection_mode = selection_mode
        if control_encoding not in ("characters", "semantic_tool_v1"):
            raise ValueError("Unknown control encoding")
        self.control_encoding = control_encoding
        self.graph_hash = graph_fingerprint(graph)
        self.action_temperature = 1.0
        self.target_temperature = 1.0
        self.calibration: dict = {"status": "uncalibrated"}
        self.vision_trained = False
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, hidden_size, padding_idx=0)
        self.text_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.sensory_projection = nn.Linear(hidden_size, hidden_size)
        self.feature_projection = nn.Linear(5, hidden_size, bias=False)
        self.cell = nn.GRUCell(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, len(ACTION_KINDS))
        self.target_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_head = nn.Linear(hidden_size, 1)
        self.pointer_head = nn.Linear(hidden_size, 4)
        self.value_decoder = nn.GRUCell(hidden_size, hidden_size)
        self.answer_decoder = nn.GRUCell(hidden_size, hidden_size)
        self.value_chars = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        self.answer_chars = nn.Linear(hidden_size, self.tokenizer.vocab_size)
        self.chart_encoder = ChartEncoder(hidden_size)
        if control_encoding == "semantic_tool_v1":
            self.control_projection = nn.Linear(len(CONTROL_LABELS), hidden_size, bias=False)
        if observation_encoding == "structured_tool_v4":
            self.tool_state_projection = nn.Linear(STATE_WIDTH, hidden_size)
        if observation_encoding in {"structured_tool_v5", "structured_tool_v6"}:
            self.tool_state_projection = nn.Linear(TASK_STATE_WIDTH, hidden_size)
        if selection_mode == "measurement_result_v3":
            self.option_projection = nn.Linear(OPTION_WIDTH, hidden_size, bias=False)
            self.option_query = nn.Linear(hidden_size, hidden_size, bias=False)
        # Conditional construction keeps strict loading of old checkpoints unchanged.
        if selection_mode in ("measurement_identity_v1", "measurement_source_v2", "measurement_result_v3"):
            self.measurement_identity = MeasurementIdentityPointer(self.tokenizer, hidden_size)
            if selection_mode in ("measurement_source_v2", "measurement_result_v3"):
                self.measurement_identity.source_request = SourceRequestPointer(self.tokenizer, hidden_size)
        n = len(graph.body_ids)
        if not n or np.asarray(graph.node_features).shape != (n, 5):
            raise ValueError("Graph must have nodes and five biological node features")
        # Biological tensors are deliberately absent from learned checkpoints.
        self.register_buffer(
            "features", torch.tensor(np.asarray(graph.node_features), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "sensory", torch.tensor(np.asarray(graph.sensory_mask), dtype=torch.bool), persistent=False
        )
        self.register_buffer(
            "efferent", torch.tensor(np.asarray(graph.efferent_mask), dtype=torch.bool), persistent=False
        )
        if not self.sensory.any() or not self.efferent.any():
            raise ValueError("Graph must contain sensory inputs and efferent readout nodes")
        indices = torch.tensor(np.stack([graph.edge_dst, graph.edge_src]), dtype=torch.long)
        weights = torch.tensor(np.asarray(graph.edge_weight), dtype=torch.float32)
        if not torch.isfinite(weights).all():
            raise ValueError("Graph weights must be finite")
        self.register_buffer(
            "adjacency",
            torch.sparse_coo_tensor(indices, weights, (n, n), check_invariants=True).coalesce(),
            persistent=False,
        )

    @property
    def device(self):
        return self.embedding.weight.device

    def configuration(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "propagation_steps": self.propagation_steps,
            "max_answer_length": self.max_answer_length,
            "architecture": type(self).__name__,
            "observation_encoding": self.observation_encoding,
            "selection_mode": self.selection_mode,
            "control_encoding": self.control_encoding,
        }

    def tool_features(self, observation: Any) -> torch.Tensor:
        """Visible state occupancy, not expert stages, answers, or next-action rules.

        Keep field identity rather than averaging mostly static reference prose.
        This supplements character input at sensory neurons, never at the readout.
        """
        obs = as_dict(observation)
        state, values = obs.get("calculation") or {}, obs.get("values") or {}
        results = {k: v for k, v in state.get("results", {}).items() if v.get("valid")}
        bindings = state.get("bindings") or {}
        last = state.get("last_operation") or {}
        features = (
            [
                bool(state.get("operation")),
                bool(state.get("parameter")),
                bool(state.get("source")),
                bool(bindings),
                bool(results),
                state.get("pending_result") in results,
                state.get("selected_result") in results,
                bool(state.get("destination")),
                bool(values.get("answers")),
                bool(values.get("units")),
                bool(state.get("tool_error")),
                last.get("kind") == "calculate",
                last.get("kind") == "copy",
                min(len(bindings), 16) / 16,
                min(len(results), 16) / 16,
                min(len(values.get("required_fields", [])), 16) / 16,
            ]
            if state
            else [0.0] * 16
        )
        vector = self.embedding.weight.new_tensor(features)
        if self.hidden_size >= 16:
            return torch.nn.functional.pad(vector, (0, self.hidden_size - 16))
        # Small synthetic test models: fixed folding, no learned neuron embeddings.
        folded = vector.new_zeros(self.hidden_size)
        return folded.scatter_add(0, torch.arange(16, device=self.device) % self.hidden_size, vector)

    def option_scores(self, observation: Any, control: Any, pooled: torch.Tensor) -> torch.Tensor:
        """Rank actual visible options using their visible metadata, not generated IDs."""
        obs, target = as_dict(observation), as_dict(control)
        if self.selection_mode in (
            "measurement_identity_v1",
            "measurement_source_v2",
            "measurement_result_v3",
        ):
            request = measurement_request(
                obs, target, include_results=self.selection_mode == "measurement_result_v3"
            )
            if request is not None:
                return self.measurement_identity([request], pooled)[0]
        state = obs.get("calculation") or {}
        sources = {**(obs.get("values") or {}).get("measurements", {}), **state.get("results", {})}
        texts = []
        for option in target.get("options", []):
            metadata = sources.get(option)
            if metadata is not None and self.selection_mode in (
                "option_pointer_semantic_v2",
                "measurement_result_v3",
            ):
                # Stellar source selection asks WHICH quantity/reference to use.
                # Magnitudes remain visible to the core/tool, but are not source identity.
                metadata = {k: v for k, v in metadata.items() if k in ("kind", "unit", "source", "valid")}
            # Opaque IDs stay selectable, but their meaning comes from visible data.
            texts.append(json.dumps(metadata, sort_keys=True) if metadata is not None else str(option))
        if not texts or any(len(text) + 2 > self.tokenizer.max_length for text in texts):
            raise ValueError("Option text is empty or exceeds a section token budget")
        keys = self.encode_text(texts)
        if self.selection_mode == "measurement_result_v3":
            features = keys.new_tensor(
                [
                    option_features(obs, option, years=self.observation_encoding == "structured_tool_v6")
                    for option in target["options"]
                ]
            )
            keys = keys + self.option_projection(features)
        # The option list's order and opaque IDs must not change the query.
        context = self.encode_text(
            [f"{target.get('role', '')} {target.get('label', '')} {target.get('surface', '')}"]
        )
        query = (
            self.value_decoder(context, pooled)
            if self.selection_mode == "option_pointer_semantic_v2"
            else self.target_query(pooled + context)
        )
        if self.selection_mode == "measurement_result_v3":
            query = self.option_query(pooled + context)
        return (keys * query).sum(-1) / self.hidden_size**0.5

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens, lengths = self.tokenizer.batch(texts, self.device)
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        sequence, state = self.text_encoder(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True)
        mask = torch.arange(sequence.shape[1], device=self.device)[None, :] < lengths.to(self.device)[:, None]
        # Mean contextual states retain early characters/instructions even when
        # later controls are long; the final state retains terminal ordering.
        pooled = (sequence * mask[:, :, None]).sum(1) / lengths.to(self.device)[:, None]
        return (pooled + state[-1]) / 2

    def encode_controls(self, controls: Sequence[Any], *, target=False) -> torch.Tensor:
        """One vector per control; never truncate the list or an individual field."""
        groups = []
        for control in controls:
            data = {k: v for k, v in as_dict(control).items() if k != "id"}
            if target and self.selection_mode == "measurement_result_v3":
                data = {k: v for k, v in data.items() if k in ("role", "label", "surface")}
                control = data
            text = control_text(control) if target else f"control: {json.dumps(data, sort_keys=True)}"
            budget = self.tokenizer.max_length - 2
            groups.append(
                [text]
                if len(text) <= budget
                else structured_sections(data, prefix="control", max_characters=budget)
            )
        encoded = self.encode_text([text for group in groups for text in group])
        return torch.stack([part.mean(0) for part in encoded.split([len(group) for group in groups])])

    def decode_characters(
        self,
        pooled: torch.Tensor,
        decoder: nn.GRUCell,
        head: nn.Linear,
        teacher_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = pooled
        current = torch.full((len(pooled),), self.tokenizer.BOS, device=self.device, dtype=torch.long)
        logits = []
        for position in range(self.max_answer_length):
            hidden = decoder(self.embedding(current), hidden)
            step = head(hidden)
            logits.append(step)
            current = teacher_tokens[:, position] if teacher_tokens is not None else step.argmax(-1)
        return torch.stack(logits, dim=1)

    def propagate(
        self, encoded: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n = len(encoded), len(self.graph.body_ids)
        if state is None:
            state = encoded.new_zeros((batch, n, self.hidden_size))
        if state.shape != (batch, n, self.hidden_size):
            raise ValueError("Recurrent state does not match this graph and batch")
        injected = self.sensory_projection(encoded)[:, None, :] * self.sensory[None, :, None]
        biological = self.feature_projection(self.features)[None, :, :]
        for _ in range(self.propagation_steps):
            flat = state.permute(1, 0, 2).reshape(n, batch * self.hidden_size)
            message = (
                torch.sparse.mm(self.adjacency, flat).reshape(n, batch, self.hidden_size).permute(1, 0, 2)
            )
            update = self.norm(message + injected + biological)
            state = self.cell(
                update.reshape(-1, self.hidden_size), state.reshape(-1, self.hidden_size)
            ).reshape(batch, n, self.hidden_size)
        return state, state[:, self.efferent, :].mean(dim=1)

    def forward(
        self,
        observations: Sequence[Any],
        state: torch.Tensor | None = None,
        chart_pixels: torch.Tensor | None = None,
        teacher_values: torch.Tensor | None = None,
        teacher_answers: torch.Tensor | None = None,
    ) -> PolicyOutput:
        if not observations:
            raise ValueError("Observation batch cannot be empty")
        fields = ("instruction", "values", "feedback", "chart", "progress")
        visible = [visible_observation(o) for o in observations]
        sections = [
            f"{key}: {json.dumps(o.get(key, ''), ensure_ascii=False, default=str)}"
            for o in visible
            for key in fields
        ]
        # Separate budgets prevent a long chart or control list from dropping
        # instructions, progress, or field values at a global string boundary.
        encoded_fields = self.encode_text(sections).reshape(len(observations), len(fields), self.hidden_size)
        control_vectors = torch.stack(
            [
                self.encode_controls(o["controls"]).mean(0)
                if o["controls"]
                else self.encode_text(["controls: []"])[0]
                for o in visible
            ]
        )
        # Keep the controls' one-sixth share of the base observation, independent
        # of control count/order, while allocating every control its own budget.
        encoded = (encoded_fields.sum(1) + control_vectors) / (len(fields) + 1)
        # Each measurement/result gets its own token budget. Do not hide the
        # sheet state past a single 1024-character serialized-observation limit.
        for index, observation in enumerate(visible):
            sheet = observation.get("spreadsheet")
            calculation = observation.get("calculation")
            if sheet or calculation:
                parts = (
                    calculation_sections(calculation)
                    if calculation
                    else [
                        f"spreadsheet {key}: {json.dumps(value, sort_keys=True)}"
                        for key, value in sheet.items()
                    ]
                )
                parts += [
                    f"measurement {key}: {json.dumps(value, sort_keys=True)}"
                    for key, value in observation["values"].get("measurements", {}).items()
                ]
                if any(len(self.tokenizer.encode(p)) >= self.tokenizer.max_length for p in parts):
                    raise ValueError("Tool observation exceeds a section token budget")
                # Stack instead of in-place mutation of encoder outputs.
                addition = self.encode_text(parts).mean(0)
                encoded = torch.stack(
                    [(row + addition) / 2 if i == index else row for i, row in enumerate(encoded)]
                )
        if chart_pixels is None and any(as_dict(o).get("chart_crop") for o in observations):
            from PIL import Image

            crops = []
            present = []
            for observation in observations:
                path = as_dict(observation).get("chart_crop")
                if path:
                    with Image.open(path) as picture:
                        pixels = np.asarray(picture.convert("L").resize((64, 64)), dtype=np.float32) / 255.0
                    crops.append(torch.tensor(pixels)[None, :, :])
                    present.append(True)
                else:
                    crops.append(torch.zeros((1, 64, 64)))
                    present.append(False)
            chart_pixels = torch.stack(crops).to(self.device)
            mask = torch.tensor(present, device=self.device)[:, None]
            encoded = encoded + self.chart_encoder(chart_pixels) * mask
            chart_pixels = None
        if chart_pixels is not None:
            encoded = encoded + self.chart_encoder(chart_pixels.to(self.device))
        if self.observation_encoding == "structured_tool_v3":
            encoded = encoded + torch.stack([self.tool_features(o) for o in observations])
        if self.observation_encoding == "structured_tool_v4":
            features = encoded.new_tensor([state_features(as_dict(o)) for o in observations])
            encoded = encoded + self.tool_state_projection(features)
        if self.observation_encoding in {"structured_tool_v5", "structured_tool_v6"}:
            features = encoded.new_tensor([task_state_features(as_dict(o)) for o in observations])
            encoded = encoded + self.tool_state_projection(features)
        state, pooled = self.propagate(encoded, state)
        controls = [as_dict(observation).get("controls", []) for observation in observations]
        max_targets = max(1, max(map(len, controls)))
        target_logits = pooled.new_full((len(observations), max_targets), -1e9)
        query = self.target_query(pooled)
        for row, candidates in enumerate(controls):
            if candidates:
                keys = self.encode_controls(candidates, target=True)
                if self.control_encoding == "semantic_tool_v1":
                    features = keys.new_tensor([control_features(as_dict(c)) for c in candidates])
                    keys = keys + self.control_projection(features)
                scores = (keys * query[row]).sum(-1) / self.hidden_size**0.5
                enabled = torch.tensor(
                    [as_dict(c).get("enabled", True) for c in candidates], device=self.device
                )
                target_logits[row, : len(candidates)] = scores.masked_fill(~enabled, -1e9)
            else:
                target_logits[row, 0] = 0
        return PolicyOutput(
            self.action_head(pooled),
            target_logits,
            self.decode_characters(pooled, self.value_decoder, self.value_chars, teacher_values),
            self.decode_characters(pooled, self.answer_decoder, self.answer_chars, teacher_answers),
            self.value_head(pooled).squeeze(-1),
            self.pointer_head(pooled).sigmoid(),
            state,
            pooled,
        )

    def neural_activity(self, state: torch.Tensor, top_k: int = 10) -> dict:
        activity = state.detach().float().square().mean(dim=(0, 2)).sqrt().cpu().numpy()
        selected = np.argsort(-activity, kind="stable")[:top_k]
        metadata = getattr(self.graph, "metadata", [])
        top = [
            {
                "index": int(i),
                "body_id": int(self.graph.body_ids[i]),
                "activity": float(activity[i]),
                **(metadata[i] if len(metadata) > i else {}),
            }
            for i in selected
        ]
        populations = {}
        for column, name in enumerate(("sensory", "visual", "intrinsic", "descending", "efferent")):
            mask = np.asarray(self.graph.node_features)[:, column] > 0
            populations[name] = float(activity[mask].mean()) if mask.any() else 0.0
        return {"top_neurons": top, "populations": populations, "source": "recurrent_hidden_state_rms"}

    @torch.no_grad()
    def act(
        self,
        observation: Any,
        state: torch.Tensor | None = None,
        *,
        sample: bool = False,
        chart_pixels: torch.Tensor | None = None,
    ):
        from habfly.contracts import Action

        self.eval()
        if (as_dict(observation).get("chart_crop") or chart_pixels is not None) and not self.vision_trained:
            raise ValueError(
                "chart_vision_untrained: train with chart-crop demonstrations before visual inference"
            )
        output = self([observation], state, chart_pixels)
        controls = as_dict(observation).get("controls", [])
        action_mask = legal_action_mask(observation, self.device)
        ap = (output.action_logits[0].masked_fill(~action_mask, -1e9) / self.action_temperature).softmax(-1)
        choose = lambda p: int(torch.multinomial(p, 1)) if sample else int(p.argmax())
        action_index = choose(ap)
        kind = ACTION_KINDS[action_index]
        target_mask = legal_target_mask(observation, kind, self.device)
        target_logits = output.target_logits[0].clone()
        if controls and kind not in ("WAIT", "STOP"):
            target_logits[: len(controls)] = target_logits[: len(controls)].masked_fill(~target_mask, -1e9)
        tp = (target_logits / self.target_temperature).softmax(-1)
        target_index = choose(tp)
        target = as_dict(controls[target_index]) if controls else {}
        value = self.tokenizer.decode(output.typed_value_logits[0].argmax(-1).tolist())
        if kind == "SELECT" and target.get("options"):
            if self.selection_mode != "characters":
                value = target["options"][
                    int(self.option_scores(observation, target, output.pooled).argmax())
                ]
            else:
                value = self.select_characters(target["options"], output.pooled)
        fields = {
            "kind": kind,
            "target": target.get("id") if kind not in ("WAIT", "STOP") else None,
            "value": value if kind in ("TYPE", "SELECT", "KEYPRESS") else None,
            "action_confidence": float(ap[action_index]),
            "target_confidence": float(tp[target_index])
            if controls and kind not in ("WAIT", "STOP")
            else None,
            "observation_revision": as_dict(observation).get("revision", 0),
            "calibrated": self.calibration.get("status") == "calibrated",
        }
        if kind in ("HOVER", "DRAG"):
            fields.update(x=float(output.pointer[0, 0]), y=float(output.pointer[0, 1]))
        if kind in ("DRAG", "SCROLL"):
            fields.update(dx=float(output.pointer[0, 2]) * 2 - 1, dy=float(output.pointer[0, 3]) * 2 - 1)
        action = Action(**fields)
        diagnostics = self.neural_activity(output.state)
        diagnostics.update(
            {
                "action_probability": float(ap[action_index]),
                "target_probability": float(tp[target_index])
                if controls and kind not in ("WAIT", "STOP")
                else None,
                "calibration": self.calibration,
                "answer": self.tokenizer.decode(output.answer_logits[0].argmax(-1).tolist()),
                "expected_value": float(output.value[0]),
            }
        )
        return action, output.state, diagnostics

    def select_characters(self, candidates, pooled):
        # Choose a legal option by the character decoder's mean log likelihood.
        labels = torch.zeros((len(candidates), self.max_answer_length), device=self.device, dtype=torch.long)
        for row, option in enumerate(candidates):
            ids = self.tokenizer.encode(option, self.max_answer_length + 1)[1:]
            labels[row, : len(ids)] = torch.tensor(ids, device=self.device)
        logits = self.decode_characters(
            pooled.expand(len(candidates), -1), self.value_decoder, self.value_chars, labels
        )
        token_scores = logits.log_softmax(-1).gather(-1, labels[:, :, None]).squeeze(-1)
        lengths = labels.ne(0).sum(-1)
        scores = (token_scores * labels.ne(0)).sum(-1) / lengths
        return candidates[int(scores.argmax())]


class TopologyFreePolicy(ConnectomePolicy):
    """Exactly parameter-matched control: shared cell sees its own previous state."""

    def propagate(
        self, encoded: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, n = len(encoded), len(self.graph.body_ids)
        if state is None:
            state = encoded.new_zeros((batch, n, self.hidden_size))
        if state.shape != (batch, n, self.hidden_size):
            raise ValueError("Recurrent state does not match this graph and batch")
        # Broadcast text to the shared recurrent population without a topology.
        injected = (
            self.sensory_projection(encoded)[:, None, :] + self.feature_projection(self.features)[None, :, :]
        )
        for _ in range(self.propagation_steps):
            update = self.norm(state + injected)
            state = self.cell(
                update.reshape(-1, self.hidden_size), state.reshape(-1, self.hidden_size)
            ).reshape(batch, n, self.hidden_size)
        return state, state[:, self.efferent, :].mean(dim=1)
