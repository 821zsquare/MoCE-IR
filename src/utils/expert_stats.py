import csv
import json
import os
import re
from collections import OrderedDict

import torch


class ExpertActivationStats:
    def __init__(self, model, degradation_names=None):
        self.degradation_names = degradation_names or []
        self.current_degradations = None
        self.layers = OrderedDict()
        self.handles = []

        for name, module in model.named_modules():
            if self._is_routing_module(name, module):
                layer_name = self._format_layer_name(name)
                self.layers[layer_name] = {
                    "module": name,
                    "num_experts": int(module.num_experts),
                    "overall": self._new_bucket(int(module.num_experts)),
                    "by_degradation": OrderedDict(),
                }
                self.handles.append(module.register_forward_hook(self._make_hook(layer_name)))

    @staticmethod
    def _is_routing_module(name, module):
        return (
            name.endswith(".adapter.routing")
            and module.__class__.__name__ == "RoutingFunction"
            and hasattr(module, "num_experts")
        )

    @staticmethod
    def _format_layer_name(module_name):
        match = re.search(r"(?:^|\.)dec\.(\d+)\.2\.layers\.(\d+)\.adapter\.routing$", module_name)
        if match is None:
            return module_name.replace(".routing", "")
        stage_idx = int(match.group(1)) + 1
        block_idx = int(match.group(2)) + 1
        return f"decoder_stage_{stage_idx}_block_{block_idx}"

    @staticmethod
    def _new_bucket(num_experts):
        return {
            "total_images": 0,
            "total_active": 0,
            "active_counts": [0 for _ in range(num_experts)],
            "gate_sums": [0.0 for _ in range(num_experts)],
        }

    def _make_hook(self, layer_name):
        def hook(module, inputs, output):
            if self.current_degradations is None:
                return
            if not isinstance(output, tuple) or len(output) < 3:
                return

            gates, top_k_indices, _ = output[:3]
            if top_k_indices.numel() == 0:
                return

            indices = top_k_indices.detach().cpu().long()
            selected_gates = gates.detach().cpu().gather(1, indices).float()
            labels = self._normalize_labels(self.current_degradations, indices.size(0))

            layer = self.layers[layer_name]
            self._update_bucket(layer["overall"], indices, selected_gates)

            for sample_idx, label in enumerate(labels):
                bucket = layer["by_degradation"].setdefault(
                    label,
                    self._new_bucket(layer["num_experts"]),
                )
                self._update_bucket(
                    bucket,
                    indices[sample_idx : sample_idx + 1],
                    selected_gates[sample_idx : sample_idx + 1],
                )

        return hook

    def _normalize_labels(self, de_ids, batch_size):
        if torch.is_tensor(de_ids):
            values = de_ids.detach().cpu().view(-1).tolist()
        elif isinstance(de_ids, (list, tuple)):
            values = list(de_ids)
        else:
            values = [de_ids]

        labels = [self._label_from_value(value) for value in values]
        if len(labels) == 1 and batch_size > 1:
            labels = labels * batch_size
        return labels[:batch_size]

    def _label_from_value(self, value):
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return value
        try:
            index = int(value)
        except (TypeError, ValueError):
            return str(value)
        if 0 <= index < len(self.degradation_names):
            return self.degradation_names[index]
        return str(index)

    @staticmethod
    def _update_bucket(bucket, indices, selected_gates):
        bucket["total_images"] += int(indices.size(0))
        bucket["total_active"] += int(indices.numel())
        flat_indices = indices.reshape(-1).tolist()
        flat_gates = selected_gates.reshape(-1).tolist()
        for expert_idx, gate_weight in zip(flat_indices, flat_gates):
            bucket["active_counts"][expert_idx] += 1
            bucket["gate_sums"][expert_idx] += float(gate_weight)

    def set_batch(self, de_ids):
        self.current_degradations = de_ids

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self):
        return {
            layer_name: {
                "module": layer["module"],
                "num_experts": layer["num_experts"],
                "overall": self._summarize_bucket(layer["overall"]),
                "by_degradation": {
                    label: self._summarize_bucket(bucket)
                    for label, bucket in layer["by_degradation"].items()
                },
            }
            for layer_name, layer in self.layers.items()
        }

    @staticmethod
    def _summarize_bucket(bucket):
        total_images = max(bucket["total_images"], 1)
        total_active = max(bucket["total_active"], 1)
        rows = []
        for expert_idx, active_count in enumerate(bucket["active_counts"]):
            gate_sum = bucket["gate_sums"][expert_idx]
            rows.append(
                {
                    "expert": expert_idx,
                    "active_count": active_count,
                    "activation_probability": active_count / total_images,
                    "active_fraction": active_count / total_active,
                    "gate_mass_per_image": gate_sum / total_images,
                    "mean_gate_when_active": gate_sum / active_count if active_count > 0 else 0.0,
                }
            )
        return {
            "total_images": bucket["total_images"],
            "total_active": bucket["total_active"],
            "experts": rows,
        }

    def save(self, output_dir, prefix="expert_activation_stats"):
        os.makedirs(output_dir, exist_ok=True)
        summary = self.summary()
        json_path = os.path.join(output_dir, f"{prefix}.json")
        csv_path = os.path.join(output_dir, f"{prefix}.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scope",
                    "degradation",
                    "layer",
                    "module",
                    "expert",
                    "total_images",
                    "total_active",
                    "active_count",
                    "activation_probability",
                    "active_fraction",
                    "gate_mass_per_image",
                    "mean_gate_when_active",
                ],
            )
            writer.writeheader()
            for layer_name, layer in summary.items():
                self._write_bucket_rows(writer, layer_name, layer, "overall", "ALL", layer["overall"])
                for degradation, bucket in layer["by_degradation"].items():
                    self._write_bucket_rows(writer, layer_name, layer, "degradation", degradation, bucket)

        return json_path, csv_path

    @staticmethod
    def _write_bucket_rows(writer, layer_name, layer, scope, degradation, bucket):
        for expert in bucket["experts"]:
            writer.writerow(
                {
                    "scope": scope,
                    "degradation": degradation,
                    "layer": layer_name,
                    "module": layer["module"],
                    "expert": expert["expert"],
                    "total_images": bucket["total_images"],
                    "total_active": bucket["total_active"],
                    "active_count": expert["active_count"],
                    "activation_probability": expert["activation_probability"],
                    "active_fraction": expert["active_fraction"],
                    "gate_mass_per_image": expert["gate_mass_per_image"],
                    "mean_gate_when_active": expert["mean_gate_when_active"],
                }
            )
