#!/usr/bin/env python3
"""Train a small Transformer on preorder node records from Newick subtrees.

Run: python newick_node_lm.py --config newick_node_lm.yaml
Smoke check: python newick_node_lm.py --profile smoke --device cpu

One sequence position represents (value, child_count), not a node identity.
The preorder degree sequence preserves the rooted, ordered topology:
https://drops.dagstuhl.de/storage/00lipics/lipics-vol154-stacs2020/LIPIcs.STACS.2020.22/LIPIcs.STACS.2020.22.pdf
Related tree positional encoding work:
https://papers.neurips.cc/paper_files/paper/2019/hash/6e0917469214d8fbd8c517dcdc6b8dcf-Abstract.html

This is a synthetic-feature experiment, not a biological forecast. Settings
are loaded from YAML only when requested; importing this module never trains.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import io
import json
import math
import platform
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
import Bio
from Bio import Phylo
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml


@dataclass(frozen=True)
class Config:
    """Experiment settings loaded from newick_node_lm.yaml."""

    seed: int
    source_path: str
    output_dir: str
    min_nodes: int
    max_nodes: int
    n_shapes: int
    n_values: int
    context_length: int
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    patience: int
    grad_clip: float
    generation_trials: int
    device: str


def validate_config(config):
    integer_fields = (
        "seed",
        "min_nodes",
        "max_nodes",
        "n_shapes",
        "n_values",
        "context_length",
        "n_layers",
        "d_model",
        "n_heads",
        "d_ff",
        "batch_size",
        "max_epochs",
        "patience",
        "generation_trials",
    )
    for name in integer_fields:
        value = getattr(config, name)
        minimum = 0 if name == "seed" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    for name in ("dropout", "learning_rate", "weight_decay", "grad_clip"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
    if not 2 <= config.min_nodes <= config.max_nodes:
        raise ValueError("Require 2 <= min_nodes <= max_nodes.")
    if config.max_nodes < 8:
        raise ValueError("max_nodes must be >= 8 to run the verification fixtures.")
    if config.n_shapes < 20 or config.n_values < 2:
        raise ValueError("Require n_shapes >= 20 and n_values >= 2.")
    if config.context_length < config.max_nodes + 1:
        raise ValueError("context_length must be >= max_nodes + 1.")
    if config.d_model % config.n_heads:
        raise ValueError("d_model must be divisible by n_heads.")
    if (
        not 0 <= config.dropout < 1
        or config.learning_rate <= 0
        or config.weight_decay < 0
        or (config.grad_clip <= 0)
    ):
        raise ValueError(
            "Invalid dropout, learning rate, weight decay, or gradient clipping setting."
        )
    if not isinstance(config.device, str) or config.device not in {
        "auto",
        "cpu",
        "cuda",
    }:
        raise ValueError("device must be auto, cpu, or cuda.")
    for name in ("source_path", "output_dir"):
        if (
            not isinstance(getattr(config, name), str)
            or not getattr(config, name).strip()
        ):
            raise ValueError(f"{name} must be a nonempty path string.")


def load_config(path, profile=None, device=None):
    """Load YAML safely. Relative data/output paths are relative to the YAML file."""
    path = Path(path).expanduser().resolve()
    with path.open() as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    values = dict(document)
    configured_profile = values.pop("profile", "full")
    profiles = values.pop("profiles", {})
    selected_profile = configured_profile if profile is None else profile
    if not isinstance(selected_profile, str) or selected_profile not in {
        "full",
        "smoke",
    }:
        raise ValueError("profile must be full or smoke.")
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a YAML mapping.")
    expected = {item.name for item in fields(Config)}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown or missing:
        raise ValueError(
            f"Invalid config fields: unknown={sorted(map(str, unknown))}, missing={sorted(missing)}"
        )
    for name, overrides in profiles.items():
        if name not in {"full", "smoke"} or not isinstance(overrides, dict):
            raise ValueError("profiles may contain full/smoke mappings only.")
        if set(overrides) - expected:
            raise ValueError(
                f"Unknown fields in profile {name}: {sorted(map(str, set(overrides) - expected))}"
            )
    if selected_profile == "smoke" and "smoke" not in profiles:
        raise ValueError("The smoke profile needs a profiles.smoke mapping in YAML.")
    values.update(profiles.get(selected_profile, {}))
    if device is not None:
        values["device"] = device
    for name in ("source_path", "output_dir"):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a nonempty path string.")
        value = Path(values[name]).expanduser()
        values[name] = str((path.parent / value).resolve())
    config = Config(**values)
    validate_config(config)
    return (config, selected_profile)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


STRUCTURAL = re.compile(b"[(),;]")
CACHE_VERSION = 1


@dataclass(frozen=True)
class NodeRecord:
    value: int
    child_count: int


@dataclass
class FeatureNode:
    value: int
    children: list = field(default_factory=list)


class NodeFeatureEncoder(nn.Module):
    """Replace or extend this module when adding real categorical/numeric features."""

    def __init__(self, config):
        super().__init__()
        self.value = nn.Embedding(config.n_values, config.d_model)
        self.child_count = nn.Embedding(config.max_nodes, config.d_model)

    def forward(self, values, child_counts):
        return self.value(values) + self.child_count(child_counts)


class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.width = config.d_model // config.n_heads
        self.dropout_p = config.dropout
        self.norm1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
        )
        attention = attention.transpose(1, 2).contiguous().view(batch, length, width)
        x = x + self.dropout(self.projection(attention))
        return x + self.mlp(self.norm2(x))


class NodeTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.n_values * config.max_nodes
        self.features = NodeFeatureEncoder(config)
        self.special = nn.Embedding(2, config.d_model)
        self.depth = nn.Embedding(config.max_nodes, config.d_model)
        self.sibling = nn.Embedding(config.max_nodes, config.d_model)
        self.position = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.n_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, self.vocab_size, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        ids = batch["input_ids"]
        if ids.size(1) > self.config.context_length:
            raise ValueError("Input exceeds configured context length.")
        values = ids.remainder(self.config.n_values)
        counts = (ids // self.config.n_values).clamp(max=self.config.max_nodes - 1)
        x = self.features(values, counts)
        controls = self.special((ids - self.vocab_size).clamp(0, 1))
        x = torch.where((ids < self.vocab_size).unsqueeze(-1), x, controls)
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.dropout(
            x
            + self.depth(batch["depths"])
            + self.sibling(batch["siblings"])
            + self.position(positions)[None]
        )
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


class MetricTotals:

    def __init__(self, config):
        self.config = config
        self.sums = dict(
            nodes=0,
            trees=0,
            joint_nll=0.0,
            joint_correct=0,
            feature_correct=0,
            child_count_correct=0,
            non_root_nodes=0,
            non_root_feature_correct=0,
            non_root_feature_nll=0.0,
            return_nodes=0,
            return_feature_correct=0,
            return_feature_nll=0.0,
        )

    @torch.no_grad()
    def update(self, logp, batch):
        target = batch["targets"]
        valid = target != -100
        safe = target.clamp(min=0)
        value, count = (safe % self.config.n_values, safe // self.config.n_values)
        joint_nll = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        joint = logp.view(*logp.shape[:-1], self.config.max_nodes, self.config.n_values)
        value_logp = torch.logsumexp(joint, dim=-2)
        count_logp = torch.logsumexp(joint, dim=-1)
        value_ok = value_logp.argmax(-1) == value
        value_nll = -value_logp.gather(-1, value.unsqueeze(-1)).squeeze(-1)
        self.sums["nodes"] += int(valid.sum())
        self.sums["trees"] += target.size(0)
        self.sums["joint_nll"] += float(joint_nll[valid].sum())
        self.sums["joint_correct"] += int(((logp.argmax(-1) == safe) & valid).sum())
        self.sums["feature_correct"] += int((value_ok & valid).sum())
        self.sums["child_count_correct"] += int(
            ((count_logp.argmax(-1) == count) & valid).sum()
        )
        non_root = valid & (
            torch.arange(target.size(1), device=target.device)[None] > 0
        )
        for name, mask in (
            ("non_root", non_root),
            ("return", valid & batch["return_mask"]),
        ):
            self.sums[f"{name}_nodes"] += int(mask.sum())
            self.sums[f"{name}_feature_correct"] += int((value_ok & mask).sum())
            self.sums[f"{name}_feature_nll"] += float(value_nll[mask].sum())

    def result(self):
        s = self.sums
        result = dict(
            nodes=s["nodes"],
            trees=s["trees"],
            joint_nll=s["joint_nll"] / s["nodes"],
            joint_accuracy=s["joint_correct"] / s["nodes"],
            feature_accuracy=s["feature_correct"] / s["nodes"],
            child_count_accuracy=s["child_count_correct"] / s["nodes"],
        )
        for name in ("non_root", "return"):
            count = s[f"{name}_nodes"]
            result[f"{name}_nodes"] = count
            result[f"{name}_feature_accuracy"] = (
                s[f"{name}_feature_correct"] / count if count else None
            )
            result[f"{name}_feature_nll"] = (
                s[f"{name}_feature_nll"] / count if count else None
            )
        return result


class Experiment:
    """Own a run's configuration, data, model, and output paths."""

    def __init__(self, config, profile="full"):
        validate_config(config)
        if profile not in {"full", "smoke"}:
            raise ValueError("profile must be full or smoke.")
        self.cfg = config
        self.profile = profile
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        if self.device.type == "cuda" and (not torch.cuda.is_available()):
            raise ValueError("CUDA was requested but no CUDA device is available.")
        self.use_bf16 = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.base_out = Path(config.output_dir)
        self.out = self.base_out if profile == "full" else self.base_out / "smoke"
        self.VOCAB_SIZE = config.n_values * config.max_nodes
        self.BOS_ID, self.PAD_ID = (self.VOCAB_SIZE, self.VOCAB_SIZE + 1)
        self.model = None
        self.checks = {}

    def run(self, check_data_only=False):
        seed_everything(self.cfg.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.out.mkdir(parents=True, exist_ok=True)
        self.environment = dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            biopython=Bio.__version__,
            matplotlib=matplotlib.__version__,
            pyyaml=yaml.__version__,
            device=str(self.device),
            precision="bfloat16" if self.use_bf16 else "float32",
            profile=self.profile,
        )
        if self.device.type == "cuda":
            self.environment["gpu"] = torch.cuda.get_device_name(self.device)
        save_json(self.out / "config.json", asdict(self.cfg))
        save_json(self.out / "environment.json", self.environment)
        (self.out / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "profile": self.profile,
                    **asdict(self.cfg),
                    "profiles": {self.profile: {}},
                },
                sort_keys=False,
            )
        )
        print(json.dumps(self.environment, indent=2), flush=True)
        print(f"Outputs: {self.out.resolve()}", flush=True)
        self.prepare_data()
        self.prepare_examples()
        self.verify_data()
        save_json(self.out / "data_checks.json", self.checks)
        if check_data_only:
            print("Data checks passed; training was not requested.", flush=True)
            return self.checks
        self.build_model()
        self.verify_optimizer()
        self.train()
        self.report_evaluation()
        self.report_generation()
        self.show_prediction()
        self.verify_reload()
        return self.results

    @classmethod
    def from_checkpoint(cls, path, destination="cpu"):
        """Load for prediction without extracting data, writing outputs, or training."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("Unsupported checkpoint version.")
        config = replace(Config(**checkpoint["config"]), device=destination)
        experiment = cls(
            config, checkpoint.get("environment", {}).get("profile", "full")
        )
        experiment.model = experiment.load_checkpoint(path, destination)
        return experiment

    def autocast_context(self):
        return (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bf16
            else nullcontext()
        )

    def file_fingerprint(self, path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}

    def scan_clades(
        self, path, min_nodes, max_nodes, sample_size, seed, block_size=8 * 1024 * 1024
    ):
        stack, seen, heap = ([], set(), [])
        pending, trees, eligible, offset = (False, 0, 0, 0)
        started = False

        def add_child(frame, size, signature, covered):
            frame["size"] += size
            frame["covered"] |= covered
            if frame["size"] > max_nodes or frame["covered"]:
                frame["children"] = None
            elif frame["children"] is not None:
                frame["children"].append(signature)

        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(block_size), b""):
                if any((mark in block for mark in (b"'", b"[", b"]"))):
                    raise ValueError(
                        "Extractor does not support quoted labels or comments."
                    )
                if trees and block.strip():
                    raise ValueError(
                        "Expected exactly one tree with no trailing content."
                    )
                for match in STRUCTURAL.finditer(block):
                    char, position = (match[0], offset + match.start())
                    if trees:
                        raise ValueError("Expected exactly one tree.")
                    if char == b"(":
                        if started and (not pending):
                            raise ValueError("Unexpected opening parenthesis.")
                        started = True
                        stack.append(
                            dict(start=position, size=1, children=[], covered=False)
                        )
                        pending = True
                    elif char == b",":
                        if not stack:
                            raise ValueError("Comma outside a clade.")
                        if pending:
                            add_child(stack[-1], 1, "()", False)
                        pending = True
                    elif char == b")":
                        if not stack:
                            raise ValueError("Unmatched closing parenthesis.")
                        if pending:
                            add_child(stack[-1], 1, "()", False)
                        frame = stack.pop()
                        signature = (
                            "(" + "".join(sorted(frame["children"])) + ")"
                            if frame["children"] is not None
                            else None
                        )
                        if (
                            not frame["covered"]
                            and min_nodes <= frame["size"] <= max_nodes
                        ):
                            eligible += 1
                            if signature not in seen:
                                seen.add(signature)
                                priority = int.from_bytes(
                                    hashlib.sha256(
                                        f"{seed}:{signature}".encode()
                                    ).digest(),
                                    "big",
                                )
                                entry = (
                                    -priority,
                                    signature,
                                    frame["start"],
                                    position + 1,
                                )
                                if len(heap) < sample_size:
                                    heapq.heappush(heap, entry)
                                elif entry > heap[0]:
                                    heapq.heapreplace(heap, entry)
                            frame["covered"] = True
                        if stack:
                            add_child(
                                stack[-1], frame["size"], signature, frame["covered"]
                            )
                        pending = False
                    else:
                        if stack or not started:
                            raise ValueError(
                                "Unclosed clade or missing tree at semicolon."
                            )
                        trees += 1
                        if block[match.end() :].strip():
                            raise ValueError(
                                "Unexpected content after the tree terminator."
                            )
                offset += len(block)
        if stack or trees != 1:
            raise ValueError("Expected one complete, semicolon-terminated tree.")
        if len(heap) < sample_size:
            raise ValueError(
                f"Found {len(heap)} distinct shapes; requested {sample_size}."
            )
        selected = [
            dict(signature=sig, start=start, body_end=end)
            for _, sig, start, end in sorted(heap, reverse=True)
        ]
        return (
            selected,
            {"eligible_disjoint_clades": eligible, "distinct_shapes": len(seen)},
        )

    def read_fragment(self, handle, start, body_end):
        handle.seek(start)
        body = handle.read(body_end - start)
        suffix = bytearray()
        while True:
            part = handle.read(256)
            if not part:
                raise ValueError("Missing delimiter after a selected clade.")
            delimiter = STRUCTURAL.search(part)
            if delimiter:
                suffix.extend(part[: delimiter.start()])
                break
            suffix.extend(part)
        return (body + bytes(suffix) + b";").decode("utf-8")

    def clade_signature(self, clade):
        return (
            "(" + "".join(sorted((self.clade_signature(c) for c in clade.clades))) + ")"
        )

    def load_subtree_pool(self, path, sample_size, seed, min_nodes, max_nodes):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Place the Newick file at {path.resolve()}, or edit source_path in your YAML configuration."
            )
        fingerprint = self.file_fingerprint(path)
        specification = dict(
            version=CACHE_VERSION,
            fingerprint=fingerprint,
            sample_size=sample_size,
            seed=seed,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
        )
        cache_path = (
            self.base_out
            / f"subtrees_{sample_size}_{seed}_{min_nodes}_{max_nodes}.json"
        )
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached.get("specification") == specification:
                print(f"Loaded {len(cached['shapes'])} cached subtree shapes.")
                return cached
        print("Scanning the full file for distinct small clades…", flush=True)
        selected, census = self.scan_clades(
            path, min_nodes, max_nodes, sample_size, seed
        )
        shapes = []
        with path.open("rb") as handle:
            for entry in selected:
                fragment = self.read_fragment(handle, entry["start"], entry["body_end"])
                tree = Phylo.read(io.StringIO(fragment), "newick")
                nodes = list(tree.find_clades(order="preorder"))
                signature = self.clade_signature(tree.root)
                if signature != entry["signature"]:
                    raise ValueError(
                        "Streaming scanner and Biopython disagree on a clade."
                    )
                shapes.append(
                    dict(
                        shape_id=hashlib.sha256(signature.encode()).hexdigest(),
                        signature=signature,
                        counts=[len(n.clades) for n in nodes],
                        source_start=entry["start"],
                        source_body_end=entry["body_end"],
                        source_newick=fragment,
                        metadata=dict(
                            names=[n.name for n in nodes],
                            branch_lengths=[n.branch_length for n in nodes],
                        ),
                    )
                )
        result = dict(specification=specification, census=census, shapes=shapes)
        save_json(cache_path, result)
        print(f"Cached {len(shapes)} shapes. Census: {census}")
        return result

    def record_id(self, record):
        if not isinstance(record, NodeRecord):
            raise TypeError("Expected NodeRecord(value, child_count).")
        if type(record.value) is not int or type(record.child_count) is not int:
            raise ValueError("Node fields must be integers.")
        if (
            not 0 <= record.value < self.cfg.n_values
            or not 0 <= record.child_count < self.cfg.max_nodes
        ):
            raise ValueError("Node fields are outside the configured vocabulary.")
        return self.cfg.n_values * record.child_count + record.value

    def id_record(self, token):
        token = int(token)
        if not 0 <= token < self.VOCAB_SIZE:
            raise ValueError("Only node token IDs can be decoded as records.")
        return NodeRecord(token % self.cfg.n_values, token // self.cfg.n_values)

    def prefix_geometry(self, records, require_complete=False):
        if len(records) > self.cfg.max_nodes:
            raise ValueError("Too many nodes for this experiment.")
        parents, depths, siblings, returns, stack = ([], [], [], [], [])
        returned = False
        for i, record in enumerate(records):
            self.record_id(record)
            if i == 0:
                parent, depth, sibling = (-1, 0, 0)
            else:
                if not stack:
                    raise ValueError("Extra node after a complete tree.")
                parent, remaining = stack[-1]
                depth = depths[parent] + 1
                sibling = records[parent].child_count - remaining + 1
                stack[-1][1] -= 1
            parents.append(parent)
            depths.append(depth)
            siblings.append(sibling)
            returns.append(returned)
            if record.child_count:
                stack.append([i, record.child_count])
            popped = 0
            while stack and stack[-1][1] == 0:
                stack.pop()
                popped += 1
            returned = bool(popped and stack)
        slots = sum((remaining for _, remaining in stack)) if records else 1
        if len(records) + slots > self.cfg.max_nodes:
            raise ValueError("Prefix cannot be completed within max_nodes.")
        complete = bool(records) and slots == 0
        if require_complete and (not complete):
            raise ValueError("Incomplete tree: child slots remain unfilled.")
        next_parent = stack[-1][0] if stack else -1
        next_depth = depths[next_parent] + 1 if stack else 0
        next_sibling = (
            records[next_parent].child_count - stack[-1][1] + 1 if stack else 0
        )
        return dict(
            parents=parents,
            depths=depths,
            siblings=siblings,
            returns=returns,
            slots=slots,
            complete=complete,
            next_parent=next_parent,
            next_depth=next_depth,
            next_sibling=next_sibling,
        )

    def encode_tree(self, root):
        records, stack = ([], [root])
        while stack:
            node = stack.pop()
            records.append(NodeRecord(node.value, len(node.children)))
            stack.extend(reversed(node.children))
        self.prefix_geometry(records, require_complete=True)
        return records

    def decode_records(self, records):
        geometry = self.prefix_geometry(records, require_complete=True)
        nodes = [FeatureNode(r.value) for r in records]
        for i, parent in enumerate(geometry["parents"]):
            if parent >= 0:
                nodes[parent].children.append(nodes[i])
        return nodes[0]

    def assign_features(self, counts, root_value):
        template = [NodeRecord(0, int(k)) for k in counts]
        geometry = self.prefix_geometry(template, require_complete=True)
        values = [int(root_value)]
        for parent, sibling in zip(geometry["parents"][1:], geometry["siblings"][1:]):
            values.append((values[parent] + sibling) % self.cfg.n_values)
        return [NodeRecord(value, count) for value, count in zip(values, counts)]

    def collate(self, rows):
        longest = max((len(row["records"]) for row in rows))
        if longest > self.cfg.context_length:
            raise ValueError("Example exceeds context; no truncation is performed.")
        shape = (len(rows), longest)
        batch = dict(
            input_ids=torch.full(shape, self.PAD_ID, dtype=torch.long),
            depths=torch.zeros(shape, dtype=torch.long),
            siblings=torch.zeros(shape, dtype=torch.long),
            targets=torch.full(shape, -100, dtype=torch.long),
            return_mask=torch.zeros(shape, dtype=torch.bool),
        )
        for i, row in enumerate(rows):
            records = row["records"]
            geometry = self.prefix_geometry(records, require_complete=True)
            tokens = [self.record_id(r) for r in records]
            n = len(tokens)
            batch["input_ids"][i, :n] = torch.tensor([self.BOS_ID] + tokens[:-1])
            batch["targets"][i, :n] = torch.tensor(tokens)
            batch["depths"][i, :n] = torch.tensor([0] + geometry["depths"][:-1])
            batch["siblings"][i, :n] = torch.tensor([0] + geometry["siblings"][:-1])
            batch["return_mask"][i, :n] = torch.tensor(geometry["returns"])
        return batch

    def to_device(self, batch, destination=None):
        destination = self.device if destination is None else destination
        return {key: value.to(destination) for key, value in batch.items()}

    def make_loader(self, rows, shuffle=False):
        return DataLoader(
            rows,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.cfg.seed),
        )

    def expect_value_error(self, function):
        try:
            function()
        except ValueError:
            return
        raise AssertionError("Expected a ValueError.")

    def check_data(self):
        chain = FeatureNode(0 % self.cfg.n_values)
        for value in range(1, 8):
            chain = FeatureNode(value % self.cfg.n_values, [chain])
        fixtures = [
            FeatureNode(3 % self.cfg.n_values),
            chain,
            FeatureNode(
                2 % self.cfg.n_values,
                [
                    FeatureNode(3 % self.cfg.n_values),
                    FeatureNode(
                        4 % self.cfg.n_values,
                        [
                            FeatureNode(5 % self.cfg.n_values),
                            FeatureNode(6 % self.cfg.n_values),
                        ],
                    ),
                ],
            ),
            FeatureNode(
                6 % self.cfg.n_values,
                [FeatureNode(i % self.cfg.n_values) for i in range(6)],
            ),
        ]
        for tree in fixtures:
            records = self.encode_tree(tree)
            assert self.decode_records(records) == tree
            assert self.encode_tree(self.decode_records(records)) == records
            assert sum((r.child_count for r in records)) == len(records) - 1
            for length in range(len(records) + 1):
                prefix = self.prefix_geometry(records[:length])
                full = self.prefix_geometry(records)
                for key in ("parents", "depths", "siblings", "returns"):
                    assert prefix[key] == full[key][:length]
        return_tree = FeatureNode(
            0 % self.cfg.n_values,
            [
                FeatureNode(
                    1 % self.cfg.n_values,
                    [
                        FeatureNode(2 % self.cfg.n_values),
                        FeatureNode(3 % self.cfg.n_values),
                    ],
                ),
                FeatureNode(4 % self.cfg.n_values),
            ],
        )
        assert self.prefix_geometry(self.encode_tree(return_tree))["returns"] == [
            False
        ] * 4 + [True]
        assert not any(self.prefix_geometry(self.encode_tree(fixtures[-1]))["returns"])
        self.expect_value_error(
            lambda: self.decode_records([NodeRecord(0, 2), NodeRecord(1, 0)])
        )
        self.expect_value_error(
            lambda: self.decode_records([NodeRecord(0, 0), NodeRecord(1, 0)])
        )
        self.expect_value_error(
            lambda: self.record_id(NodeRecord(self.cfg.n_values, 0))
        )
        self.expect_value_error(
            lambda: self.prefix_geometry(
                [
                    NodeRecord(0, self.cfg.max_nodes - 1),
                    NodeRecord(0, self.cfg.max_nodes - 1),
                ]
            )
        )
        fixture_dir = self.out / "check_fixtures"
        fixture_dir.mkdir(exist_ok=True)
        path = fixture_dir / "extractor.nwk"
        path.write_text("((a:1,b:2)x:3,(c:1,d:2,e:3)y:4)r:0;\n")
        reference = self.scan_clades(path, 3, 7, 2, 42, block_size=4096)
        for block_size in (1, 7, 13):
            assert (
                self.scan_clades(path, 3, 7, 2, 42, block_size=block_size) == reference
            )
        with path.open("rb") as handle:
            for entry in reference[0]:
                parsed = Phylo.read(
                    io.StringIO(
                        self.read_fragment(handle, entry["start"], entry["body_end"])
                    ),
                    "newick",
                )
                assert self.clade_signature(parsed.root) == entry["signature"]
                assert parsed.root.name in {"x", "y"}
                assert parsed.root.branch_length in {3.0, 4.0}
        for malformed in (
            "('a,b',c)r;",
            "(a[note],b)r;",
            "(a,b)r",
            "(a,b;",
            "(a,b)r;extra",
        ):
            path.write_text(malformed)
            self.expect_value_error(
                lambda: self.scan_clades(path, 2, 7, 1, 42, block_size=7)
            )
        sig1 = self.clade_signature(
            Phylo.read(io.StringIO("(a:1,(b,c)x)r;"), "newick").root
        )
        sig2 = self.clade_signature(
            Phylo.read(io.StringIO("((e,d)y:99,f)z;"), "newick").root
        )
        assert sig1 == sig2
        groups = [set(ids) for ids in self.split_ids.values()]
        assert all(
            (not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3))
        )
        assert len(set.union(*groups)) == len(self.shapes)
        assert len({s["signature"] for s in self.shapes}) == len(self.shapes)
        intervals = sorted(
            ((s["source_start"], s["source_body_end"]) for s in self.shapes)
        )
        assert all(
            (end <= start for (_, end), (start, _) in zip(intervals, intervals[1:]))
        )
        for name, rows in self.examples.items():
            assert len(rows) == len(self.split_ids[name]) * self.cfg.n_values
            allowed = set(self.split_ids[name])
            for row in rows:
                records = row["records"]
                geometry = self.prefix_geometry(records, require_complete=True)
                assert row["shape_id"] in allowed
                assert self.cfg.min_nodes <= len(records) <= self.cfg.max_nodes
                assert self.encode_tree(self.decode_records(records)) == records
                assert records[0].value == row["root_value"]
                for i in range(1, len(records)):
                    assert (
                        records[i].value
                        == (
                            records[geometry["parents"][i]].value
                            + geometry["siblings"][i]
                        )
                        % self.cfg.n_values
                    )
        rows = [
            min(self.examples["train"], key=lambda r: len(r["records"])),
            max(self.examples["train"], key=lambda r: len(r["records"])),
        ]
        batch = self.collate(rows)
        assert set(batch) == {
            "input_ids",
            "depths",
            "siblings",
            "targets",
            "return_mask",
        }
        for i, row in enumerate(rows):
            tokens = [self.record_id(r) for r in row["records"]]
            n = len(tokens)
            assert batch["input_ids"][i, :n].tolist() == [self.BOS_ID] + tokens[:-1]
            assert batch["targets"][i, :n].tolist() == tokens
            assert (batch["input_ids"][i, n:] == self.PAD_ID).all()
            assert (batch["targets"][i, n:] == -100).all()
        self.checks["data_contract"] = True
        print(
            "PASS: extraction boundaries, round trips, shape splits, features, shifts, and padding."
        )

    def next_node_loss(self, logits, targets):
        return F.cross_entropy(
            logits.float().reshape(-1, self.VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=-100,
        )

    @torch.no_grad()
    def check_model_math(self, net):
        net.eval()
        rows = [
            min(self.examples["train"], key=lambda r: len(r["records"])),
            max(self.examples["train"], key=lambda r: len(r["records"])),
        ]
        batch = self.to_device(self.collate(rows))
        logits = net(batch)
        changed = {key: value.clone() for key, value in batch.items()}
        cut = 4
        suffix = changed["input_ids"][:, cut:]
        changed["input_ids"][:, cut:] = torch.where(
            suffix < self.VOCAB_SIZE,
            suffix // self.cfg.n_values * self.cfg.n_values
            + (suffix + 1) % self.cfg.n_values,
            suffix,
        )
        torch.testing.assert_close(
            net(changed)[:, :cut], logits[:, :cut], rtol=1e-05, atol=1e-06
        )
        for i, row in enumerate(rows):
            single = net(self.to_device(self.collate([row])))
            torch.testing.assert_close(
                single[0], logits[i, : len(row["records"])], rtol=0.0001, atol=1e-05
            )
        valid = batch["targets"] != -100
        direct = F.cross_entropy(logits[valid].float(), batch["targets"][valid])
        torch.testing.assert_close(
            self.next_node_loss(logits, batch["targets"]), direct
        )
        self.checks["causality_and_padding"] = True
        print("PASS: future-token perturbations, batch padding, and masked loss.")

    def check_tiny_overfit(self):
        seed_everything(self.cfg.seed + 1)
        net = NodeTransformer(replace(self.cfg, dropout=0.0)).to(self.device)
        row = min(self.examples["train"], key=lambda r: len(r["records"]))
        batch = self.to_device(self.collate([row]))
        optimizer = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.0)
        final_loss, accuracy = (math.inf, 0.0)
        for step in range(1, 201):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            loss = self.next_node_loss(net(batch), batch["targets"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
            optimizer.step()
            if step % 10 == 0:
                net.eval()
                with torch.no_grad():
                    logits = net(batch)
                    final_loss = self.next_node_loss(logits, batch["targets"]).item()
                    accuracy = (
                        (logits.argmax(-1) == batch["targets"]).float().mean().item()
                    )
                if final_loss < 0.05 and accuracy == 1.0:
                    break
        if final_loss >= 0.05 or accuracy != 1.0:
            raise AssertionError(
                f"Tiny overfit failed: loss={final_loss:.4f}, accuracy={accuracy:.3f}"
            )
        self.checks["tiny_overfit"] = dict(
            passed=True, steps=step, loss=final_loss, accuracy=accuracy
        )
        print(
            f"PASS: one-tree overfit in {step} steps; loss={final_loss:.4f}, accuracy={accuracy:.1%}."
        )
        del net, optimizer
        seed_everything(self.cfg.seed)

    @torch.no_grad()
    def evaluate(self, net, loader):
        net.eval()
        totals = MetricTotals(self.cfg)
        for batch in loader:
            batch = self.to_device(batch)
            with self.autocast_context():
                logits = net(batch)
            totals.update(F.log_softmax(logits.float(), dim=-1), batch)
        return totals.result()

    def fit(self, net):
        seed_everything(self.cfg.seed)
        optimizer = torch.optim.AdamW(
            net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        history, best_loss, stale = ([], math.inf, 0)
        started = time.monotonic()
        for epoch in range(1, self.cfg.max_epochs + 1):
            net.train()
            epoch_loss, epoch_nodes = (0.0, 0)
            epoch_started = time.monotonic()
            for batch in self.loaders["train"]:
                batch = self.to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self.autocast_context():
                    logits = net(batch)
                    loss = self.next_node_loss(logits, batch["targets"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), self.cfg.grad_clip)
                optimizer.step()
                n = int((batch["targets"] != -100).sum())
                epoch_loss += loss.item() * n
                epoch_nodes += n
            validation = self.evaluate(net, self.loaders["validation"])
            row = dict(
                epoch=epoch,
                train_joint_nll=epoch_loss / epoch_nodes,
                validation=validation,
                seconds=time.monotonic() - epoch_started,
            )
            history.append(row)
            if validation["joint_nll"] < best_loss:
                best_loss, stale = (validation["joint_nll"], 0)
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in net.state_dict().items()
                }
                torch.save(
                    dict(
                        version=1,
                        config=asdict(self.cfg),
                        state_dict=best_state,
                        best_epoch=epoch,
                        validation=validation,
                        source_fingerprint=self.pool["specification"]["fingerprint"],
                        split_ids=self.split_ids,
                        environment=self.environment,
                    ),
                    self.out / "checkpoint.pt",
                )
            else:
                stale += 1
            save_json(self.out / "history.json", history)
            ret = validation["return_feature_accuracy"]
            ret_text = "n/a" if ret is None else f"{ret:.1%}"
            print(
                f"Epoch {epoch:02d} | train NLL {row['train_joint_nll']:.4f} | val NLL {validation['joint_nll']:.4f} | val return-value accuracy {ret_text} | {row['seconds']:.1f}s",
                flush=True,
            )
            if stale >= self.cfg.patience:
                print("Early stopping.")
                break
        checkpoint = torch.load(
            self.out / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        elapsed = time.monotonic() - started
        print(
            f"Restored epoch {checkpoint['best_epoch']}; total training time {elapsed:.1f}s."
        )
        return (history, checkpoint["best_epoch"], elapsed)

    def fit_baselines(self, rows):
        unigram = torch.ones(self.VOCAB_SIZE, dtype=torch.float64)
        bigram = torch.ones(self.VOCAB_SIZE + 1, self.VOCAB_SIZE, dtype=torch.float64)
        for row in rows:
            tokens = [self.record_id(r) for r in row["records"]]
            previous = [self.BOS_ID] + tokens[:-1]
            for prev, token in zip(previous, tokens):
                unigram[token] += 1
                bigram[prev, token] += 1
        return dict(
            unigram=(unigram / unigram.sum()).log().float(),
            bigram=(bigram / bigram.sum(-1, keepdim=True)).log().float(),
        )

    @torch.no_grad()
    def evaluate_baseline(self, name, log_tables, loader):
        totals = MetricTotals(self.cfg)
        for batch in loader:
            if name == "unigram":
                logp = log_tables[name].expand(*batch["targets"].shape, self.VOCAB_SIZE)
            else:
                logp = log_tables[name][batch["input_ids"].clamp(max=self.BOS_ID)]
            totals.update(logp, batch)
        return totals.result()

    def percent(self, value):
        return "n/a" if value is None else f"{100 * value:.1f}%"

    def prefix_batch(self, records, destination):
        geometry = self.prefix_geometry(records)
        tokens = [self.BOS_ID] + [self.record_id(r) for r in records]
        return dict(
            input_ids=torch.tensor([tokens], dtype=torch.long, device=destination),
            depths=torch.tensor(
                [[0] + geometry["depths"]], dtype=torch.long, device=destination
            ),
            siblings=torch.tensor(
                [[0] + geometry["siblings"]], dtype=torch.long, device=destination
            ),
        )

    @torch.no_grad()
    def next_logits(self, prefix, net=None):
        net = self.model if net is None else net
        if (net.config.n_values, net.config.max_nodes) != (
            self.cfg.n_values,
            self.cfg.max_nodes,
        ):
            raise ValueError(
                "Checkpoint encoding differs from the active configuration."
            )
        geometry = self.prefix_geometry(prefix)
        if geometry["complete"]:
            raise ValueError("This tree is already complete; no next node exists.")
        net.eval()
        destination = next(net.parameters()).device
        return net(self.prefix_batch(prefix, destination))[0, -1].float().cpu()

    @torch.no_grad()
    def predict_next(self, prefix, top_k=5, net=None):
        if not 1 <= top_k <= self.VOCAB_SIZE:
            raise ValueError("top_k must be within the node vocabulary.")
        geometry = self.prefix_geometry(prefix)
        probabilities = self.next_logits(prefix, net).softmax(-1)
        values, tokens = probabilities.topk(top_k)
        candidates = []
        for probability, token in zip(values.tolist(), tokens.tolist()):
            record = self.id_record(token)
            candidates.append(
                dict(
                    value=record.value,
                    child_count=record.child_count,
                    probability=probability,
                    fits_node_budget=len(prefix)
                    + geometry["slots"]
                    + record.child_count
                    <= self.cfg.max_nodes,
                )
            )
        return dict(
            parent_preorder_index=geometry["next_parent"],
            depth=geometry["next_depth"],
            sibling_position=geometry["next_sibling"],
            candidates=candidates,
        )

    @torch.no_grad()
    def generate(
        self,
        prefix=(),
        max_nodes=None,
        constrained=True,
        temperature=1.0,
        seed=42,
        net=None,
    ):
        limit = self.cfg.max_nodes if max_nodes is None else max_nodes
        if type(limit) is not int or not 1 <= limit <= self.cfg.max_nodes:
            raise ValueError(
                "max_nodes must be an integer within the configured capacity."
            )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        records = list(prefix)
        geometry = self.prefix_geometry(records)
        if len(records) + geometry["slots"] > limit:
            raise ValueError(
                "The supplied prefix cannot be completed within max_nodes."
            )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        counts = torch.arange(self.VOCAB_SIZE) // self.cfg.n_values
        while not geometry["complete"]:
            logits = self.next_logits(records, net) / temperature
            if constrained:
                feasible = counts <= limit - len(records) - geometry["slots"]
                logits = logits.masked_fill(~feasible, -torch.inf)
            token = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            records.append(self.id_record(token))
            try:
                geometry = self.prefix_geometry(records)
            except ValueError as error:
                return dict(
                    records=records,
                    valid=False,
                    reason=str(error),
                    constrained=constrained,
                )
            if len(records) + geometry["slots"] > limit:
                return dict(
                    records=records,
                    valid=False,
                    reason="Declared children exceed the node budget.",
                    constrained=constrained,
                )
        return dict(
            records=records,
            valid=True,
            reason="All child slots filled.",
            constrained=constrained,
        )

    def serializable_generation(self, result):
        return {**result, "records": [asdict(r) for r in result["records"]]}

    def generated_feature_score(self, records, prefix_length):
        geometry = self.prefix_geometry(records, require_complete=True)
        indices = range(max(1, prefix_length), len(records))
        correct = sum(
            (
                records[i].value
                == (records[geometry["parents"][i]].value + geometry["siblings"][i])
                % self.cfg.n_values
                for i in indices
            )
        )
        count = max(0, len(records) - max(1, prefix_length))
        return (correct, count)

    def evaluate_generation(self):
        rng = np.random.default_rng(self.cfg.seed + 100)
        selected = rng.choice(
            len(self.examples["test"]),
            min(self.cfg.generation_trials, len(self.examples["test"])),
            replace=False,
        ).tolist()
        metrics, saved = ({}, {})
        for constrained in (False, True):
            name = "constrained" if constrained else "unconstrained"
            valid, correct, count = (0, 0, 0)
            samples = []
            for trial, index in enumerate(selected):
                row = self.examples["test"][index]
                prefix = row["records"][: min(8, len(row["records"]) - 1)]
                result = self.generate(
                    prefix, constrained=constrained, seed=self.cfg.seed + trial
                )
                assert result["records"][: len(prefix)] == prefix
                if result["valid"]:
                    valid += 1
                    assert (
                        self.encode_tree(self.decode_records(result["records"]))
                        == result["records"]
                    )
                    c, n = self.generated_feature_score(result["records"], len(prefix))
                    correct += c
                    count += n
                samples.append(
                    dict(
                        shape_id=row["shape_id"],
                        root_value=row["root_value"],
                        prefix_length=len(prefix),
                        **self.serializable_generation(result),
                    )
                )
            metrics[name] = dict(
                trials=len(selected),
                valid_completions=valid,
                structural_validity=valid / len(selected),
                generated_feature_accuracy_on_valid_completions=(
                    correct / count if count else None
                ),
                scored_generated_nodes=count,
            )
            saved[name] = samples
            print(
                f"{name}: {valid}/{len(selected)} valid completions; generated value accuracy on valid completions: {self.percent(correct / count if count else None)}"
            )
        assert metrics["constrained"]["valid_completions"] == len(selected)
        self.checks["constrained_generation_validity"] = True
        save_json(self.out / "generation_samples.json", saved)
        return (metrics, saved)

    def draw_records(self, records, prefix_length=0, title="Encoded tree"):
        geometry = self.prefix_geometry(records, require_complete=True)
        children = [[] for _ in records]
        for i, parent in enumerate(geometry["parents"]):
            if parent >= 0:
                children[parent].append(i)
        leaves = [i for i, kids in enumerate(children) if not kids]
        x = {node: float(rank) for rank, node in enumerate(leaves)}
        for i in reversed(range(len(records))):
            if children[i]:
                x[i] = float(np.mean([x[child] for child in children[i]]))
        fig, axis = plt.subplots(
            figsize=(
                min(18, max(9, len(leaves) * 0.8)),
                max(3.5, 1.1 * (max(geometry["depths"]) + 1)),
            )
        )
        for i, parent in enumerate(geometry["parents"]):
            if parent >= 0:
                axis.plot(
                    [x[parent], x[i]],
                    [-geometry["depths"][parent], -geometry["depths"][i]],
                    color="#b8c4ce",
                    zorder=1,
                )
        for i, record in enumerate(records):
            color = (
                "#d8eaff"
                if i < prefix_length
                else "#ffd49b" if i == prefix_length else "#edf0f2"
            )
            axis.text(
                x[i],
                -geometry["depths"][i],
                f"v={record.value}\nk={record.child_count}",
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.3", facecolor=color, edgecolor="#687886"
                ),
                zorder=2,
            )
        axis.set_title(title, pad=20)
        axis.set_xlim(-0.8, max(x.values()) + 0.8)
        axis.set_ylim(-max(geometry["depths"]) - 0.7, 0.7)
        axis.axis("off")
        fig.tight_layout()
        return fig

    def load_checkpoint(self, path=None, destination="cpu"):
        path = self.out / "checkpoint.pt" if path is None else path
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("Unsupported checkpoint version.")
        config = Config(**checkpoint["config"])
        loaded = NodeTransformer(config).to(destination)
        loaded.load_state_dict(checkpoint["state_dict"])
        loaded.eval()
        return loaded

    @torch.no_grad()
    def check_inference_and_reload(self):
        row = self.examples["test"][0]
        cut = min(4, len(row["records"]) - 1)
        batch = self.to_device(self.collate([row]))
        expected = self.model(batch)[0, cut].float().cpu()
        actual = self.next_logits(row["records"][:cut])
        torch.testing.assert_close(actual, expected, rtol=0.0001, atol=1e-05)
        loaded = self.load_checkpoint()
        torch.testing.assert_close(
            self.next_logits(row["records"][:cut], loaded),
            actual,
            rtol=0.0001,
            atol=0.0001,
        )
        prediction = self.predict_next(row["records"][:cut], net=loaded)
        assert all((0 <= item["probability"] <= 1 for item in prediction["candidates"]))
        assert (
            sum((item["probability"] for item in prediction["candidates"])) <= 1 + 1e-06
        )
        self.expect_value_error(lambda: self.predict_next(row["records"], net=loaded))
        self.expect_value_error(
            lambda: self.generate([NodeRecord(0, 3)], max_nodes=2, net=loaded)
        )
        complete = [NodeRecord(0, 0)]
        assert self.generate(complete, net=loaded)["records"] == complete
        for seed in (1, 2, 3):
            result = self.generate([], max_nodes=8, seed=seed, net=loaded)
            assert result["valid"] and len(result["records"]) <= 8
        self.checks["training_inference_alignment"] = True
        self.checks["checkpoint_reload_cpu"] = True
        print(
            "PASS: inference alignment, CPU checkpoint reload, and generation boundaries."
        )

    def prepare_data(self):
        self.pool = self.load_subtree_pool(
            self.cfg.source_path,
            max(1024, self.cfg.n_shapes),
            self.cfg.seed,
            self.cfg.min_nodes,
            self.cfg.max_nodes,
        )
        self.shapes = self.pool["shapes"][: self.cfg.n_shapes]
        print(
            f"Experiment: {len(self.shapes)} shapes; sizes {min((len(s['counts']) for s in self.shapes))}–{max((len(s['counts']) for s in self.shapes))} nodes."
        )

    def prepare_examples(self):
        rng = np.random.default_rng(self.cfg.seed)
        order = rng.permutation(len(self.shapes)).tolist()
        train_end, valid_end = (int(0.8 * len(order)), int(0.9 * len(order)))
        split_indices = dict(
            train=order[:train_end],
            validation=order[train_end:valid_end],
            test=order[valid_end:],
        )
        self.split_ids = {
            name: [self.shapes[i]["shape_id"] for i in indices]
            for name, indices in split_indices.items()
        }
        self.examples = {
            name: [
                dict(
                    shape_id=self.shapes[i]["shape_id"],
                    root_value=value,
                    records=self.assign_features(self.shapes[i]["counts"], value),
                )
                for i in indices
                for value in range(self.cfg.n_values)
            ]
            for name, indices in split_indices.items()
        }
        save_json(self.out / "split_ids.json", self.split_ids)
        save_json(
            self.out / "examples.json",
            {
                name: [
                    dict(
                        shape_id=row["shape_id"],
                        root_value=row["root_value"],
                        token_ids=[self.record_id(r) for r in row["records"]],
                    )
                    for row in rows
                ]
                for name, rows in self.examples.items()
            },
        )
        save_json(
            self.out / "encoding.json",
            dict(
                version=1,
                n_values=self.cfg.n_values,
                max_nodes=self.cfg.max_nodes,
                vocab_size=self.VOCAB_SIZE,
                bos_id=self.BOS_ID,
                pad_id=self.PAD_ID,
                node_id_formula="n_values * child_count + value",
                traversal="preorder",
            ),
        )
        for name in self.examples:
            print(
                f"{name:10s}: {len(self.split_ids[name]):4d} shapes, {len(self.examples[name]):5d} trees"
            )
        self.loaders = {
            name: self.make_loader(rows, name == "train")
            for name, rows in self.examples.items()
        }

    def verify_data(self):
        self.checks = {}
        self.check_data()

    def build_model(self):
        seed_everything(self.cfg.seed)
        self.model = NodeTransformer(self.cfg).to(self.device)
        print(f"Parameters: {sum((p.numel() for p in self.model.parameters())):,}")

    def verify_optimizer(self):
        self.check_model_math(self.model)
        self.check_tiny_overfit()
        save_json(self.out / "checks.json", self.checks)

    def train(self):
        self.history, self.best_epoch, self.training_seconds = self.fit(self.model)

    def report_evaluation(self):
        self.baseline_tables = self.fit_baselines(self.examples["train"])
        self.test_metrics = {
            "transformer": self.evaluate(self.model, self.loaders["test"])
        }
        self.test_metrics.update(
            {
                name: self.evaluate_baseline(
                    name, self.baseline_tables, self.loaders["test"]
                )
                for name in ("unigram", "bigram")
            }
        )
        transformer_metrics = self.test_metrics["transformer"]
        joint_pass = all(
            (
                transformer_metrics["joint_nll"] < self.test_metrics[name]["joint_nll"]
                for name in ("unigram", "bigram")
            )
        )
        return_pass = transformer_metrics["return_nodes"] > 0 and all(
            (
                transformer_metrics["return_feature_accuracy"]
                > self.test_metrics[name]["return_feature_accuracy"]
                for name in ("unigram", "bigram")
            )
        )
        self.learning_check = dict(
            joint_nll_beats_both=joint_pass,
            return_feature_accuracy_beats_both=return_pass,
            criterion_met=joint_pass and return_pass,
            eligible_for_full_experiment_claim=self.profile == "full",
        )
        self.results = dict(
            profile=self.profile,
            test_shapes=len(self.split_ids["test"]),
            best_epoch=self.best_epoch,
            training_seconds=self.training_seconds,
            metrics=self.test_metrics,
            learning_check=self.learning_check,
        )
        save_json(self.out / "metrics.json", self.results)
        table = [
            "| Method | Joint NLL ↓ | Joint accuracy | Value accuracy (non-root) | Child-count accuracy | Value accuracy (subtree returns) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, metrics in self.test_metrics.items():
            table.append(
                f"| {name} | {metrics['joint_nll']:.4f} | {self.percent(metrics['joint_accuracy'])} | {self.percent(metrics['non_root_feature_accuracy'])} | {self.percent(metrics['child_count_accuracy'])} | {self.percent(metrics['return_feature_accuracy'])} |"
            )
        print("\n".join(table))
        print(
            f"Test set: {len(self.split_ids['test'])} shape groups; {transformer_metrics['nodes']:,} node targets; {transformer_metrics['return_nodes']:,} subtree-return targets."
        )
        print(
            "Learning criterion:",
            "PASS" if self.learning_check["criterion_met"] else "NOT MET",
        )
        if self.profile == "smoke":
            print(
                "SMOKE PROFILE: this only checks execution; it does not establish the full learning result."
            )
        epochs = [h["epoch"] for h in self.history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(
            epochs, [h["train_joint_nll"] for h in self.history], label="Train"
        )
        axes[0].plot(
            epochs,
            [h["validation"]["joint_nll"] for h in self.history],
            label="Validation",
        )
        axes[0].set(
            xlabel="Epoch", ylabel="Joint next-node NLL", title="Checkpoint selection"
        )
        for key, label in (
            ("non_root_feature_accuracy", "All non-root nodes"),
            ("return_feature_accuracy", "Subtree returns"),
        ):
            axes[1].plot(
                epochs, [h["validation"][key] for h in self.history], label=label
            )
        axes[1].set(
            xlabel="Epoch",
            ylabel="Validation value accuracy",
            ylim=(0, 1),
            title="Feature learning",
        )
        for axis in axes:
            axis.axvline(
                self.best_epoch, color="gray", linestyle="--", label="Selected epoch"
            )
            axis.legend()
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(self.out / "training_curves.png", dpi=150)
        plt.close(fig)

    def report_generation(self):
        self.generation_metrics, self.generation_samples = self.evaluate_generation()
        self.results["generation"] = self.generation_metrics
        save_json(self.out / "metrics.json", self.results)

    def show_prediction(self):
        demo_rows = [
            row
            for row in self.examples["test"]
            if row["root_value"] == 3 % self.cfg.n_values
            and any(self.prefix_geometry(row["records"])["returns"])
        ]
        demo = min(
            demo_rows or self.examples["test"], key=lambda row: len(row["records"])
        )
        geometry = self.prefix_geometry(demo["records"])
        prefix_length = (
            geometry["returns"].index(True)
            if any(geometry["returns"])
            else min(8, len(demo["records"]) - 1)
        )
        prefix = demo["records"][:prefix_length]
        prediction = self.predict_next(prefix)
        actual = demo["records"][prefix_length]
        print("Input prefix:", [(r.value, r.child_count) for r in prefix])
        print(
            "Next attachment (preorder index, not a learned identity):",
            prediction["parent_preorder_index"],
        )
        print("Observed next record:", actual)
        candidate_table = [
            "| Value | Children | Probability | Fits node budget |",
            "|---:|---:|---:|---|",
        ]
        for candidate in prediction["candidates"]:
            candidate_table.append(
                f"| {candidate['value']} | {candidate['child_count']} | {candidate['probability']:.4f} | {candidate['fits_node_budget']} |"
            )
        print("\n".join(candidate_table))
        fig = self.draw_records(
            demo["records"],
            prefix_length,
            "Held-out subtree: blue prefix → orange next node",
        )
        fig.savefig(self.out / "held_out_subtree.png", dpi=150)
        plt.close(fig)
        continuation = self.generate(prefix, seed=self.cfg.seed + 200)
        print(
            "Generated continuation:",
            [(r.value, r.child_count) for r in continuation["records"][prefix_length:]],
        )
        fig = self.draw_records(
            continuation["records"],
            prefix_length,
            "Sampled continuation of the same prefix",
        )
        fig.savefig(self.out / "generated_subtree.png", dpi=150)
        plt.close(fig)
        save_json(
            self.out / "prediction_example.json",
            dict(
                shape_id=demo["shape_id"],
                prefix=[asdict(r) for r in prefix],
                observed_next=asdict(actual),
                prediction=prediction,
                continuation=self.serializable_generation(continuation),
            ),
        )

    def verify_reload(self):
        self.check_inference_and_reload()
        save_json(self.out / "checks.json", self.checks)
        print(f"Saved {self.profile} experiment to {self.out.resolve()}")
        print(
            "Full learning criterion:",
            (
                "not assessed by the smoke profile"
                if self.profile == "smoke"
                else (
                    "PASS"
                    if self.learning_check["criterion_met"]
                    else "NOT MET — see metrics and curves"
                )
            ),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_suffix(".yaml"),
        help="YAML settings (default: newick_node_lm.yaml beside this script)",
    )
    parser.add_argument(
        "--profile", choices=("full", "smoke"), help="Override the YAML profile"
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), help="Override the YAML device"
    )
    parser.add_argument(
        "--check-data", action="store_true", help="Extract and validate data, then exit"
    )
    args = parser.parse_args(argv)
    try:
        config, profile = load_config(args.config, args.profile, args.device)
        Experiment(config, profile).run(check_data_only=args.check_data)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
