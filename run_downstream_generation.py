#!/usr/bin/env python3
"""
Load a FedLLM SeqCLS checkpoint (NewsClassifierModel), transfer backbone to CausalLM, run generate() on probes.

Example:
  python run_downstream_generation.py \\
    --checkpoint results/global_checkpoint \\
    --probes data/financial_probes.json \\
    --output results/downstream_gen.jsonl

  python run_downstream_generation.py \\
    --checkpoint results/global_checkpoint \\
    --compare-checkpoint results_baseline/global_checkpoint \\
    --output results/downstream_compare.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from decoder_adapters import resolve_adapter
from models import NewsClassifierModel


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_paths(checkpoint: Path) -> Tuple[Path, Path]:
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        return checkpoint / "global_model.pt", checkpoint / "checkpoint_metadata.json"
    return checkpoint, checkpoint.parent / "checkpoint_metadata.json"


def _load_metadata(pack: Dict[str, Any], meta_path: Path) -> Dict[str, Any]:
    meta = pack.get("metadata")
    if meta is None and meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    if not meta:
        raise ValueError("Missing metadata: expected 'metadata' in .pt or checkpoint_metadata.json next to checkpoint.")
    return meta


def build_news_classifier(meta: Dict[str, Any]) -> NewsClassifierModel:
    use_lora = bool(meta.get("use_lora", False))
    kw: Dict[str, Any] = {
        "model_name": meta["model_name"],
        "num_labels": int(meta["num_labels"]),
        "use_lora": use_lora,
    }
    if use_lora:
        kw["lora_r"] = meta.get("lora_r", 16)
        kw["lora_alpha"] = meta.get("lora_alpha", 32)
        kw["lora_dropout"] = meta.get("lora_dropout", 0.1)
        tm = meta.get("lora_target_modules")
        kw["lora_target_modules"] = None if tm is None else list(tm)
    return NewsClassifierModel(**kw)


def load_news_classifier(pt_path: Path, meta_path: Path) -> Tuple[NewsClassifierModel, Dict[str, Any]]:
    pack = _torch_load(pt_path)
    meta = _load_metadata(pack, meta_path)
    model = build_news_classifier(meta)
    incompatible = model.load_state_dict(pack["state_dict"], strict=False)
    if incompatible.missing_keys:
        print(f"  Warning: missing_keys when loading NewsClassifierModel: {len(incompatible.missing_keys)} keys")
    if incompatible.unexpected_keys:
        print(f"  Warning: unexpected_keys when loading NewsClassifierModel: {len(incompatible.unexpected_keys)} keys")
    model.eval()
    return model, meta


def build_causal_lm(base_model_name: str, device: torch.device) -> torch.nn.Module:
    causal = AutoModelForCausalLM.from_pretrained(base_model_name)
    causal.to(device)
    causal.eval()
    return causal


def build_tokenizer(base_model_name: str):
    tok = AutoTokenizer.from_pretrained(base_model_name)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok


def default_prompt(news_text: str, question: str) -> str:
    return f"### News\n{news_text}\n\n### Task\n{question}\n\n### Answer\n"


def load_probes(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Probes JSON must be a list of objects with id, news_text, question")
    return data


def generate_completion(
    causal,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = max(temperature, 1e-5)
    else:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        out = causal.generate(**inputs, **gen_kw)
    inp_len = inputs["input_ids"].shape[1]
    gen_ids = out[0, inp_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def run_for_checkpoint(
    news: NewsClassifierModel,
    meta: Dict[str, Any],
    base_model_name: str,
    probes: List[Dict[str, Any]],
    device: torch.device,
    prompt_fn,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> List[str]:
    adapter = resolve_adapter(meta["model_name"])
    causal = build_causal_lm(base_model_name, device)
    tokenizer = build_tokenizer(base_model_name)
    adapter.transfer_backbone(news.model, causal)

    completions = []
    for p in probes:
        prompt = prompt_fn(p["news_text"], p["question"])
        text = generate_completion(
            causal, tokenizer, prompt, device, max_new_tokens, do_sample, temperature
        )
        completions.append(text)
    del causal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return completions


def main() -> None:
    parser = argparse.ArgumentParser(description="Downstream causal LM generation from Fed SeqCLS checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to global_model.pt or directory containing global_model.pt + checkpoint_metadata.json",
    )
    parser.add_argument(
        "--compare-checkpoint",
        type=Path,
        default=None,
        help="Optional second checkpoint (e.g. clean baseline) for paired completions on the same probes",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=Path("data/financial_probes.json"),
        help="JSON list of {id, news_text, question}",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override HF model id for CausalLM/tokenizer (default: model_name from checkpoint metadata)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--greedy", action="store_true", help="Disable sampling (greedy decode)")
    args = parser.parse_args()

    device = torch.device(args.device)
    pt_path, meta_path = resolve_checkpoint_paths(args.checkpoint)
    if not pt_path.is_file():
        print(f"Checkpoint file not found: {pt_path}", file=sys.stderr)
        sys.exit(1)

    probes = load_probes(args.probes)
    news_a, meta_a = load_news_classifier(pt_path, meta_path)
    base_name = args.base_model or meta_a["model_name"]
    if args.base_model and args.base_model != meta_a["model_name"]:
        print(
            f"  Note: --base-model {args.base_model!r} overrides metadata model_name {meta_a['model_name']!r}; "
            "ensure architecture matches the saved weights."
        )

    do_sample = not args.greedy

    def prompt_fn(news_text: str, question: str) -> str:
        return default_prompt(news_text, question)

    print(f"  Primary checkpoint: {pt_path}")
    completions_a = run_for_checkpoint(
        news_a,
        meta_a,
        base_name,
        probes,
        device,
        prompt_fn,
        args.max_new_tokens,
        do_sample,
        args.temperature,
    )

    completions_b: Optional[List[str]] = None
    if args.compare_checkpoint:
        pt_b, meta_b = resolve_checkpoint_paths(args.compare_checkpoint)
        if not pt_b.is_file():
            print(f"Compare checkpoint not found: {pt_b}", file=sys.stderr)
            sys.exit(1)
        news_b, meta_b = load_news_classifier(pt_b, meta_b)
        if meta_b["model_name"] != meta_a["model_name"]:
            print(
                "  Warning: compare checkpoint has different model_name than primary; "
                "verify adapters and base-model compatibility.",
                file=sys.stderr,
            )
        print(f"  Compare checkpoint: {pt_b}")
        completions_b = run_for_checkpoint(
            news_b,
            meta_b,
            base_name,
            probes,
            device,
            prompt_fn,
            args.max_new_tokens,
            do_sample,
            args.temperature,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, p in enumerate(probes):
            row = {
                "probe_id": p.get("id", i + 1),
                "news_text": p["news_text"],
                "question": p["question"],
                "completion_primary": completions_a[i],
            }
            if completions_b is not None:
                row["completion_compare"] = completions_b[i]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Wrote {len(probes)} lines to {args.output}")


if __name__ == "__main__":
    main()
