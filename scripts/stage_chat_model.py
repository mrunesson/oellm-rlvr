#!/usr/bin/env python3
"""Stage a base checkpoint for chat-format RL: weights from one Hugging Face repo, the tokenizer from another, and a
chat template file. Like OpenEuroLLM post-training (build_tokenizer / align_generation_eos), the turn terminator is
read off a rendered conversation and becomes the tokenizer's eos and the first generation eos id (the model's own eos
stays second), so vLLM stops on what the template ends a turn with.

    python scripts/stage_chat_model.py --model openeurollm/prelude-base --model-revision <sha> \
      --tokenizer openeurollm/tokenizer-256k-reason --tokenizer-revision <sha> \
      --chat-template qwen3.jinja --output <dir>

Needs network (Hugging Face) and transformers + huggingface_hub; run it with the SkyRL venv's python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

WEIGHT_PATTERNS = ["config.json", "*.safetensors", "model.safetensors.index.json"]
TOKENIZER_PATTERNS = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model"]


def end_token_of(tokenizer) -> str | None:
    """The added token a rendered conversation ends on (post-training's infer_end_token_from_render)."""
    probe = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    tail = tokenizer.apply_chat_template(probe, tokenize=False).rstrip("\n")
    matches = [token for token in tokenizer.get_added_vocab() if token and tail.endswith(token)]
    return max(matches, key=len) if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output
    snapshot_download(args.model, revision=args.model_revision, local_dir=output, allow_patterns=WEIGHT_PATTERNS)
    tokenizer_dir = output / ".tokenizer-source"
    snapshot_download(args.tokenizer, revision=args.tokenizer_revision, local_dir=tokenizer_dir,
                      allow_patterns=TOKENIZER_PATTERNS)

    template = args.chat_template.read_text()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.chat_template = template
    end_token = end_token_of(tokenizer)
    if end_token is None:
        raise SystemExit("the chat template does not end a turn on an added token; refusing to guess the eos")
    config = AutoConfig.from_pretrained(output)
    if len(tokenizer) > config.vocab_size:
        raise SystemExit(f"tokenizer has {len(tokenizer)} tokens, the model's embedding only {config.vocab_size}")
    model_eos = config.eos_token_id
    tokenizer.eos_token = end_token
    tokenizer.save_pretrained(output)
    # transformers 5 writes chat_template.jinja; write it ourselves too so the file is byte-identical to the source.
    (output / "chat_template.jinja").write_text(template)

    eos_ids = [tokenizer.eos_token_id] + [i for i in ([model_eos] if isinstance(model_eos, int) else model_eos or [])
                                          if i != tokenizer.eos_token_id]
    generation_config = {"bos_token_id": tokenizer.bos_token_id, "eos_token_id": eos_ids,
                         "pad_token_id": tokenizer.pad_token_id}
    (output / "generation_config.json").write_text(json.dumps(generation_config, indent=2) + "\n")

    messages = [{"role": "user", "content": "Natalia sold 48 clips in April. How many in May if half as many?"}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    source = {
        "model": args.model, "model_revision": args.model_revision,
        "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "eos_token": end_token, "generation_eos_token_ids": eos_ids,
    }
    (output / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(json.dumps(source, indent=2))
    print("rendered prompt:", repr(rendered))
    print("prompt ids:", ids)
    for token in ("<|im_start|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise SystemExit(f"{token} is not a single token in {args.tokenizer}")
    if ids[0] != tokenizer.convert_tokens_to_ids("<|im_start|>"):
        raise SystemExit("the prompt does not start with <|im_start|>")
    print(f"staged {output}")


if __name__ == "__main__":
    main()
