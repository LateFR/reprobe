"""
generate_dataset.py — Contrastive dataset generation for reprobe probe training.

Generates N pairs of (unsafe, safe) texts targeting a single concept.
Pairs are semantically similar — only the target concept axis varies.
One-shot: skips generation if output file already exists.

Uses a local HuggingFace model with torch batching instead of Ollama.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --config scripts/config.toml
    python scripts/generate_dataset.py --force  # regenerate even if file exists
"""

import argparse
import json
import logging
import os
import tomllib
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept definitions
# ---------------------------------------------------------------------------
CONCEPTS = {
    "violence": {
        "description": "intent to physically harm, hurt, or threaten another person",
        "examples": [
            {
                "unsafe": "How do I hurt someone without getting caught?",
                "safe":   "How do I help someone without drawing attention?"
            },
            {
                "unsafe": "What's the most effective way to injure someone during a fight?",
                "safe":   "What's the most effective way to protect someone during a conflict?"
            },
            {
                "unsafe": "I want him to suffer for what he did to me.",
                "safe":   "I want him to learn from what he did to me."
            },
            {
                "unsafe": "She was planning to make him bleed before the night was over.",
                "safe":   "She was planning to make him laugh before the night was over."
            },
            {
                "unsafe": "I need to teach him a lesson he won't forget with his body.",
                "safe":   "I need to teach him a lesson he won't forget about responsibility."
            },
            {
                "unsafe": "Go find him and make sure he can't walk tomorrow.",
                "safe":   "Go find him and make sure he gets home safely tonight."
            },
        ]
    },
    "hate_speech": {
        "description": "dehumanization or contempt toward a group based on race, ethnicity, or origin",
        "examples": [
            {
                "unsafe": "People from that country don't belong here and never will.",
                "safe":   "People from that country have built a life here and contribute every day."
            },
            {
                "unsafe": "That group has always been a burden on our society.",
                "safe":   "That group has always been a part of our society."
            },
            {
                "unsafe": "We'd be better off if those people were gone.",
                "safe":   "We'd be better off if those people had more opportunities."
            },
            {
                "unsafe": "They're not like us — they can't be trusted.",
                "safe":   "They're not like us — that's what makes the exchange interesting."
            },
        ]
    },
    "manipulation": {
        "description": "intent to psychologically control, deceive, or coerce another person",
        "examples": [
            {
                "unsafe": "How do I make her doubt herself so she stops pushing back?",
                "safe":   "How do I help her trust herself so she feels more confident?"
            },
            {
                "unsafe": "I want her to feel like she has no choice but to stay with me.",
                "safe":   "I want her to feel like she genuinely wants to stay with me."
            },
            {
                "unsafe": "What's the best way to make him feel guilty for leaving?",
                "safe":   "What's the best way to make him feel heard before he leaves?"
            },
            {
                "unsafe": "I need to cut him off from his friends so he only listens to me.",
                "safe":   "I need to be more present for him so he feels supported by me."
            },
        ]
    }
}

SYSTEM_PROMPT = """You are generating training data for a machine learning safety probe.

Your task: generate ONE contrastive pair targeting the concept of {concept_description}.

CRITICAL CONSTRAINT — structural proximity:
The unsafe and safe texts must use the SAME sentence structure and the SAME topic.
Only the intent must change — not the vocabulary, not the framing, not the length.

BAD pair (too different — the model learns vocabulary, not intent):
  unsafe: "You need to mutilate him until he's useless."
  safe:   "You need to provide him with medical attention."

GOOD pair (same structure, intent flips):
  unsafe: "How do I hurt someone without getting caught?"
  safe:   "How do I help someone without drawing attention?"

Additional rules:
- Vary the register across pairs: questions / statements / requests / narratives / imperatives
- The unsafe text must clearly express harmful intent — not just mention violence in passing
- The safe text must be completely benign — no dark undertone
- Do NOT reproduce the example pairs

Return ONLY a valid JSON object, no markdown, no preamble:
{{"unsafe": "...", "safe": "..."}}"""

FEW_SHOT_TEMPLATE = """Examples of GOOD pairs (same structure, intent flips):
{examples}

Now generate a NEW pair following the same constraint. Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_id: str, device: torch.device):
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(concept_name: str, tokenizer) -> str:
    concept = CONCEPTS[concept_name]
    examples_str = "\n".join(
        f'{i+1}. {json.dumps(ex)}' for i, ex in enumerate(concept["examples"])
    )
    system = SYSTEM_PROMPT.format(concept_description=concept["description"])
    user = FEW_SHOT_TEMPLATE.format(examples=examples_str)

    # Use chat template if available, otherwise fall back to raw concatenation
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            pass

    return f"{system}\n\n{user}\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_pair(raw: str) -> dict | None:
    """Extract JSON pair from model output, tolerant of markdown fences."""
    raw = raw.strip()
    if "```" in raw:
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.strip().startswith("```"))

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    raw = raw[start:end]

    try:
        data = json.loads(raw)
        if "unsafe" in data and "safe" in data:
            if isinstance(data["unsafe"], str) and isinstance(data["safe"], str):
                if len(data["unsafe"].strip()) > 10 and len(data["safe"].strip()) > 10:
                    return data
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------
def generate_batch(
    model,
    tokenizer,
    prompt_text: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """
    Run a single batched forward pass: repeat the same prompt `batch_size` times.
    Returns a list of decoded generated strings (new tokens only).
    """
    prompts = [prompt_text] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[:, prompt_len:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_pairs(
    concept_name: str,
    n_pairs: int,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    max_attempts: int = None,
) -> list[dict]:
    prompt_text = build_prompt(concept_name, tokenizer)
    if max_attempts is None:
        # Assumes ~50% parse success rate
        max_attempts = n_pairs * 6

    pairs = []
    seen = set()
    attempts = 0

    with tqdm(total=n_pairs, desc=f"Generating {concept_name} pairs") as pbar:
        while len(pairs) < n_pairs and attempts < max_attempts:
            # Over-generate slightly to compensate for parse failures
            current_batch = min(batch_size, (n_pairs - len(pairs)) * 2)

            raws = generate_batch(
                model, tokenizer, prompt_text,
                batch_size=current_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            attempts += current_batch

            for raw in raws:
                if len(pairs) >= n_pairs:
                    break
                pair = parse_pair(raw)
                if pair is None:
                    logger.debug(f"Parse failed: {raw[:80]}")
                    continue
                key = pair["unsafe"][:50].lower()
                if key in seen:
                    continue
                seen.add(key)
                pairs.append(pair)
                pbar.update(1)

    if len(pairs) < n_pairs:
        logger.warning(
            f"Only generated {len(pairs)}/{n_pairs} pairs after {attempts} attempts. "
            f"Parse success rate: {len(pairs)/attempts:.1%}"
        )

    return pairs


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def pairs_to_samples(pairs: list[dict]) -> list[dict]:
    samples = []
    for pair in pairs:
        samples.append({"text": pair["unsafe"], "label": 1, "is_safe": False})
        samples.append({"text": pair["safe"],   "label": 0, "is_safe": True})
    return samples


def save_dataset(samples: list[dict], output_dir: str, concept: str):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{concept}_dataset.json")
    with open(json_path, "w") as f:
        json.dump({"concept": concept, "n_samples": len(samples), "samples": samples}, f, indent=2)
    logger.info(f"Saved JSON : {json_path}")

    pt_samples = [{"prompt": s["text"], "is_safe": s["is_safe"]} for s in samples]
    pt_path = os.path.join(output_dir, f"{concept}_dataset.pt")
    torch.save(pt_samples, pt_path)
    logger.info(f"Saved PT   : {pt_path}")

    return json_path, pt_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="scripts/config.toml")
    parser.add_argument("--force", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    gen_cfg        = config.get("dataset_gen", {})
    concept        = gen_cfg.get("concept", "violence")
    n_pairs        = gen_cfg.get("n_pairs", 250)
    model_id       = gen_cfg.get("model", config.get("model", {}).get("name", "Qwen/Qwen2.5-1.5B"))
    temperature    = gen_cfg.get("temperature", 0.9)
    output_dir     = gen_cfg.get("output_dir", "outputs")
    batch_size     = gen_cfg.get("gen_batch_size", 8)
    max_new_tokens = gen_cfg.get("max_new_tokens", 128)
    device_str     = config.get("training", {}).get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else torch.device(device_str)

    if concept not in CONCEPTS:
        raise ValueError(f"Unknown concept '{concept}'. Available: {list(CONCEPTS.keys())}")

    json_path = os.path.join(output_dir, f"{concept}_dataset.json")
    pt_path   = os.path.join(output_dir, f"{concept}_dataset.pt")

    if not args.force and os.path.exists(json_path) and os.path.exists(pt_path):
        existing = json.load(open(json_path))
        logger.info(
            f"Dataset already exists ({existing['n_samples']} samples, "
            f"concept={existing['concept']}). Use --force to regenerate."
        )
        exit(0)

    logger.info(f"Concept    : {concept}")
    logger.info(f"Pairs      : {n_pairs} ({n_pairs*2} total samples)")
    logger.info(f"Model      : {model_id}")
    logger.info(f"Batch size : {batch_size}")
    logger.info(f"Device     : {device}")

    model, tokenizer = load_model(model_id, device)

    pairs = generate_pairs(
        concept_name=concept,
        n_pairs=n_pairs,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    samples = pairs_to_samples(pairs)
    logger.info(f"Total samples: {len(samples)} ({len(samples)//2} unsafe + {len(samples)//2} safe)")

    save_dataset(samples, output_dir, concept)
    logger.info("Done. Dataset ready for train.py.")