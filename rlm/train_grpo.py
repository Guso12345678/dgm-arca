"""Phase 1, step 3: reinforcement learning with verifiable rewards using GRPO.

Starting point: the SFT adapter from ``train_sft.py`` (or the base model, if you want to
reproduce the R1-Zero experiment and see what happens without cold start). Output: your
final reasoning model, ``rlm/weights/final_rlm_lora``.

Run::

    uv run python -m rlm.train_grpo --data rlm/data/train.jsonl --init-adapter rlm/weights/sft_lora

The smoke test (``smoke/smoke_grpo.py``) is the minimal version of this script on GSM8K.
Here you add what makes it *yours*:

1. Your domain dataset (``rlm/data.py::load_domain_dataset``) and your verifier.
2. A third, domain-specific reward (``domain_reward`` below). Think about what a good
   answer looks like for your user beyond being correct: language, units, length,
   citing a source, respecting a schema. Justify it in EXPERIMENTS.md and, if you use
   ``reward_weights``, justify those too.
3. The hyper-parameters. Group size, completion length, learning rate and KL
   coefficient all change what the model learns. Change one thing at a time and log it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

from rlm.data import load_domain_dataset, load_gsm8k
from rlm.rewards import (
    FORMAT_PATTERN,
    NUMBER_PATTERN,
    _completion_text,
    accuracy_reward,
    format_reward,
)

STRICT_ANSWER_PATTERN = re.compile(r"^-?\d+\.\d{2}$")
MIN_CALC_NUMBERS = 3  # distinct numbers in <think> that count as a real derivation

def domain_reward(prompts: Sequence, completions: Sequence, **kwargs) -> list[float]:
    """Tu turno: a reward that captures what "good" means in your domain.

    Same signature as the other rewards: one float per completion, dataset columns arrive
    in ``kwargs``. Keep it deterministic and cheap. Examples from class and beyond:

    * language consistency: fraction of words in the answer's language (DeepSeek-R1);
    * length shaping: penalise thinking that exceeds a budget, or reward concise answers;
    * a units check for physical quantities; a schema check for structured answers;
    * a code-execution reward: run the unit tests of the problem (only in a sandbox!).

    
    """
    """Recompensa propia del dominio de crédito al consumo: premia que el
    modelo DERIVE la respuesta en vez de alucinarla, y que la presente en el
    formato limpio (2 decimales, sin símbolos)"""
    texts = [_completion_text(c) for c in completions]
    rewards = []
    for text in texts:
        match = FORMAT_PATTERN.match(text)
        if match is None:
            rewards.append(0.0)
            continue

        think = match.group("think")
        answer = match.group("answer").strip()

        reward = 0.0
        distinct_numbers = len(set(NUMBER_PATTERN.findall(think)))
        reward += 0.5 * min(distinct_numbers / MIN_CALC_NUMBERS, 1.0)
        if STRICT_ANSWER_PATTERN.match(answer):
            reward += 0.5

        rewards.append(reward)
    return rewards


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.data == "gsm8k":
        dataset = load_gsm8k("train", n_examples=args.n_examples, seed=args.seed)
    else:
        dataset = load_domain_dataset(args.data)
    print(f"{len(dataset)} training problems")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GRPOConfig(
        output_dir=args.output,
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=(args.num_generations // args.micro_batch) * args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        epsilon=0.2,
        bf16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        logging_steps=1,
        save_steps=args.save_steps,
        save_strategy="steps",
        report_to="none",
        seed=args.seed,
        log_completions=True,
        num_completions_to_print=2,
        model_init_kwargs={"dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        reward_weights=[1.0, 2.0, 0.5] #lets you weight format / accuracy / domain.
    )
    if args.num_generations % args.micro_batch:
        raise SystemExit("--micro-batch must divide --num-generations")
    if args.init_adapter:
        # Continue training the SFT adapter: load base + adapter as a trainable PeftModel.
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )
        model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
        peft_config = None
    else:
        model = args.model
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, accuracy_reward, domain_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output)


    history_path = Path(args.output) / "history.json"
    history_path.write_text(json.dumps(trainer.state.log_history, indent=2, default=str))
    print(f"final adapter saved to {args.output}; history -> {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' or path to your domain JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--init-adapter", default=None, help="SFT adapter to start from")
    parser.add_argument("--output", default="rlm/weights/final_rlm_lora")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=2,
        help="sequences per forward pass; must divide --num-generations (lower it if OOM)",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 disables it)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="path to a checkpoint-XXX folder to continue an interrupted run (24h sessions!)",
    )
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
