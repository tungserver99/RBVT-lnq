from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from calibration_utils import load_calibration_data
from eval_perplexity import RBVTSlidingWindowEvaluator
from guidedquant_adapter import (
    build_hessian_cache_path,
    build_lnq_cache_path,
    load_tokenizer,
    materialize_lnq_variant,
    run_lnq_pipeline,
)
from lm_eval_runner import LMEvalHarnessRunner
from runtime_utils import load_runtime_env, resolve_hf_token

DEFAULT_LM_EVAL_TASKS = [
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "piqa",
    "winogrande",
    "boolq",
    "rte",
    "openbookqa",
    "lambada_openai",
    "mmlu",
    "gsm8k",
]


def save_run_summary(output_dir: Path, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Saved run summary to {summary_path}")


def build_run_name(args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = args.model_path.rstrip("/").split("/")[-1]
    return (
        f"rbvt_lnq_{model_slug}_{args.bits}bit_"
        f"{args.dataset}_g{args.num_groups}_iter{args.num_iterations}_cd{args.cd_cycles}_{timestamp}"
    )


def evaluate_models(model_paths: dict[str, str], args) -> tuple[dict, dict | None]:
    evaluator = RBVTSlidingWindowEvaluator(
        device=args.device,
        seed=args.seed,
        stride=args.eval_stride,
        max_length=args.eval_max_length,
        cache_dir=args.eval_cache_dir,
        hf_token=args.hf_token,
    )

    ppl_results: dict[str, dict] = {}
    datasets = {
        "wikitext2": evaluator.load_wikitext2_test(),
        "c4": evaluator.load_c4_validation(args.eval_c4_samples),
    }
    for variant_name, model_path in model_paths.items():
        ppl_results[variant_name] = {}
        for dataset_name, texts in datasets.items():
            result = evaluator.evaluate_model_on_dataset(
                model_path=model_path,
                model_name=variant_name,
                texts=texts,
                dataset_name=dataset_name,
            )
            ppl_results[variant_name][dataset_name] = result

    lm_eval_results = None
    if args.include_lm_eval and args.lm_eval_tasks:
        runner = LMEvalHarnessRunner(
            tasks=list(args.lm_eval_tasks),
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_dir=args.lm_eval_output_dir,
            run_name=args.run_name,
            hf_token=args.hf_token,
        )
        lm_eval_results = runner.run(model_paths)

    return ppl_results, lm_eval_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GuidedQuant LNQ, then evaluate LNQ and LNQ + RBVT-last on the same learned codebook."
    )
    parser.add_argument("--model-path", required=True, help="HF model name or local path")
    parser.add_argument("--bits", type=int, required=True, help="LNQ bit-width")
    parser.add_argument("--output-root", default="./outputs")
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--yaml-path", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dataset",
        default="c4",
        choices=["c4", "wikitext2", "ptb", "pileval", "redpajama"],
        help="GuidedQuant calibration dataset for initialization + Hessians + LNQ.",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--cd-cycles", type=int, default=4)
    parser.add_argument("--cpu-count", type=int, default=None)
    parser.add_argument("--overwrite-tokens", action="store_true")
    parser.add_argument("--overwrite-gradients", action="store_true")
    parser.add_argument("--overwrite-quantize", action="store_true")
    parser.add_argument("--overwrite-pack", action="store_true")
    parser.add_argument("--sub-qlayer", nargs=2, type=int, default=None)
    parser.add_argument("--is-nosal", action="store_true")

    parser.add_argument(
        "--rbvt-calib-dataset",
        default="c4",
        help="Calibration dataset for RBVT activation stats. Default: reuse --dataset if supported, else wikitext2.",
    )
    parser.add_argument("--rbvt-n-calib", type=int, default=128)
    parser.add_argument("--rbvt-max-length", type=int, default=2048)
    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument(
        "--rbvt-topk",
        type=int,
        default=0,
        help="Optional per-row candidate prefilter for RBVT; 0 keeps the full candidate set.",
    )
    parser.add_argument(
        "--rbvt-budget-p",
        type=float,
        default=0.005,
        help="Fraction of sorted RBVT candidates retained before relaxation; 0 disables flips.",
    )
    parser.add_argument(
        "--rbvt-target-ratio",
        type=float,
        default=0.1,
        help="Scaled RBVT target magnitude; 0.1 matches the 1/10 target reduction suggestion.",
    )
    parser.add_argument(
        "--rbvt-mse-guard",
        dest="rbvt_mse_guard",
        action="store_true",
        default=False,
        help="Keep only neighbour moves satisfying the MSE-improving guard gap < 2|e|.",
    )
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument(
        "--rbvt-position",
        default="assignment_last",
        choices=["codebook_last", "assignment_last"],
        help="Where to insert RBVT relative to LNQ: "
        "codebook_last = apply RBVT directly on the final learned codebook/cache; "
        "assignment_last = first run the final LNQ assignment on the final codebook, then apply RBVT.",
    )
    parser.add_argument(
        "--rbvt-mode",
        default="lnq_aware",
        choices=["naive", "lnq_aware"],
        help="RBVT-last variant: naive uses raw LNQ weights as the target; "
        "lnq_aware uses an LNQ effective target with error propagation from the cached assignment.",
    )
    parser.add_argument("--allow-overshoot", action="store_true")
    parser.add_argument("--skip-rbvt", action="store_true")

    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-cache-dir", default="./dataset_cache")
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-c4-samples", type=int, default=500)
    parser.add_argument("--include-lm-eval", action="store_true")
    parser.add_argument("--lm-eval-tasks", nargs="+", default=list(DEFAULT_LM_EVAL_TASKS))
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=None)
    parser.add_argument("--lm-eval-limit", type=float, default=None)
    parser.add_argument("--lm-eval-output-dir", default="./outputs/lm_eval")

    return parser.parse_args()


def main():
    load_runtime_env()
    args = parse_args()
    if args.hf_token is None:
        args.hf_token = resolve_hf_token()
    if args.rbvt_lambda < 0.0:
        raise ValueError("--rbvt-lambda must be non-negative")
    if not 0.0 <= args.rbvt_budget_p <= 1.0:
        raise ValueError("--rbvt-budget-p must be in [0, 1]")
    if not 0.0 <= args.rbvt_target_ratio <= 1.0:
        raise ValueError("--rbvt-target-ratio must be in [0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.run_name = build_run_name(args)
    run_root = Path(args.output_root) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RBVT-LNQ")
    print("=" * 80)
    print(f"Run name: {args.run_name}")
    print(f"Model: {args.model_path}")
    print(f"Bits: {args.bits}")
    print(f"GuidedQuant dataset: {args.dataset}")
    print(f"RBVT enabled: {not args.skip_rbvt}")
    if not args.skip_rbvt:
        print(f"RBVT position: {args.rbvt_position}")
        print(f"RBVT mode: {args.rbvt_mode}")
    print("=" * 80)

    run_lnq_pipeline(
        model_path=args.model_path,
        bits=args.bits,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        num_groups=args.num_groups,
        num_iterations=args.num_iterations,
        cd_cycles=args.cd_cycles,
        yaml_path=args.yaml_path,
        cpu_count=args.cpu_count,
        overwrite_tokens=args.overwrite_tokens,
        overwrite_gradients=args.overwrite_gradients,
        overwrite_quantize=args.overwrite_quantize,
        overwrite_pack=args.overwrite_pack,
        random_state=args.seed,
        sub_qlayer=tuple(args.sub_qlayer) if args.sub_qlayer else None,
        is_nosal=args.is_nosal,
    )

    lnq_cache_path = build_lnq_cache_path(
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        bits=args.bits,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        num_groups=args.num_groups,
        num_iterations=args.num_iterations,
        cd_cycles=args.cd_cycles,
        is_nosal=args.is_nosal,
    )
    print(f"LNQ cache: {lnq_cache_path}")
    hessian_cache_path = build_hessian_cache_path(
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        dataset=args.dataset,
        seq_len=args.seq_len,
        num_examples=args.num_examples,
        num_groups=args.num_groups,
        is_nosal=args.is_nosal,
    )
    print(f"Hessian cache: {hessian_cache_path}")

    model_paths: dict[str, str] = {}

    lnq_model_dir = run_root / "lnq_model"
    lnq_summary = materialize_lnq_variant(
        model_path=args.model_path,
        lnq_cache_path=str(lnq_cache_path),
        bits=args.bits,
        output_dir=str(lnq_model_dir),
        device=args.device,
        hf_token=args.hf_token,
    )
    model_paths["lnq"] = str(lnq_model_dir)

    rbvt_last_summary = None
    rbvt_variant_name = None
    if not args.skip_rbvt:
        rbvt_calib_dataset = args.rbvt_calib_dataset
        if rbvt_calib_dataset is None:
            rbvt_calib_dataset = args.dataset if args.dataset in {"c4", "wikitext2"} else "wikitext2"

        tokenizer_texts = load_calibration_data(
            dataset_name=rbvt_calib_dataset,
            tokenizer=load_tokenizer(args.model_path, args.hf_token),
            n_samples=args.rbvt_n_calib,
            seqlen=args.rbvt_max_length,
            seed=args.seed,
            cache_dir=str(run_root / "rbvt_calibration_cache"),
        )

        rbvt_variant_name = (
            f"lnq_rbvt_{args.rbvt_position}_"
            f"{'naive' if args.rbvt_mode == 'naive' else 'lnq_aware'}"
        )
        rbvt_model_dir = run_root / f"{rbvt_variant_name}_model"
        rbvt_last_summary = materialize_lnq_variant(
            model_path=args.model_path,
            lnq_cache_path=str(lnq_cache_path),
            bits=args.bits,
            output_dir=str(rbvt_model_dir),
            device=args.device,
            hf_token=args.hf_token,
            rbvt_position=args.rbvt_position,
            rbvt_target_mode="raw_weight" if args.rbvt_mode == "naive" else "lnq_aware",
            hessian_cache_path=str(hessian_cache_path),
            cd_cycles=args.cd_cycles,
            rbvt_calib_texts=tokenizer_texts,
            rbvt_lambda=args.rbvt_lambda,
            rbvt_topk=args.rbvt_topk,
            rbvt_budget_p=args.rbvt_budget_p,
            rbvt_target_ratio=args.rbvt_target_ratio,
            rbvt_mse_guard=args.rbvt_mse_guard,
            gap_floor=args.gap_floor,
            row_chunk=args.row_chunk,
            rbvt_max_length=args.rbvt_max_length,
            strict_descent=not args.allow_overshoot,
        )
        model_paths[rbvt_variant_name] = str(rbvt_model_dir)

    ppl_results = None
    lm_eval_results = None
    if not args.skip_eval:
        ppl_results, lm_eval_results = evaluate_models(model_paths, args)

    summary = {
        "run_name": args.run_name,
        "model_path": args.model_path,
        "bits": args.bits,
        "dataset": args.dataset,
        "num_groups": args.num_groups,
        "num_iterations": args.num_iterations,
        "cd_cycles": args.cd_cycles,
        "lnq_cache_path": str(lnq_cache_path),
        "rbvt_position": None if args.skip_rbvt else args.rbvt_position,
        "rbvt_mode": None if args.skip_rbvt else args.rbvt_mode,
        "rbvt_lambda": None if args.skip_rbvt else args.rbvt_lambda,
        "rbvt_topk": None if args.skip_rbvt else args.rbvt_topk,
        "rbvt_budget_p": None if args.skip_rbvt else args.rbvt_budget_p,
        "rbvt_target_ratio": None if args.skip_rbvt else args.rbvt_target_ratio,
        "rbvt_mse_guard": None if args.skip_rbvt else args.rbvt_mse_guard,
        "variants": {
            "lnq": lnq_summary,
            **({rbvt_variant_name: rbvt_last_summary} if rbvt_variant_name is not None else {}),
        },
        "model_paths": model_paths,
        "perplexity": ppl_results,
        "lm_eval": lm_eval_results,
    }
    save_run_summary(run_root, summary)


if __name__ == "__main__":
    main()
