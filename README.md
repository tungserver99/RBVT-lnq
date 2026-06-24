# RBVT-lnq

`RBVT-lnq` is a diagnostic project for the setting:

`RBVT-lnq` separates two orthogonal design axes on top of the LNQ baseline:

- RBVT position:
  `codebook_last` or `assignment_last`
- RBVT target:
  `naive` or `lnq_aware`

This project does not claim to replace the assignment step inside every LNQ iteration. Instead, it evaluates whether RBVT can improve the final fixed-codebook assignment after GuidedQuant LNQ has already finished learning the codebook.

The GuidedQuant baseline is kept as a full third-party source tree under `thirdparty/GuidedQuant`. The project-specific code in this repository only imports and consumes its artifacts; it does not rewrite the baseline LNQ implementation.

## What is implemented

- GuidedQuant initialization and LNQ cache generation via an unmodified third-party checkout of `snu-mllab/GuidedQuant`
- RBVT last-step post-pass via the standard RBVT solver adapted from `RBVTQuant`
- Dense model materialization for the LNQ baseline plus RBVT variants such as:
  - `lnq`
  - `lnq_rbvt_codebook_last_naive`
  - `lnq_rbvt_codebook_last_lnq_aware`
  - `lnq_rbvt_assignment_last_naive`
  - `lnq_rbvt_assignment_last_lnq_aware`
- Perplexity and `lm-eval` evaluation using the same evaluation backbone as `RBVTQuant`

## Protocol

1. Run GuidedQuant initialization with `seed_precision = parent_precision = bits`.
2. Run GuidedQuant `layerwise_nuq` to produce the LNQ cache.
3. Materialize the cached LNQ result as the baseline.
4. Choose the RBVT position:
   `codebook_last` applies RBVT directly on the final learned codebook/cache;
   `assignment_last` first runs the final LNQ assignment on the final codebook, then applies RBVT.
5. Choose the RBVT target:
   `naive` uses the raw LNQ weight as the target;
   `lnq_aware` uses an LNQ effective target derived from the cached assignment and Hessian error propagation.
6. Save and evaluate the selected RBVT variant together with the LNQ baseline.

## Post-Pass Interpretation

For the third-party GuidedQuant implementation, `train_least_squares(...)` in `layerwise_quantize.py` updates:

- `P` on iterations after the first
- then `C`

The saved LNQ cache therefore already contains LNQ's latest available assignment together with the codebook it learned.

So this project explicitly separates:

- `codebook_last`: use the cached LNQ state `(P^T, C^T)`
- `assignment_last`: recompute a final LNQ assignment on `C^T` before RBVT
- `naive`: `RBVT(W_raw, P, C)`
- `lnq_aware`: `RBVT(W_eff, P, C)`, where `W_eff` is built from LNQ's propagated assignment target

## Important positioning

Use this as:

- `LNQ baseline`
- `LNQ + RBVT-codebook-last`
- `LNQ + RBVT-assignment-last`

Do not describe it as:

- replacing the LNQ assignment solver in every alternating-minimization round
- preserving LNQ convergence guarantees after swapping CD by RBVT

## Layout

- [main.py](./main.py): end-to-end experiment entrypoint
- [guidedquant_adapter.py](./guidedquant_adapter.py): bridge between GuidedQuant caches and RBVT
- [quantizers/rbvt.py](./quantizers/rbvt.py): RBVT post-pass assignment solver
- [thirdparty/GuidedQuant](./thirdparty/GuidedQuant): full upstream GuidedQuant source tree kept for baseline transparency

## Example

```bash
cd RBVT-lnq
python main.py \
  --model-path meta-llama/Llama-2-7b-hf \
  --bits 2 \
  --num-groups 4 \
  --dataset c4 \
  --seq-len 2048 \
  --num-examples 128 \
  --rbvt-calib-dataset c4 \
  --rbvt-n-calib 128 \
  --rbvt-max-length 2048 \
  --include-lm-eval
```

Or use:

```bash
bash scripts/run_lnq_plain.sh
bash scripts/run_lnq_rbvt_codebook_last.sh
bash scripts/run_lnq_rbvt_last.sh
bash scripts/run_lnq_suite.sh --mode all --rbvt-mode all
```

Use `RBVT_MODE=naive` or `RBVT_MODE=lnq_aware` with the RBVT scripts to choose the target type.
For one unified entrypoint, use `run_lnq_suite.sh` and choose `--mode`, `--rbvt-mode`, `--model`, and `--bits`.
`RBVT-lnq` also reads `.env` from the project root; if `--hf-token` is not provided, it will use `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, or `HUGGINGFACE_TOKEN`.

## Notes

- GuidedQuant LNQ still needs its initialization cache, so the project first runs the GuidedQuant seed quantizer before `layerwise_nuq`.
- The saved output models are dense Hugging Face checkpoints for clean evaluation with the `RBVTQuant` eval flow.
- The default CLI settings are `--rbvt-position assignment_last` and `--rbvt-mode lnq_aware`.
- Reviewer-facing note: the baseline code lives in `thirdparty/GuidedQuant`; the experiment glue in this repo only orchestrates runs and post-processes artifacts.
- Experiment defaults are aligned with `RBVTQuant`: C4 calibration with `128/2048`, perplexity on `WikiText-2` and `C4`, and `lm-eval` on `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, `boolq`, `rte`, `openbookqa`, `lambada_openai`, `mmlu`, and `gsm8k`.
