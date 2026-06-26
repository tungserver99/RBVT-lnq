from __future__ import annotations

import gc
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
GUIDEDQUANT_ROOT = ROOT / "thirdparty" / "GuidedQuant"
if str(GUIDEDQUANT_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDEDQUANT_ROOT))

from any_precision.analyzer import get_analyzer
from any_precision.quantization import any_precision_quantize, layerwise_nuq
from any_precision.quantization import datautils as guidedquant_datautils
from any_precision.quantization import layerwise_main as guidedquant_layerwise_main
from any_precision.quantization import main as guidedquant_main
from any_precision.quantization.layerwise_quantize import fix_hessian_shape, update_P

from calibration_utils import get_c4_calibration_data
from quantizers.base_quantizer import QuantResult
from quantizers.rbvt import apply_rbvt


def _hf_device_map(device: str):
    return {"": device} if device != "auto" else "auto"


class ActStatsCollector:
    def __init__(self, want_var: bool = True):
        self.sum: Dict[str, torch.Tensor] = {}
        self.sumsq: Dict[str, torch.Tensor] = {}
        self.count: Dict[str, int] = {}
        self.want_var = want_var
        self.hooks = []

    def _hook(self, name: str):
        def hook(_module, inp, _out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.reshape(-1, x.shape[-1]).detach().float()
            s = x.sum(dim=0).cpu()
            n = x.shape[0]
            if name not in self.sum:
                self.sum[name] = s
                self.count[name] = n
                if self.want_var:
                    self.sumsq[name] = (x * x).sum(dim=0).cpu()
            else:
                self.sum[name] += s
                self.count[name] += n
                if self.want_var:
                    self.sumsq[name] += (x * x).sum(dim=0).cpu()

        return hook

    def register(self, modules: Iterable[Tuple[str, nn.Module]]):
        for name, module in modules:
            self.hooks.append(module.register_forward_hook(self._hook(name)))

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def mean(self, name: str) -> torch.Tensor:
        return self.sum[name] / max(1, self.count[name])

    def var(self, name: str) -> torch.Tensor | None:
        if not self.want_var or name not in self.sumsq:
            return None
        mean = self.mean(name)
        ex2 = self.sumsq[name] / max(1, self.count[name])
        return (ex2 - mean * mean).clamp(min=0.0)


def _model_name(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def build_init_cache_path(
    cache_dir: str,
    model_path: str,
    bits: int,
    dataset: str,
    seq_len: int,
    num_examples: int,
) -> Path:
    return (
        Path(cache_dir)
        / "quantized"
        / f"{_model_name(model_path)}-w{bits}_orig{bits}-{dataset}_s{num_examples}_blk{seq_len}"
    )


def build_lnq_cache_path(
    cache_dir: str,
    model_path: str,
    bits: int,
    dataset: str,
    seq_len: int,
    num_examples: int,
    num_groups: int,
    num_iterations: int,
    cd_cycles: int,
    is_nosal: bool,
) -> Path:
    suffix = "_nosal" if is_nosal else ""
    return (
        Path(cache_dir)
        / "layerwise_quantized"
        / (
            f"{_model_name(model_path)}-w{bits}-{dataset}_s{num_examples}_blk{seq_len}"
            f"_g{num_groups}_iter{num_iterations}_cd{cd_cycles}{suffix}"
        )
    )


def build_hessian_cache_path(
    cache_dir: str,
    model_path: str,
    dataset: str,
    seq_len: int,
    num_examples: int,
    num_groups: int,
    is_nosal: bool,
) -> Path:
    suffix = "_nosal" if is_nosal else ""
    return (
        Path(cache_dir)
        / "hessians"
        / f"{_model_name(model_path)}-{dataset}_s{num_examples}_blk{seq_len}_g{num_groups}{suffix}"
    )


@contextmanager
def _patched_guidedquant_c4_tokens():
    original_datautils_get_tokens = guidedquant_datautils.get_tokens
    original_main_get_tokens = guidedquant_main.get_tokens
    original_layerwise_get_tokens = guidedquant_layerwise_main.get_tokens

    def squeeze_style_get_tokens(
        dataset_name,
        split,
        tokenizer,
        seq_len,
        num_samples,
        save_path=None,
        seed=None,
    ):
        if dataset_name != "c4":
            return original_datautils_get_tokens(
                dataset_name,
                split,
                tokenizer,
                seq_len,
                num_samples,
                save_path,
                seed,
            )

        if save_path is not None and os.path.isfile(save_path):
            return torch.load(save_path)

        cache_dir = Path(save_path).parent if save_path is not None else ROOT / "calibration_cache"
        token_batches = get_c4_calibration_data(
            tokenizer=tokenizer,
            n_samples=num_samples,
            seqlen=seq_len,
            seed=42 if seed is None else seed,
            return_tensors=True,
            cache_dir=cache_dir,
        )
        tokens = [batch.squeeze(0).cpu() for batch in token_batches]

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(tokens, save_path)
        return tokens

    guidedquant_datautils.get_tokens = squeeze_style_get_tokens
    guidedquant_main.get_tokens = squeeze_style_get_tokens
    guidedquant_layerwise_main.get_tokens = squeeze_style_get_tokens
    try:
        yield
    finally:
        guidedquant_datautils.get_tokens = original_datautils_get_tokens
        guidedquant_main.get_tokens = original_main_get_tokens
        guidedquant_layerwise_main.get_tokens = original_layerwise_get_tokens


def run_lnq_pipeline(
    model_path: str,
    bits: int,
    cache_dir: str,
    dataset: str,
    seq_len: int,
    num_examples: int,
    num_groups: int,
    num_iterations: int,
    cd_cycles: int,
    yaml_path: str | None,
    cpu_count: int | None,
    overwrite_tokens: bool,
    overwrite_gradients: bool,
    overwrite_quantize: bool,
    overwrite_pack: bool,
    random_state: int,
    sub_qlayer: tuple[int, int] | None,
    is_nosal: bool,
):
    with _patched_guidedquant_c4_tokens():
        print("Running GuidedQuant initialization cache build ...")
        any_precision_quantize(
            model=model_path,
            seed_precision=bits,
            parent_precision=bits,
            mode="quantize",
            yaml_path=yaml_path,
            cache_dir=cache_dir,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            cpu_count=cpu_count,
            overwrite_tokens=overwrite_tokens,
            overwrite_gradients=overwrite_gradients,
            overwrite_quantize=overwrite_quantize,
            overwrite_pack=overwrite_pack,
            random_state=random_state,
            num_groups=num_groups,
        )

        print("Running GuidedQuant LNQ cache build ...")
        layerwise_nuq(
            model=model_path,
            seed_precision=bits,
            mode="quantize",
            yaml_path=yaml_path,
            cache_dir=cache_dir,
            dataset=dataset,
            seq_len=seq_len,
            num_examples=num_examples,
            cpu_count=cpu_count,
            overwrite_tokens=overwrite_tokens,
            overwrite_quantize=overwrite_quantize,
            overwrite_pack=overwrite_pack,
            random_state=random_state,
            num_groups=num_groups,
            num_iterations=num_iterations,
            cd_cycles=cd_cycles,
            sub_qlayer=sub_qlayer,
            is_nosal=is_nosal,
        )


def load_tokenizer(model_path: str, hf_token: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _full_module_name(analyzer, layer_idx: int, module_name: str) -> str:
    return f"{analyzer.model_name}.{analyzer.layers_name}.{layer_idx}.{module_name}"


def _target_modules(analyzer) -> List[Tuple[int, str, str, nn.Module]]:
    result = []
    layers = analyzer.get_layers()
    for layer_idx in range(analyzer.num_layers):
        modules = analyzer.get_modules(layers[layer_idx])
        for module_name, module in modules.items():
            result.append((layer_idx, module_name, _full_module_name(analyzer, layer_idx, module_name), module))
    return result


def collect_activation_stats(
    model,
    tokenizer,
    analyzer,
    calib_texts: List[str],
    device: str,
    max_length: int,
    want_var: bool,
):
    collector = ActStatsCollector(want_var=want_var)
    modules = [(full_name, module) for _, _, full_name, module in _target_modules(analyzer)]
    collector.register(modules)

    for text in calib_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model(**inputs, use_cache=False)

    collector.remove()
    means = {name: collector.mean(name) for name in collector.sum}
    variances = {}
    if want_var:
        for name in collector.sum:
            value = collector.var(name)
            if value is not None:
                variances[name] = value
    del collector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return means, variances


def _load_layer_cache(lnq_cache_path: str, bits: int, layer_idx: int):
    cache_root = Path(lnq_cache_path)
    qweight_path = cache_root / "weights" / f"l{layer_idx}.pt"
    lut_path = cache_root / f"lut_{bits}" / f"l{layer_idx}.pt"
    # GuidedQuant LNQ stores the current labels and the latest updated codebook.
    # Those labels come from LNQ's latest original assignment step, so applying
    # RBVT on top of this cache is a true post-pass after the LNQ assignment that
    # already exists in the upstream pipeline.
    qweight = torch.load(qweight_path, map_location="cpu")
    lut = torch.load(lut_path, map_location="cpu")
    return qweight, lut


def _guidedquant_to_qres_from_tensors(
    indices_tensor: torch.Tensor,
    codebooks_tensor: torch.Tensor,
    device: torch.device,
) -> QuantResult:
    indices = torch.as_tensor(indices_tensor, device=device, dtype=torch.long)
    codebooks = torch.as_tensor(codebooks_tensor, device=device, dtype=torch.float32)

    if indices.ndim != 3 or codebooks.ndim != 3:
        raise ValueError(
            f"Expected GuidedQuant tensors with shapes [out, groups, group_size] and [out, groups, K], "
            f"got {tuple(indices.shape)} and {tuple(codebooks.shape)}"
        )

    group_size = indices.shape[-1]
    expanded_codebooks = codebooks.unsqueeze(2).expand(-1, -1, group_size, -1)
    w_blocks = torch.gather(expanded_codebooks, 3, indices.unsqueeze(-1)).squeeze(-1)
    w_dequant = w_blocks.reshape(indices.shape[0], -1)

    return QuantResult(
        W_dequant=w_dequant,
        indices=indices.reshape(indices.shape[0], -1),
        q_levels=torch.arange(codebooks.shape[-1], device=device, dtype=torch.float32),
        block_scales=torch.ones(indices.shape[0], indices.shape[1], device=device, dtype=torch.float32),
        block_size=group_size,
        block_codebooks=codebooks,
        block_zeros=torch.zeros(indices.shape[0], indices.shape[1], device=device, dtype=torch.float32),
    )


def _guidedquant_to_qres(module_qweight, module_lut, device: torch.device) -> QuantResult:
    return _guidedquant_to_qres_from_tensors(module_qweight, module_lut, device)


def _load_layer_hessian(hessian_cache_path: str, layer_idx: int, module_name: str) -> torch.Tensor:
    hessian_path = Path(hessian_cache_path) / f"l{layer_idx}.pt"
    hessian_layer = torch.load(hessian_path, map_location="cpu")
    return fix_hessian_shape(hessian_layer[module_name]).float()


def _compute_lnq_effective_target(
    weight: torch.Tensor,
    qres: QuantResult,
    hessian: torch.Tensor,
) -> torch.Tensor:
    device = weight.device
    W = weight.float()
    W_hat = qres.W_dequant.to(device).float()
    H = hessian.to(device).float().clone()

    num_groups = H.shape[0]
    if W.shape[0] % num_groups != 0:
        raise ValueError(
            f"Weight rows {W.shape[0]} must be divisible by Hessian groups {num_groups}"
        )

    group_size = W.shape[0] // num_groups
    d = W.shape[1]
    diag = torch.arange(d, device=device)

    H_grp = H.clone()
    for group_idx in range(num_groups):
        H_diag = H_grp[group_idx, diag, diag].reshape(1, 1, -1)
        H_grp[group_idx, :, :] = H_grp[group_idx, :, :] / H_diag.clamp(min=1e-12)

    W_grp = W.reshape(num_groups, group_size, d)
    W_hat_grp = W_hat.reshape(num_groups, group_size, d)
    B_grp = torch.bmm(W_hat_grp - W_grp, torch.tril(H_grp, diagonal=-1))
    return (W_grp - B_grp).reshape_as(W)


def _recompute_final_lnq_assignment(
    weight: torch.Tensor,
    qres: QuantResult,
    hessian: torch.Tensor,
    cd_cycles: int,
) -> QuantResult:
    labels = qres.indices.reshape(weight.shape[0], -1).detach().cpu().long()
    codebook = qres.block_codebooks
    if codebook is None:
        raise ValueError("LNQ assignment_last requires realized block_codebooks")
    codebook = codebook.reshape(weight.shape[0], -1).detach().cpu().float()

    reassigned = update_P(
        W=weight.float(),
        H=hessian.float(),
        labels=labels,
        C=codebook,
        cd_cycles=cd_cycles,
        verbose=False,
    )
    reassigned = reassigned.reshape(weight.shape[0], 1, weight.shape[1]).cpu()
    codebook = codebook.reshape(weight.shape[0], 1, -1).cpu()
    return _guidedquant_to_qres_from_tensors(reassigned, codebook, weight.device)


def materialize_lnq_variant(
    model_path: str,
    lnq_cache_path: str,
    bits: int,
    output_dir: str,
    device: str,
    hf_token: str | None = None,
    rbvt_position: str = "assignment_last",
    rbvt_target_mode: str = "raw_weight",
    hessian_cache_path: str | None = None,
    cd_cycles: int = 4,
    rbvt_calib_texts: List[str] | None = None,
    rbvt_lambda: float = 1.0,
    rbvt_topk: int = 0,
    rbvt_budget_p: float = 1.0,
    rbvt_target_ratio: float = 1.0,
    rbvt_mse_guard: bool = False,
    gap_floor: float = 1e-8,
    row_chunk: int = 256,
    rbvt_max_length: int = 512,
    strict_descent: bool = True,
) -> dict:
    tokenizer = load_tokenizer(model_path, hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map=_hf_device_map(device),
        trust_remote_code=True,
        token=hf_token,
    )
    model.eval()

    analyzer = get_analyzer(model)
    stats_summary = None
    means: dict[str, torch.Tensor] = {}
    variances: dict[str, torch.Tensor] = {}
    if rbvt_calib_texts is not None:
        means, variances = collect_activation_stats(
            model=model,
            tokenizer=tokenizer,
            analyzer=analyzer,
            calib_texts=rbvt_calib_texts,
            device=device,
            max_length=rbvt_max_length,
            want_var=rbvt_lambda > 0.0,
        )

    total_rbvt = None
    if rbvt_calib_texts is not None:
        total_rbvt = {
            "flips": 0,
            "channels": 0,
            "candidates": 0,
            "boundary_kept": 0,
            "bias_before": 0.0,
            "bias_after": 0.0,
            "objective_before": 0.0,
            "objective_after": 0.0,
            "variance_increase": 0.0,
        }

    for layer_idx, module_name, full_name, module in _target_modules(analyzer):
        qweight_layer, lut_layer = _load_layer_cache(lnq_cache_path, bits, layer_idx)
        qres = _guidedquant_to_qres(qweight_layer[module_name], lut_layer[module_name], module.weight.device)
        if rbvt_calib_texts is not None and rbvt_position == "assignment_last":
            if not hessian_cache_path:
                raise ValueError("hessian_cache_path is required for rbvt_position='assignment_last'")
            layer_hessian = _load_layer_hessian(hessian_cache_path, layer_idx, module_name)
            qres = _recompute_final_lnq_assignment(
                weight=module.weight.data,
                qres=qres,
                hessian=layer_hessian,
                cd_cycles=cd_cycles,
            )
        weight_out = qres.W_dequant

        if rbvt_calib_texts is not None:
            rbvt_target = module.weight.data.float()
            if rbvt_target_mode == "lnq_aware":
                if not hessian_cache_path:
                    raise ValueError("hessian_cache_path is required for rbvt_target_mode='lnq_aware'")
                layer_hessian = _load_layer_hessian(hessian_cache_path, layer_idx, module_name)
                rbvt_target = _compute_lnq_effective_target(
                    weight=module.weight.data,
                    qres=qres,
                    hessian=layer_hessian,
                )
            mu = means[full_name].to(module.weight.device)
            sigma_ii = variances.get(full_name)
            if sigma_ii is not None:
                sigma_ii = sigma_ii.to(module.weight.device)
            weight_out, layer_stats = apply_rbvt(
                W_fp=rbvt_target,
                qres=qres,
                mu=mu,
                sigma_ii=sigma_ii,
                rbvt_lambda=rbvt_lambda,
                rbvt_topk=rbvt_topk if rbvt_topk > 0 else None,
                rbvt_budget_p=rbvt_budget_p,
                target_ratio=rbvt_target_ratio,
                mse_guard=rbvt_mse_guard,
                row_chunk=row_chunk,
                gap_floor=gap_floor,
                strict_descent=strict_descent,
            )
            for key, value in asdict(layer_stats).items():
                total_rbvt[key] += value

        module.weight.data = weight_out.to(module.weight.dtype)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    if total_rbvt is not None:
        stats_summary = total_rbvt

    return {
        "output_dir": str(output_path),
        "rbvt_position": rbvt_position,
        "rbvt_target_mode": rbvt_target_mode,
        "rbvt_applied": rbvt_calib_texts is not None,
        "rbvt_lambda": rbvt_lambda,
        "rbvt_topk": rbvt_topk,
        "rbvt_budget_p": rbvt_budget_p,
        "rbvt_target_ratio": rbvt_target_ratio,
        "rbvt_mse_guard": rbvt_mse_guard,
        "rbvt_stats": stats_summary,
    }
