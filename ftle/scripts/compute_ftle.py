# ABOUTME: JVP-based FTLE (top-1 Lyapunov exponent) estimator for transformer depth dynamics.
# ABOUTME: Computes λ₁ via tangent propagation through transformer layers with QR renormalization.

import math
import time
from contextlib import contextmanager
from typing import Optional

import torch
from torch.func import jvp


@contextmanager
def _math_attention():
    """Force math SDPA backend — flash attention lacks forward-mode AD support."""
    with torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    ):
        yield


def _get_transformer_layers(model):
    """Extract the ordered list of transformer layers from a HuggingFace model.

    Supports GPT-2 (model.transformer.h) and Llama/Qwen/Mistral (model.model.layers).
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2 family
        return list(model.transformer.h)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama / Qwen / Mistral family
        return list(model.model.layers)
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model).__name__}. "
            "Expected model.transformer.h (GPT-2) or model.model.layers (Llama/Qwen/Mistral)."
        )


def _get_hidden_states(model, input_ids, device):
    """Run a forward pass and capture hidden states at the input of each transformer layer.

    Returns (hidden_states, position_embeddings) where:
    - hidden_states: list of tensors [h_0, h_1, ..., h_{L-1}, h_L]
      h_l is the input to layer l, h_0 is the embedding output.
    - position_embeddings: tuple (cos, sin) for RoPE models, None for GPT-2.
    """
    hidden_states = []
    position_embeddings = None

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2: manually replicate the forward pass to capture intermediate states
        wte = model.transformer.wte
        wpe = model.transformer.wpe
        ln_f = model.transformer.ln_f
        layers = model.transformer.h

        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        h = wte(input_ids) + wpe(position_ids)

        with _math_attention():
            for layer in layers:
                hidden_states.append(h.detach().clone())
                h = layer(h)[0]  # GPT2Block returns (hidden_states, ...) tuple

        # Append final hidden state (output of last layer, before ln_f)
        hidden_states.append(h.detach().clone())

    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Llama/Qwen/Mistral: use output_hidden_states
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        # output_hidden_states gives [embedding_output, layer_0_output, ..., layer_L-1_output]
        for hs in out.hidden_states:
            hidden_states.append(hs.detach().clone())

        # Pre-compute RoPE position embeddings for layer-by-layer JVP
        if hasattr(model.model, "rotary_emb"):
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            position_embeddings = model.model.rotary_emb(hidden_states[0], position_ids)
    else:
        raise ValueError(f"Unsupported model architecture: {type(model).__name__}")

    return hidden_states, position_embeddings


def _layer_residual_fn(layer, h, position_embeddings=None):
    """Compute the residual update F_l(h) = layer(h) - h.

    The full layer is h_{l+1} = h_l + F_l(h_l) where F_l captures the
    attention + MLP transformation minus the identity (residual connection).
    But in practice, GPT-2 layers already include the residual connection
    internally, so layer(h) = h + F_l(h). We compute F_l(h) = layer(h) - h.

    For RoPE models (Llama/Qwen/Mistral), position_embeddings must be passed
    so each layer can apply rotary position encoding.
    """
    if position_embeddings is not None:
        layer_out = layer(h, position_embeddings=position_embeddings)
    else:
        layer_out = layer(h)
    # HuggingFace layers return tuples; first element is the hidden state
    if isinstance(layer_out, tuple):
        layer_out = layer_out[0]
    return layer_out - h


def compute_ftle(
    model,
    input_ids: torch.Tensor,
    tangent_seed: int = 42,
    renorm_cadence: int = 2,
    device: str = "cpu",
) -> dict:
    """Compute top-1 FTLE (λ₁) for a transformer model on given input.

    Uses JVP-based tangent propagation through transformer layers with
    periodic QR renormalization to prevent overflow.

    Args:
        model: HuggingFace transformer model (GPT-2 or Llama/Qwen/Mistral).
        input_ids: Token IDs tensor of shape (1, seq_len).
        tangent_seed: Random seed for tangent vector initialization.
        renorm_cadence: Renormalize tangent vector every k layers.
        device: Computation device.

    Returns:
        Dictionary with lambda1, layerwise_profile, metadata, etc.
    """
    t0 = time.time()
    model.eval()
    input_ids = input_ids.to(device)

    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    # Step 1: Get hidden states at each layer input
    hidden_states, position_embeddings = _get_hidden_states(model, input_ids, device)
    # hidden_states[l] = input to layer l (shape: [1, seq_len, d_model])

    d_model = hidden_states[0].shape[-1]
    seq_len = hidden_states[0].shape[1]

    # Step 2: Initialize tangent vector (random unit vector per token position)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(tangent_seed)
    # Shape: [1, seq_len, d_model] — same shape as hidden states
    v = torch.randn(1, seq_len, d_model, generator=rng, dtype=hidden_states[0].dtype)
    v = v.to(device)
    # Normalize per-token: each token position gets a unit tangent vector
    token_norms = v.norm(dim=-1, keepdim=True)
    v = v / token_norms

    # Step 3: Propagate tangent vector through layers
    log_norms_accumulated = torch.zeros(1, seq_len, device=device)
    layerwise_log_norms = []
    nan_detected = False
    inf_detected = False
    abort_layer = None

    with _math_attention():
        for l_idx in range(n_layers):
            layer = layers[l_idx]
            h_l = hidden_states[l_idx]
            h_l_jvp = h_l.detach().clone()

            # Compute JVP: (F_l(h_l), ∂F_l/∂h_l · v)
            def residual_fn(h, _layer=layer, _pos_emb=position_embeddings):
                return _layer_residual_fn(_layer, h, position_embeddings=_pos_emb)

            try:
                _, jvp_out = jvp(residual_fn, (h_l_jvp,), (v,))
            except Exception as e:
                nan_detected = True
                abort_layer = l_idx
                break

            # v_{l+1} = J_l · v_l = v_l + (∂F_l/∂h_l) · v_l
            # Since F_l(h) = layer(h) - h, J_l = I + ∂F_l/∂h_l,
            # jvp gives ∂F_l/∂h_l · v, so full tangent update is v + jvp_out
            v = v + jvp_out

            # NaN/Inf check
            if torch.isnan(v).any():
                nan_detected = True
                abort_layer = l_idx
                break
            if torch.isinf(v).any():
                inf_detected = True
                abort_layer = l_idx
                break

            # Record layerwise log-norms (per-token, then mean over tokens)
            per_token_norm = v.norm(dim=-1)  # [1, seq_len]
            log_norm = torch.log(per_token_norm + 1e-30)  # avoid log(0)
            layerwise_log_norms.append(log_norm.mean().item())

            # QR renormalization every renorm_cadence layers
            if (l_idx + 1) % renorm_cadence == 0:
                per_token_norm = v.norm(dim=-1, keepdim=True)  # [1, seq_len, 1]
                log_norms_accumulated += torch.log(per_token_norm.squeeze(-1) + 1e-30)
                v = v / per_token_norm

    # Step 4: Final accounting — add the remaining norm if not renormalized at last layer
    if not (nan_detected or inf_detected):
        final_norm = v.norm(dim=-1)  # [1, seq_len]
        log_norms_accumulated += torch.log(final_norm + 1e-30)

        # λ₁ = (1/L) * total_log_norm — averaged over token positions
        lambda1_per_token = log_norms_accumulated / n_layers  # [1, seq_len]
        lambda1 = lambda1_per_token.mean().item()
    else:
        lambda1 = float("nan")

    compute_time = time.time() - t0

    return {
        "lambda1": lambda1,
        "layerwise_profile": layerwise_log_norms,
        "n_layers": n_layers,
        "d_model": d_model,
        "seq_len": seq_len,
        "tangent_seed": tangent_seed,
        "renorm_cadence": renorm_cadence,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "abort_layer": abort_layer,
        "compute_time_s": round(compute_time, 3),
    }


def compute_ftle_finite_difference(
    model,
    input_ids: torch.Tensor,
    tangent_seed: int = 42,
    renorm_cadence: int = 2,
    device: str = "cpu",
    epsilon: float = 1e-5,
) -> dict:
    """Compute top-1 FTLE via finite-difference approximation.

    For each layer, approximates J_l · v ≈ (F_l(h_l + εv) - F_l(h_l)) / ε.
    This serves as ground truth for validating the JVP implementation.

    Args:
        model: HuggingFace transformer model.
        input_ids: Token IDs tensor.
        tangent_seed: Random seed for tangent vector initialization.
        renorm_cadence: Renormalize every k layers.
        device: Computation device.
        epsilon: Finite-difference step size.

    Returns:
        Dictionary with lambda1, layerwise_profile, metadata.
    """
    t0 = time.time()
    model.eval()
    input_ids = input_ids.to(device)

    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    hidden_states, position_embeddings = _get_hidden_states(model, input_ids, device)
    d_model = hidden_states[0].shape[-1]
    seq_len = hidden_states[0].shape[1]

    # Initialize tangent vector (same seed as JVP method)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(tangent_seed)
    v = torch.randn(1, seq_len, d_model, generator=rng, dtype=hidden_states[0].dtype)
    v = v.to(device)
    token_norms = v.norm(dim=-1, keepdim=True)
    v = v / token_norms

    log_norms_accumulated = torch.zeros(1, seq_len, device=device)
    layerwise_log_norms = []
    nan_detected = False
    inf_detected = False
    abort_layer = None

    with _math_attention():
        for l_idx in range(n_layers):
            layer = layers[l_idx]
            h_l = hidden_states[l_idx]

            # Finite difference: (F_l(h_l + ε*v) - F_l(h_l)) / ε
            with torch.no_grad():
                f_base = _layer_residual_fn(layer, h_l, position_embeddings=position_embeddings)
                f_perturbed = _layer_residual_fn(layer, h_l + epsilon * v, position_embeddings=position_embeddings)
                jvp_approx = (f_perturbed - f_base) / epsilon

            # Full tangent update: v_{l+1} = v + J_F · v (since J_layer = I + J_F)
            v = v + jvp_approx

            if torch.isnan(v).any():
                nan_detected = True
                abort_layer = l_idx
                break
            if torch.isinf(v).any():
                inf_detected = True
                abort_layer = l_idx
                break

            per_token_norm = v.norm(dim=-1)
            log_norm = torch.log(per_token_norm + 1e-30)
            layerwise_log_norms.append(log_norm.mean().item())

            if (l_idx + 1) % renorm_cadence == 0:
                per_token_norm = v.norm(dim=-1, keepdim=True)
                log_norms_accumulated += torch.log(per_token_norm.squeeze(-1) + 1e-30)
                v = v / per_token_norm

    if not (nan_detected or inf_detected):
        final_norm = v.norm(dim=-1)
        log_norms_accumulated += torch.log(final_norm + 1e-30)
        lambda1_per_token = log_norms_accumulated / n_layers
        lambda1 = lambda1_per_token.mean().item()
    else:
        lambda1 = float("nan")

    compute_time = time.time() - t0

    return {
        "lambda1": lambda1,
        "layerwise_profile": layerwise_log_norms,
        "n_layers": n_layers,
        "d_model": d_model,
        "seq_len": seq_len,
        "tangent_seed": tangent_seed,
        "renorm_cadence": renorm_cadence,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "abort_layer": abort_layer,
        "epsilon": epsilon,
        "compute_time_s": round(compute_time, 3),
    }
