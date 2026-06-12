import torch

MIN_RATIO = .85

def matched_ratio(
    output: torch.Tensor,
    reference: torch.Tensor,
    atol: float = 0.1,
    rtol: float = 0.2,
) -> float:
    x = output.to(torch.float32)
    y = reference.to(torch.float32)
    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)
    exceeds = (abs_error > atol) & (rel_error > rtol)
    total = abs_error.numel()
    if total == 0:
        return 1.0
    return 1.0 - (float(exceeds.sum().item()) / float(total))


def global_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine of flattened ``a`` and ``b``. F.cosine_similarity is 0 for (0,0) pairs; treat as 1 if both near-zero and close."""
    u = a.flatten().float()
    v = b.flatten().float()
    nu = float(u.norm().item())
    nv = float(v.norm().item())
    if nu < 1e-12 and nv < 1e-12:
        return 1.0 if torch.allclose(u, v, atol=1e-6, rtol=1e-6) else 0.0
    return float((u @ v / (nu * nv)).clamp(-1.0, 1.0))


def check_correctness(
    output: torch.Tensor,
    reference: torch.Tensor,
):
    print('\n\n ### Correctness checking ### \n')
    ratio = matched_ratio(output, reference, atol=0.1, rtol=0.2)
    cos = global_cosine_similarity(output, reference)
    if ratio < MIN_RATIO:
        print(f'matched ratio ({ratio:.3f})< min ratio ({MIN_RATIO})')
    if cos < .99:
        print(f'cos similarity ({cos:.3f}) less than min (99%)')
    print(f"  correctness: matched_ratio={ratio*100:.2f}%  cosine_sim={cos:.6f}")
