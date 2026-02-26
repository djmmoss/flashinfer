"""Reference implementation for SM100 GDN prefill tests."""

import torch


def recurrent_ref(q, k, v, g, beta, h0, cu_seqlens, scale=None):
    """Recurrent GDN reference using SM100 gate semantics.

    Args:
        q, k, v: [1, T, H, D] tensors
        g: [1, T, H] log-space decay
        beta: [1, T, H] sigmoid gate
        h0: Optional[N, H, D, D] K-major initial state (float32)
        cu_seqlens: list of ints, cumulative sequence lengths
        scale: Optional scaling factor for q (default: 1/sqrt(D))

    Returns:
        ref_o: [1, T, H, D] float32 output
        ref_state: [N, H, D, D] K-major final state (float32)
    """
    q, k, v, beta, g = [x.squeeze(0).to(torch.float32) for x in [q, k, v, beta, g]]
    T, H, D = q.shape
    if scale is None:
        scale = 1.0 / (D**0.5)
    q = q * scale
    N = len(cu_seqlens) - 1
    V = v.shape[-1]

    o = torch.zeros(T, H, V, device=q.device)
    states = []

    for seq_idx in range(N):
        s, e = cu_seqlens[seq_idx], cu_seqlens[seq_idx + 1]
        h = (
            h0[seq_idx].clone().float().transpose(-1, -2)
            if h0 is not None
            else torch.zeros(H, D, V, device=q.device)
        )

        for i in range(s, e):
            qi, ki, vi = q[i], k[i], v[i]
            gi, bi = g[i], beta[i]

            # Decay
            h = h * gi.exp().unsqueeze(-1).unsqueeze(-1)
            # Delta
            pred = torch.einsum("hk,hkv->hv", ki, h)
            v_delta = bi.unsqueeze(-1) * (vi - pred)
            # Update
            h = h + torch.einsum("hk,hv->hkv", ki, v_delta)
            # Output
            o[i] = torch.einsum("hk,hkv->hv", qi, h)

        states.append(h.clone().transpose(-1, -2))

    return o.unsqueeze(0), torch.stack(states)
