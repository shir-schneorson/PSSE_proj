
# PSSE Diffusion Prior (Unconditional) — GNN Skeleton

A minimal, framework-agnostic skeleton for training an **unconditional diffusion model** with a **GNN ε-predictor** on power-system states. 
Use it as a starting point and plug in your PF measurement model and GN Jacobians for posterior guidance later.

> **Status**: Skeleton. Replace `TODO:` blocks with your data + PF code. Tested structure only (not executable as-is without your data & PyG).

## Structure

```
psse_diffusion_skeleton/
├─ README.md
├─ requirements.txt         # (suggested deps; edit to your stack)
├─ configs/
│  └─ default.json          # training config
├─ data/
│  ├─ dataset.py            # GraphDataset & collation
│  └─ scalers.py            # per-channel normalization helpers
├─ models/
│  ├─ eps_gnn.py            # ε-predictor GNN (PyG-style)
│  └─ layers.py             # message passing block
├─ utils/
│  ├─ schedules.py          # beta/alpha schedules
│  └─ time_embed.py         # Fourier time embedding
├─ training/
│  ├─ train_uncond.py       # main training loop (DSM)
│  └─ ema.py                # EMA helper
└─ sampling/
   └─ sample_prior.py       # DDIM/SDE-style prior sampling
```

## Quick start

1. Fill `data/dataset.py` to return your graph samples (`x0`, `edge_index`, `edge_feats`, etc.).  
2. Confirm/replace GNN layers in `models/layers.py` (PyG ops used as placeholders).  
3. Adjust config in `configs/default.json`.  
4. Run:
   ```bash
   python -m training.train_uncond --config configs/default.json
   ```

## Posterior Guidance (later)
Once the unconditional prior is trained, use your PF measurement model `h(x)` and Jacobian `J(x)` to sample from the posterior:
\[ \nabla_x \log p(x|z) = s_\theta(x) + J(x)^\top R^{-1}(z - h(x)) \]
See `sampling/sample_prior.py` for the hook.

## License
MIT (modify freely within your thesis repo).
