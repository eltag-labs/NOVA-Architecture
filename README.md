# NOVA — Neural Organic Vivid Architecture

**A sub-quadratic sequence architecture combining recurrent multi-timescale memory (FLUX), periodic full-window antisymmetric attention (IRIS), and a role-separated representational substrate (RSLA).**

<p align="center">
  <img src="public.png" alt="NOVA Architecture Overview" width="100%">
</p>

> Status: Architectural and mathematical specification, released for public review.

---

## Overview

NOVA addresses three structural limitations of GPT-class Transformers:

| Limitation | GPT Mechanism | NOVA Response |
|---|---|---|
| Quadratic context complexity | O(T²·d) attention per layer | O(T·d²) FLUX recurrent state + O(T·W·d) windowed IRIS |
| Frozen associative memory | Fixed FFN weights post-training | FLUX slow path: surprise-gated test-time gradient update |
| Symmetric relational representation | dot(q,k) = dot(k,q) | Antisymmetric component, guaranteed by construction (§2.3) |

At T = 64,000 tokens (d = 1024), NOVA's FLOP arithmetic works out to roughly 20× fewer FLOPs per layer than a dense-attention GPT-class layer; the gap widens as context grows, since NOVA's cost scales linearly in T rather than quadratically. Full derivations and mathematical constructions are in the [technical specification](docs/NOVA_Architecture.pdf).

---

## Key Innovations

NOVA's architecture is built upon three foundational pillars detailed in the specification:

* **FLUX (Fast-Slow Recurrent Memory Mixer):** A dual-path recurrent structure. The fast path handles local token mixing using a gated-delta update, while the slow path leverages a surprise-gated test-time gradient update to dynamically acquire and store new associations directly at inference time.
* **IRIS (Antisymmetric Windowed Attention):** A periodic, localized attention mechanism that enforces strict relational directionality by algebraic construction rather than through learning alone, using a mathematically guaranteed antisymmetric score matrix.
* **RSLA (Role-Separated Linear Algebra):** A representational substrate that explicitly partitions every vector into four semantic roles (Salience, Directional, Relational, Contextual), each independently normalised via RSN (Role-Separated Normalisation).

---

## Architecture Structure

Every NOVA layer (a "Nova Cell") pairs a sequence mixer with a position-independent transform — the same structural template as the Transformer, with richer blocks:

- **Standard Cell** (3 layers out of 4): `FLUX → REASON`
- **Attention Cell** (1 layer out of 4, every index ≡ 0 mod 4): `IRIS → REASON`

**REASON** is NOVA's feedforward block: a two-pass SwiGLU with a learned depth gate, allowing each token to receive one or two passes of computation depending on its complexity.

**LUMEN** is the embedding front-end: it produces role-stratified embeddings, applies RoPE selectively to the Directional and Relational roles, and seeds the recurrent state with learned persistent tokens.

The 3:1 ratio of FLUX layers to IRIS layers is grounded in the independent convergence of two 2025 industrial architectures (Qwen3-Next, Kimi Linear) toward the same ratio — presented in the specification as meaningful empirical evidence, not as proof of universal optimality.

---

## Computational Advantages (Exact Arithmetic, d = 1024)

| Context Length T | GPT FLOPs (attention, per layer) | NOVA FLOPs (weighted average, per layer) | NOVA Advantage |
|---|---|---|---|
| 4K | 3.44 × 10¹⁰ | 7.73 × 10¹¹ | Comparable (FLUX overhead dominates at short T) |
| 64K | 1.84 × 10¹³ | 9.02 × 10¹¹ | ~20× fewer FLOPs |
| 512K | 1.18 × 10¹⁵ | 9.02 × 10¹¹ | ~1,300× fewer FLOPs |
| 1M | 4.72 × 10¹⁵ | 9.02 × 10¹¹ | ~5,230× fewer FLOPs |

On the memory side, the FLUX recurrent state is constant in T (≈ 6 MB regardless of context length), versus a KV cache that grows linearly with T for a dense GPT-class model (≈ 93 GB at T = 1M). Full derivations are in Sections 5.3 and 5.5 of the specification.

---

## Methodological Scope

This document explicitly distinguishes between:

- what is established by construction or by exact algebraic arithmetic (FLOP and memory counts, the guaranteed antisymmetry of the relational component);
- what remains an open empirical question, to be validated by training and evaluating real models at scale — notably the architecture's generalisation capability and the practical usefulness of the adaptive memory mechanism.

This distinction is maintained throughout the technical specification, including in its summary sections (§10 and §11).

---

## Repository Contents

- [`docs/NOVA_Architecture.pdf`](docs/NOVA_Architecture.pdf) — the complete technical specification: RSLA (Role-Separated Linear Algebra), the FLUX memory mixer, IRIS attention, dimension-selection rules, mathematical derivations, and exact FLOP/memory arithmetic.
- `figures/` — architecture diagrams referenced in the paper.

---

## Citation

If you reference this architecture or method in your research, please cite:

```bibtex
@techreport{champaney2026nova,
  author       = {Champaney, E.},
  title        = {NOVA: A Sub-Quadratic Adaptive Memory Architecture},
  institution  = {EltagAI Research},
  year         = {2026},
  url          = {https://github.com/EltagAI/NOVA-Architecture}
}
```

---

## License

This project is distributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.
