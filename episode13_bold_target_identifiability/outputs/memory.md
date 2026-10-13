# Memory

2026-10-02 — The zero-aware separable criterion and certificates are useful consequences of classical linear estimability. The current package does not establish independent research-paper novelty.

Independent design review found that the original adaptive CEGIS framing was
unnecessary. The exact theorem follows directly from linear-functional
estimability and the row space of a Kronecker product.

Critical repairs were to declare the full latent domain, freeze
column-major vec(X), handle zero target factors explicitly, and distinguish
the proof from finite implementation tests.
