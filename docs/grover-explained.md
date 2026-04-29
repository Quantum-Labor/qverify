# Grover's algorithm in QVerify

*Last updated: 2026-04-29*

Grover's algorithm searches a database of `N = 2^n` items for a marked subset of `M` items in roughly `π/4 · √(N/M)` queries, instead of the `N/2` expected of a classical brute-force search. In QVerify the "database" is the assignment space of a Boolean formula and the "marked items" are the satisfying assignments — finding one is the same as finding a counter-model that witnesses an inconsistency between the LLM's reasoning step and the surrounding premises.

## The four ingredients

1. **State preparation.** Apply Hadamards to put the assignment register into a uniform superposition over all `2^n` possible truth assignments.
2. **Oracle.** A unitary that flips the phase of every basis state corresponding to a satisfying assignment, leaving the rest unchanged. We build it from the CNF directly: each clause is OR'd into a per-clause ancilla, all clause-ancillas are AND'd into a flag qubit, the flag is phase-flipped, then everything is uncomputed in reverse so the ancillas return to `|0⟩`.
3. **Diffusion.** The Grover diffusion operator `D = 2|s⟩⟨s| - I` reflects the state about the uniform-superposition vector `|s⟩`. Together with the oracle reflection, one Grover iteration rotates the state vector by `2θ` (where `sin(θ) = √(M/N)`) towards the "marked" subspace.
4. **Measurement.** After the right number of iterations the marked states have near-unit amplitude. Measuring the assignment register collapses (with high probability) onto a satisfying assignment.

## Iteration count

The optimal number of iterations is `k* = round(π/4 · √(N/M))`. With `M = 1` and `N = 2^n`, this is `round(π/4 · 2^(n/2))`. Examples used in QVerify:

| `n` | `N` | `k*` (assuming M=1) |
|-----|-----|---------------------|
| 1   | 2   | 1                   |
| 2   | 4   | 2                   |
| 3   | 8   | 2                   |
| 4   | 16  | 3                   |
| 6   | 64  | 6                   |
| 12  | 4096| 50                  |

**Over-iteration is destructive.** The amplitude of the marked subspace evolves as `sin((2k+1)θ)` — past `k*` it overshoots and starts dropping. We always estimate `M = 1` because the verifier rarely knows the true count up front; this is conservative for small `M` and harmless because the classical post-check (below) catches any noise.

## Why classical post-verification is mandatory

Quantum measurement is probabilistic, and Grover with the wrong iteration count, or on an UNSAT formula, returns near-uniform noise. We never trust the most-frequent measurement directly. Instead, we run the measured bitstrings through the deterministic [`satisfies()`](../qverify/verifier/classical_check.py) check: if the top bitstring is a true satisfying assignment, we report `contradiction_found=True`; otherwise we walk down the count ranking and try the next ones. If no measured assignment satisfies the formula, we report UNSAT.

This is standard Grover practice. The quantum advantage is in *finding* the candidate quickly; *verifying* it is cheap and classical.

## Phase 3 limits

- 16 propositional variables, hard cap. The state-vector simulator's memory grows as `2^(n + n_clauses + 1)`, so even at the cap practical instances stay small.
- Ground atoms only. Free first-order variables (single-letter or short lowercase tokens) are rejected at the encoder; substitute concrete constants before calling `verify()`.
- PennyLane `default.qubit` simulator and IBM Quantum's Heron r2 hardware (verified job `d7o7dsqk4prs73dt4s6g` on `ibm_fez`, 2026-04-28). The simulator is the inner-loop default; hardware is opt-in. See [`docs/quantum-hardware-notes.md`](quantum-hardware-notes.md).

## Worked example: the penguin CNF

Consider the three-clause CNF the spec calls "the penguin paradox":

```
(Bird(penguin)) ∧ (¬Flies(penguin)) ∧ (¬Bird(penguin) ∨ Flies(penguin))
```

- **Encoding:** two ground atoms, sorted lexicographically — `Bird(penguin) → qubit 0`, `Flies(penguin) → qubit 1`.
- **Iteration count:** `n = 2`, `N = 4`, optimal `k* = 2`.
- **Outcome:** classically, no assignment satisfies all three clauses (Bird=T forces Flies=T from clause 3, but clause 2 forbids it). The simulator's measured histogram is approximately uniform over the four bitstrings; the classical post-check confirms none satisfy. `verify()` returns `VerificationResult(contradiction_found=False, counter_model=None, n_variables=2, n_clauses=3, n_grover_iterations=2, …)`.

In QVerify's framing this is the "step is entailed" case: `premises ∧ ¬step` was unsatisfiable, so the negated step contradicts the premises, so the step itself is forced. The LLM's reasoning move was logically valid.

By contrast, `(P ∨ Q) ∧ (¬P ∨ Q)` is satisfiable — both satisfying assignments have `Q = True`. `verify()` returns `contradiction_found=True` with a counter-model showing the negated step (`¬Q`) cannot hold.
