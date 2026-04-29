---
title: QVerify
emoji: 🔬
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
hardware: cpu-basic
tags:
  - quantum-computing
  - qiskit
  - pennylane
  - llm-verification
  - logic
  - grover
---

# QVerify (verifier-only Space)

This Space exposes the QVerify verifier component: a CNF is fed
straight to Grover's search, on either a CPU-side PennyLane simulator
or IBM Quantum's Heron r2 processor.

The translator (Gemma 4 E4B with grammar-constrained generation)
requires a GPU and is not loaded here. The full pipeline (translator
plus grounding plus verifier) runs locally from the GitHub repo.

[Code](https://github.com/Quantum-Labor/qverify) ·
[Documentation](https://github.com/Quantum-Labor/qverify/tree/main/docs)
