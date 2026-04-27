"""Unit tests for the Grover diffusion operator."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest

from qverify.verifier.diffusion import build_diffusion


def _state_after(n_qubits: int, prep: callable) -> np.ndarray:  # type: ignore[type-arg]
    diffusion = build_diffusion(n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        prep()
        diffusion()
        return qml.state()

    return np.asarray(circuit())


def test_diffusion_uniform_superposition_is_invariant() -> None:
    n = 3

    def prep() -> None:
        for w in range(n):
            qml.Hadamard(wires=w)

    state = _state_after(n, prep)
    expected = np.full(2**n, 1.0 / math.sqrt(2**n))
    assert np.allclose(state.real, expected)
    assert np.allclose(state.imag, 0.0)


def test_diffusion_uniform_invariant_one_qubit() -> None:
    def prep() -> None:
        qml.Hadamard(wires=0)

    state = _state_after(1, prep)
    expected = np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
    assert np.allclose(state.real, expected)


def test_diffusion_basis_zero_one_qubit_flips_to_one() -> None:
    # D|0⟩ = (2|s⟩⟨s|0⟩ - I)|0⟩ = 2/√2 |s⟩ - |0⟩ = |1⟩  for n=1
    state = _state_after(1, lambda: None)
    assert np.isclose(state[0].real, 0.0)
    assert np.isclose(state[1].real, 1.0)


def test_diffusion_basis_zero_two_qubits() -> None:
    # n=2, N=4. D|00⟩ = -1/2 |00⟩ + 1/2 (|01⟩+|10⟩+|11⟩)
    state = _state_after(2, lambda: None)
    assert np.isclose(state[0].real, -0.5)
    assert np.isclose(state[1].real, 0.5)
    assert np.isclose(state[2].real, 0.5)
    assert np.isclose(state[3].real, 0.5)


def test_diffusion_preserves_norm() -> None:
    for n in (1, 2, 3, 4):
        state = _state_after(n, lambda: None)
        assert math.isclose(float(np.linalg.norm(state)), 1.0, abs_tol=1e-9)


def test_diffusion_self_inverse() -> None:
    # D² = I (D is reflection about |s⟩, reflections are involutions)
    n = 3
    diffusion = build_diffusion(n)
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        qml.Hadamard(wires=0)  # arbitrary non-trivial input state
        qml.Hadamard(wires=2)
        diffusion()
        diffusion()
        return qml.state()

    @qml.qnode(dev)
    def reference() -> np.ndarray:
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=2)
        return qml.state()

    assert np.allclose(np.asarray(circuit()), np.asarray(reference()))


def test_diffusion_zero_qubits_raises() -> None:
    with pytest.raises(ValueError, match="n_qubits >= 1"):
        build_diffusion(0)
