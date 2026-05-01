"""Classical SAT oracle for benchmark ground truth.

The benchmark runner compares the verifier's output against
:func:`pysat_satisfies`, which delegates to PySAT's Glucose3 solver. The
verifier's quantum or simulated output is correct iff it agrees with this
oracle.
"""

from __future__ import annotations

from pysat.formula import CNF as _PYSAT_CNF
from pysat.solvers import Glucose3

from qverify.translator.cnf import CNF


def pysat_satisfies(cnf: CNF) -> bool:
    """Return True iff the CNF is satisfiable per the Glucose3 SAT solver.

    The empty CNF (no clauses) is trivially satisfiable. A CNF containing
    an empty clause is trivially unsatisfiable. Everything else goes
    through PySAT via :meth:`CNF.to_dimacs`.
    """
    if not cnf.clauses:
        return True
    if any(len(cl.literals) == 0 for cl in cnf.clauses):
        return False

    dimacs = cnf.to_dimacs()
    pysat_cnf = _PYSAT_CNF(from_string=dimacs)
    with Glucose3(bootstrap_with=pysat_cnf) as solver:
        return bool(solver.solve())
