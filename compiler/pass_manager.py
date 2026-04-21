"""
PassManager infrastructure mirroring Qiskit's transpiler architecture.

Structure mirrors qiskit/transpiler/passmanager.py and basepasses.py:
  - PropertySet: shared dict passed between all passes in one run()
  - BasePass / AnalysisPass / TransformationPass: same hierarchy as Qiskit's
  - PassManager: ordered list of passes; injects property_set before each run
"""

from abc import ABC, abstractmethod
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import (
    AnalysisPass as _QiskitAnalysisPass,
    TransformationPass as _QiskitTransformationPass,
)


class PropertySet(dict):
    """
    Shared analysis results passed between compiler passes.

    Mirrors qiskit.transpiler.propertyset.PropertySet. Analysis passes write
    to it; transformation passes read from it. One instance lives for the
    lifetime of a single PassManager.run() call.
    """
    pass


class BasePass(ABC):
    """Abstract base for all compiler passes (mirrors Qiskit's BasePass)."""

    def __init__(self):
        self.property_set: PropertySet = PropertySet()
        self.requires: list = []
        self.preserves: list = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def is_analysis_pass(self) -> bool:
        return isinstance(self, AnalysisPass)

    @property
    def is_transformation_pass(self) -> bool:
        return isinstance(self, TransformationPass)

    def __str__(self) -> str:
        return self.name


class AnalysisPass(BasePass, ABC):
    """
    Reads the DAG and writes results into property_set. Does not modify the
    circuit. Mirrors qiskit.transpiler.basepasses.AnalysisPass.
    """

    @abstractmethod
    def run(self, dag: DAGCircuit):
        pass


class TransformationPass(BasePass, ABC):
    """
    Reads the DAG (and optionally property_set), returns a modified DAG.
    Mirrors qiskit.transpiler.basepasses.TransformationPass.
    """

    @abstractmethod
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        pass


class PassManager:
    """
    Ordered sequence of passes applied to a QuantumCircuit.

    Mirrors qiskit.transpiler.PassManager. The key contract:
      1. circuit → DAGCircuit via circuit_to_dag()
      2. Each pass receives the shared property_set before run()
      3. TransformationPass.run() → new DAG replaces current DAG
      4. AnalysisPass.run() → results accumulate in property_set
      5. DAGCircuit → circuit via dag_to_circuit()

    Also accepts Qiskit's own pass classes (SabreSwap, TrivialLayout, etc.)
    in the pass list — they follow the same run(dag) interface.
    """

    def __init__(self, passes=None):
        self._passes: list = []
        self.property_set: PropertySet = PropertySet()
        # Set when wrapping a Qiskit StagedPassManager (preset levels 0-3)
        self._qiskit_pm = None

        if passes is not None:
            self.append(passes)

    @classmethod
    def from_qiskit(cls, qiskit_pm) -> 'PassManager':
        """Wrap a Qiskit PassManager / StagedPassManager in our interface."""
        pm = cls()
        pm._qiskit_pm = qiskit_pm
        return pm

    def append(self, passes) -> 'PassManager':
        """Add one pass or a list of passes."""
        if isinstance(passes, (list, tuple)):
            self._passes.extend(passes)
        else:
            self._passes.append(passes)
        return self

    def run(self, circuit, **kwargs):
        """
        Transpile *circuit* through all passes and return the compiled circuit.

        When wrapping a Qiskit preset PM, delegates directly to it (which
        handles all the internal Qiskit infrastructure like RunningPassManager).
        When using our own pass list, runs the loop manually.
        """
        if self._qiskit_pm is not None:
            result = self._qiskit_pm.run(circuit)
            # Sync property_set so callers can inspect analysis results
            if hasattr(self._qiskit_pm, 'property_set') and self._qiskit_pm.property_set:
                self.property_set.update(self._qiskit_pm.property_set)
            return result

        dag = circuit_to_dag(circuit)

        for pass_ in self._passes:
            # Inject our shared property_set into the pass before execution,
            # matching Qiskit's RunningPassManager behaviour.
            pass_.property_set = self.property_set

            if isinstance(pass_, (_QiskitTransformationPass, TransformationPass)):
                new_dag = pass_.run(dag)
                if new_dag is not None:
                    dag = new_dag
            elif isinstance(pass_, (_QiskitAnalysisPass, AnalysisPass)):
                pass_.run(dag)

        return dag_to_circuit(dag)

    def passes(self) -> list:
        """Return a human-readable list of pass names in execution order."""
        if self._qiskit_pm is not None:
            return [str(p) for stage in self._qiskit_pm.stages
                    for p in (self._qiskit_pm.__getattribute__(stage).passes()
                               if self._qiskit_pm.__getattribute__(stage) else [])]
        return [str(p) for p in self._passes]

    def __len__(self) -> int:
        if self._qiskit_pm is not None:
            return sum(
                len(getattr(self._qiskit_pm, s).passes())
                for s in self._qiskit_pm.stages
                if getattr(self._qiskit_pm, s) is not None
            )
        return len(self._passes)

    def __repr__(self) -> str:
        if self._qiskit_pm is not None:
            return f"PassManager(qiskit_pm={type(self._qiskit_pm).__name__})"
        return f"PassManager([{', '.join(str(p) for p in self._passes)}])"
