# TASK-035 — Colab CPU L-BFGS hotfix

**TASK035 = HOTFIX_READY / COLAB_REVALIDATION_PENDING.** EXP-028 has no
verified fold-3 result from this task. E50+L remains production.

The owner run `20261118T193037Z-0d18489a` failed at `CPU_TESTS` with
direct subprocess exit `-11` (SIGSEGV). The archived stack ends in
`scipy.optimize._lbfgsb_py._minimize_lbfgsb` line 433 during
`baseline.fit_lr` in the original objective-parity regression. The exported
ZIP and all nine member hashes verified. ZIP SHA-256 is
`6ff769458a861b798ad50b802d14a25ee5f707a37d8a8d53b614e015fb6d02da`.

## Reproduction and limits

Ubuntu WSL2 x86-64, kernel 6.6.87.2 and glibc 2.43 independently reproduced
the same native crash in the exact regression using CPython 3.12.10 and the
unchanged 16-package lock. NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.6.1
installed CPython 3.12 x86-64 manylinux 2.17/manylinux2014 wheels. Loaded
libraries were NumPy OpenBLAS 0.3.29 ILP64, SciPy OpenBLAS 0.3.28 LP64
(pthreads, Haswell dispatch in both), and scikit-learn's bundled libgomp.
The previous Colab dependency log records the same package versions. That
run did not capture its loaded native library configuration or effective
thread limits.

Every row below ran the exact regression in a fresh Linux process. BLAS and
OpenMP columns are the startup environment. The scoped limit is the value
used by the existing Model C fit inside the process.

| OPENBLAS_NUM_THREADS | OMP_NUM_THREADS | Scoped limit | Result |
| ---: | ---: | ---: | --- |
| 4 | 4 | 4 | PASS, including three additional fresh repetitions |
| 1 | 4 | 4 | SIGSEGV |
| 4 | 1 | 4 | PASS |
| 1 | 1 | 4 | SIGSEGV, including three additional fresh repetitions |
| 1 | 1 | 1 | PASS, including five additional fresh repetitions |

The one-thread startup setting alone reproduced the crash when the old
hard-coded context raised the limit to four. Varying OpenBLAS startup while
holding OpenMP fixed changed the outcome. This implicates a native BLAS
thread-limit transition in the Linux reproduction. The archived Colab
notebook requested four threads at launch, and the WSL four-to-four case
passed. The exact Colab trigger and native fault mechanism remain unknown.
The same finite CSR fixture passes in both steady profiles, and successful
imports and fits provide no evidence of an incorrect wheel architecture.
No package incompatibility or input-matrix defect was established.

In separate Linux subprocesses, a small SciPy L-BFGS-B quadratic and a weighted
sparse scikit-learn L-BFGS fit passed with both steady profiles. The 75-painting
synthetic screen produced byte-identical six-decimal predictions, evaluations,
paired uncertainty and C selections for the four-thread and one-thread
profiles, SHA-256
`8acb671edf7aa67637fa15187828867d01f937b2c6445e5e9d57c73e14a4b0fe`.
This measures synthetic parity. The real historical E50+L parity gate still
applies in the owner run.

## Hotfix and fresh run

The Colab notebook pins the runtime code at
`bbbbe3c786fc4ec541cd84b9c30f7e824bf387d1` in a new source-specific namespace.
It uses one consistent thread from startup through nested Model C contexts and
the worker. The default for existing non-Colab Model C callers remains four.
The dependency lock is unchanged. The frozen-source checks permit only the
explicit runtime-thread edits. The solver, weighted objective, C grid, PCA,
painting partitions, eight arms, matched reference and metric gates are unchanged.

The notebook runs three subprocesses in order before the full suite:
SciPy L-BFGS-B, sparse weighted sklearn LR and the exact failing regression.
The native receipt distinguishes signals such as SIGSEGV, ordinary Python or
test failures, launch failures and timeouts. The child logs include wheel
tags, backend and effective thread limits. A failing receipt stops execution
and is included in the existing checked Drive export. WSL2 Linux preflight
and the relevant 61-test suite passed under the exact lock.

Open the notebook linked in PR #33, choose a fresh **CPU** runtime and use
**Run all**. Watch for `NATIVE_PREFLIGHT` and the three zero exit codes in
`native/native_preflight.json`. A successful preflight allows the existing
full tests and verified screen to proceed. If it stops, use the new attempt's
`outputs.zip` under `MyDrive/exp028_stage1_cpu/bbbbe3c786fc/` for diagnosis.
The same Drive input paths apply. No new GPU or real fold-3 job was run during
TASK-035.
