"""Fail-closed isolated native preflight for the EXP-028 CPU environment."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
THREAD_SETTINGS = dict.fromkeys(
    ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
     "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "ST3_EXP028_CPU_THREADS"), "1",
)
REGRESSION = "tests/test_exp028_stage1.py::test_objective_matches_official_lr_and_unseen_fallback"


def smoke(case):
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_info

    wheel_tags = {
        name: [line for line in importlib.metadata.distribution(name).read_text("WHEEL").splitlines()
               if line.startswith("Tag:")]
        for name in ("numpy", "scipy", "scikit-learn")
    }
    pools = threadpool_info()
    print(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
                      "architecture": platform.machine(), "libc": platform.libc_ver(),
                      "versions": {n: importlib.metadata.version(n) for n in wheel_tags},
                      "wheel_tags": wheel_tags, "threadpools": pools,
                      "thread_environment": {k: os.environ.get(k) for k in THREAD_SETTINGS}}, indent=2), flush=True)
    if any(p["num_threads"] != 1 for p in pools):
        raise RuntimeError("Effective native thread limit differs from the one-thread profile")
    if case == "scipy":
        scipy.show_config()
        target = np.array([2., -3., 4.])
        result = minimize(lambda x: (float(np.dot(x-target, x-target)), 2*(x-target)),
                          np.zeros(3), jac=True, method="L-BFGS-B")
        if not result.success or not np.allclose(result.x, target, atol=1e-10, rtol=0):
            raise RuntimeError("SciPy L-BFGS-B did not reach the known optimum")
    else:
        x = csr_matrix(np.array([[-2., 0.], [-1., 1.], [1., 1.], [2., 0.]]))
        lr = LogisticRegression(C=0.3, solver="lbfgs", max_iter=3000)
        lr.fit(x, [0, 0, 1, 1], sample_weight=[3, 4, 2, 5])
        if not np.array_equal(lr.predict(x), [0, 0, 1, 1]) or not np.isfinite(lr.coef_).all():
            raise RuntimeError("Sparse weighted sklearn L-BFGS fit differs from expectation")


def outcome(returncode):
    if returncode == 0:
        return {"exit_code": 0, "status": "PASS"}
    if returncode < 0:
        return {"exit_code": returncode, "status": "NATIVE_SIGNAL", "signal": signal.Signals(-returncode).name}
    return {"exit_code": returncode, "status": "PYTHON_OR_TEST_FAILURE"}


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **THREAD_SETTINGS, CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    commands = {
        "scipy": [str(Path(__file__).resolve()), "--child", "scipy"],
        "sklearn": [str(Path(__file__).resolve()), "--child", "sklearn"],
        "original_regression": ["-m", "pytest", "-q", REGRESSION],
    }
    report = {"status": "RUNNING", "checks": {}, "thread_environment": THREAD_SETTINGS}
    receipt = output / "native_preflight.json"

    def save():
        pending = receipt.with_suffix(".part")
        pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        pending.replace(receipt)

    save()
    for name, command in commands.items():
        log = output / (name + ".log")
        with log.open("w", encoding="utf-8") as stream:
            try:
                process = subprocess.run([sys.executable, "-X", "faulthandler", "-u", *command],
                                         cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120)
                result = outcome(process.returncode)
            except subprocess.TimeoutExpired:
                result = {"status": "TIMEOUT"}
            except OSError as error:
                result = {"status": "LAUNCH_FAILURE", "error": str(error)}
        report["checks"][name] = dict(result, log=log.name)
        if result["status"] != "PASS":
            report["status"] = "BLOCKED_VERIFICATION"
            save()
            print(f"{name}: {result}. See {log}", flush=True)
            return 2
        save()
    report["status"] = "PASS"
    save()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--child", choices=["scipy", "sklearn"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        smoke(args.child)
        return 0
    return run(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
