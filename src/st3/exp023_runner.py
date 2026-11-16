"""Frozen EXP-023: one automatic pipeline, persistently sealed model boundaries."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import pandas as pd
import torch

from . import data, submission
from .ablation import predict_lr
from .caption_data import inner_partition, require_isolation
from .caption_student import CaptionStudent, fit_student, make_rows, predict_student, seed_model, train_epoch, validation_bce
from .e50_heads import fit_frozen_head, head_from_arrays
from .e50_inference import read_input, safe_image_path
from .exp023_data import load_inputs, read_confirmation, read_csv, exact
from .exp023_evaluation import aggregate, analyze, scientific_gate
from .exp023_targets import prepare_partitions
from .exp023_budget import condition_cost, forecast_work
from .experiment_workflow import (Budget, StopRun, execute_stages, fingerprint, publish_model, sha256, verify_model, write_json,
                                  RESEARCH_ONLY_CV, COMPETITION_READY, execution_scope, eligibility_evidence,
                                  scoped_cv_decision, require_run_identity)

ROOT = Path(__file__).resolve().parents[2]


def verify_eligibility(evidence, owner_status, fetch=None):
    if owner_status != "VERIFIED" or evidence.get("status") != "VERIFIED_ORGANIZER_PERMISSION":
        raise StopRun("BLOCKED_ELIGIBILITY", "Training-only ST3 caption supervision is unresolved; a checkbox cannot replace organizer evidence")
    if evidence.get("covers") != ["ST3", "original_training_captions_as_auxiliary_targets", "pinned_pretrained_CLIP_L_text"]:
        raise StopRun("BLOCKED_ELIGIBILITY", "Evidence does not cover all EXP-023 resources/procedures")
    url = evidence.get("url", "")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not ((parsed.hostname == "github.com" and parsed.path.startswith("/artemotion-28/artemotion-28/"))
                                          or parsed.hostname == "www.codabench.org"):
        raise StopRun("BLOCKED_ELIGIBILITY", "Eligibility must cite an authoritative organizer source")
    if fetch is None:
        def fetch(address):
            with urlopen(address, timeout=30) as handle:
                return handle.read()
    raw = fetch(url)
    import hashlib
    if (hashlib.sha256(raw).hexdigest() != evidence.get("source_sha256") or not evidence.get("exact_permission_excerpt")
            or evidence["exact_permission_excerpt"] not in raw.decode("utf-8")):
        raise StopRun("BLOCKED_ELIGIBILITY", "Organizer evidence could not be independently verified")
    return {"status": "PASS", "url": url, "source_sha256": evidence["source_sha256"]}


def runtime_identity(device):
    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "threadpoolctl")}
    return {"python": platform.python_version(), "packages": packages, "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
            "deterministic": torch.are_deterministic_algorithms_enabled(), "matmul_precision": torch.get_float32_matmul_precision()}


class Experiment:
    def __init__(self, inputs, run_dir, config, authorization, *, device="cuda", started=None):
        self.inputs, self.run_dir = Path(inputs), Path(run_dir)
        self.config, self.authorization = config, authorization
        self.execution_scope = execution_scope(authorization)
        self.competition_eligible = False
        self.device = torch.device(device)
        self.operations = json.loads((ROOT / "configs/exp023_runtime.json").read_text())
        gpu_hours, wall_hours = authorization["MAX_GPU_HOURS"], authorization["MAX_TOTAL_WALL_HOURS"]
        self.budget = Budget(gpu_hours, wall_hours, cuda=device == "cuda", reserve=self.operations["persistence_reserve_seconds"], started=started)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.verified = 0
        self.final_verified = 0
        self.timing_samples = None

    def progress(self, stage, **details):
        row = {"stage": stage, "elapsed_seconds": time.time() - self.budget.started,
               "execution_scope": self.execution_scope,
               "remaining_allocation_seconds": self.budget.remaining(), "gpu_hours_cap": self.authorization["MAX_GPU_HOURS"],
               "wall_hours_cap": self.authorization["MAX_TOTAL_WALL_HOURS"],
               "estimated_remaining_work_seconds": getattr(self, "forecast_remaining", None),
               "runtime_estimate_basis": getattr(self, "forecast_basis", "Benchmark not yet complete"), **details}
        print(json.dumps(row, sort_keys=True), flush=True)
        with (self.run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        self.budget.check()

    def validate(self):
        if self.authorization.get("AUTHORIZE_EXPERIMENT") is not True:
            raise StopRun("BLOCKED_AUTHORIZATION", "Advance owner GPU authorization and workflow review are required in either scope")
        if self.execution_scope == COMPETITION_READY:
            evidence, receipt = eligibility_evidence(self.authorization, ROOT / "configs/exp023_eligibility.json")
            proof = {**verify_eligibility(evidence, self.authorization["CAPTION_ELIGIBILITY_STATUS"]), "evidence_document": receipt}
            self.competition_eligible = True
        else:
            proof = {"status": "RESEARCH_ONLY_NOT_COMPETITION_ELIGIBLE", "competition_eligibility": "NOT_ESTABLISHED",
                     "resource_scope": "Pinned released official training artifacts only; no new data/checkpoints"}
        write_json(self.run_dir / "eligibility.json", proof)
        write_json(self.run_dir / "authorization_history" / f"{time.time_ns()}.json",
                   {"execution_scope": self.execution_scope, "authorization": self.authorization, "eligibility": proof})
        if self.authorization["REQUIRE_SUBMISSION_ZIP"] and not self.authorization["AUTHORIZE_GUARDED_CONFIRMATION"]:
            raise StopRun("BLOCKED_AUTHORIZATION", "Submission scope needs guarded confirmation authorization in the initial Configuration")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise StopRun("BLOCKED_INPUT", "CUDA GPU unavailable; select a GPU before Run all")
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        self.runtime = runtime_identity(self.device)
        if self.runtime["python"] != "3.12.10":
            raise StopRun("BLOCKED_INPUT", "The pinned Python 3.12.10 environment is required")
        required = {"torch": "2.7.1+cu126", "numpy": "2.2.6", "pandas": "2.2.3", "scikit-learn": "1.6.1", "scipy": "1.15.3", "threadpoolctl": "3.6.0"}
        if self.device.type == "cuda" and self.runtime["packages"] != required:
            raise StopRun("BLOCKED_INPUT", "Scientific dependency versions differ; no automatic substitution")
        (self.train, self.captions, self.vectors, self.banks, self.visual,
         self.protocol, historical) = load_inputs(self.inputs, self.config)
        self.r0 = read_csv(self.inputs / "history/exp022_R0.csv")
        self.eval_csv = Path(self.authorization["EVAL_INPUT_CSV"] or self.inputs / "evaluation/st3_test.csv")
        self.eval_dir = Path(self.authorization["EVAL_DATA_DIR"] or self.inputs / "evaluation")
        if self.authorization["REQUIRE_SUBMISSION_ZIP"]:
            official = json.loads((ROOT / "configs/e50_inference.json").read_text())["official_languages"]
            self.eval_inputs = read_input(self.eval_csv, official)
            expected = read_csv(self.inputs / "evaluation/st3_test.csv")
            if not self.eval_inputs.equals(expected) or len(expected) != 849:
                raise StopRun("BLOCKED_INPUT", "Only the pinned official Development input-only rows are authorized")
            image_hashes = json.loads((self.inputs / "visual/image_hashes.json").read_text())
            for image in self.eval_inputs.image.unique():
                if sha256(safe_image_path(self.eval_dir, image)) != image_hashes[image] or image not in self.visual:
                    raise StopRun("BLOCKED_INPUT", "Official image bytes or frozen feature coverage differ")
        manifest = json.loads((self.inputs / "input_manifest.json").read_text())
        source = {name: sha256(ROOT / name) for name in manifest["source_files_sha256"]}
        executable = {p.relative_to(ROOT).as_posix() for directory in ("src", "scripts", "tests")
                      for p in (ROOT / directory).rglob("*.py")}
        if not executable <= set(source):
            raise StopRun("BLOCKED_INPUT", "Additional unpinned executable source files")
        if manifest["config_sha256"] != sha256(ROOT / "configs/exp023.json") or source != manifest["source_files_sha256"]:
            raise StopRun("BLOCKED_INPUT", "Pinned source or scientific config differs from handoff")
        self.identity = {"source_revision": manifest["source_revision"], "source": source, "config": fingerprint(self.config),
                         "inputs": sha256(self.inputs / "input_manifest.json"), "runtime": self.runtime}
        identity_path = self.run_dir / "run_identity.json"
        if identity_path.exists():
            require_run_identity(json.loads(identity_path.read_text()), self.identity)
        write_json(identity_path, self.identity)
        write_json(self.run_dir / "historical_parity.json", historical)
        # Revoke prior successful decisions until the current complete replay passes.
        write_json(self.run_dir / "decision.json", {"status": "VERIFICATION_PENDING", "production": "E50+L"})
        return historical

    def partition(self, fold):
        fitting = self.train if fold == 0 else self.train[self.train.fold.ne(fold)]
        held = self.train.iloc[:0] if fold == 0 else self.train[self.train.fold.eq(fold)]
        inner, stopping = inner_partition(fitting, self.config, fold)
        require_isolation(inner, stopping, held, self.protocol[self.protocol.role.eq("holdout")])
        return fitting, held, inner, stopping

    def targets(self, fold):
        fitting, _, inner, _ = self.partition(fold)
        tick = time.perf_counter()
        prepared = prepare_partitions(self.captions, fitting, inner, self.vectors, self.config)
        path = self.run_dir / "targets" / f"fold_{fold}"
        identity = {**self.identity, "fold": fold, "fitting_paintings": sorted(fitting.painting.unique())}
        def writer(stage):
            for phase in ("inner", "full"):
                original, weights, pca, records, metadata = prepared["C1"][phase]
                centered = prepared["C2"][phase][0]
                keys = sorted(weights)
                np.savez(stage / f"{phase}.npz", components=pca.components_, mean=pca.mean_, keys=np.array(keys).reshape(-1, 2),
                         original=np.array([original[k] for k in keys], dtype=np.float32).reshape(-1, self.config["target_dimensions"]),
                         centered=np.array([centered[k] for k in keys], dtype=np.float32).reshape(-1, self.config["target_dimensions"]))
                write_json(stage / f"{phase}.json", metadata)
                records.to_csv(stage / f"{phase}_records.csv", index=False, lineterminator="\n")
        existed = path.exists()
        publish_model(path, identity, writer)
        # Independently regenerated transforms/masks must still match a resumed stage.
        for phase in ("inner", "full"):
            result = prepared["C1"][phase]
            with np.load(path / f"{phase}.npz", allow_pickle=False) as saved:
                if not np.array_equal(saved["components"], result[2].components_) or not np.array_equal(saved["mean"], result[2].mean_):
                    raise StopRun("FAILED_VERIFICATION", "Partition-local PCA replay differs exactly")
            if json.loads((path / f"{phase}.json").read_text()) != result[4]:
                raise StopRun("FAILED_VERIFICATION", "Common target mask/PCA metadata changed")
        return prepared, time.perf_counter() - tick

    def benchmark(self):
        previous = self.run_dir / "benchmark.json"
        if previous.exists():
            recorded = json.loads(previous.read_text())
            if recorded.get("identity") != self.identity:
                raise StopRun("FAILED_VERIFICATION", "Completed benchmark identity changed")
            self.scoring_seconds = recorded["scoring_seconds"]
            self.scoring_breakdown = recorded["scoring_breakdown"]
            return self.forecast(recorded["samples"], recorded["target_seconds"], reused=True)
        fitting, held, inner, stopping = self.partition(1)
        prepared, target_seconds = self.targets(1)
        epochs = self.operations["benchmark_epochs"]
        samples = []
        baseline = held[data.KEYS].merge(self.r0[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
        for condition in self.config["conditions"]:
            targets, weights = prepared.get(condition, {}).get("inner", (None, None))[:2]
            languages = sorted(inner.language.unique())
            rows = make_rows(inner, self.visual, languages, self.device, targets, weights)
            model, rng, optimizer = seed_model(languages, 2027, self.config, self.device)
            validation = make_rows(stopping, self.visual, languages, self.device)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            training_times, validation_times, refit_times = [], [], []
            for epoch in range(epochs):
                self.progress("Benchmark", condition=condition, epoch=epoch + 1, fold=1, seed=2027)
                _, elapsed = self.timed(lambda: train_epoch(model, rows, rng, optimizer, self.config, self.config["text_weight"] if condition != "C0" else 0.))
                training_times.append(elapsed)
                _, elapsed = self.timed(lambda: validation_bce(model, validation, self.config["batch_size"]))
                validation_times.append(elapsed)
            del model, optimizer, rows, validation
            # Fresh reset, original full-partition batches; diagnostic epochs are not CV fits.
            languages = sorted(fitting.language.unique())
            targets, weights = prepared.get(condition, {}).get("full", (None, None))[:2]
            rows = make_rows(fitting, self.visual, languages, self.device, targets, weights)
            model, rng, optimizer = seed_model(languages, 2027, self.config, self.device)
            for epoch in range(epochs):
                self.progress("BenchmarkRefit", condition=condition, epoch=epoch + 1, fold=1, seed=2027)
                _, elapsed = self.timed(lambda: train_epoch(model, rows, rng, optimizer, self.config, self.config["text_weight"] if condition != "C0" else 0.))
                refit_times.append(elapsed)
            raw, replay_seconds = self.timed(lambda: predict_student(model, held[data.INPUT_COLUMNS], self.visual, baseline, self.device, self.config["batch_size"]))
            with tempfile.TemporaryDirectory(prefix=".pending-benchmark-", dir=self.run_dir) as temp:
                checkpoint = Path(temp) / "diagnostic.pt"
                _, checkpoint_seconds = self.timed(lambda: torch.save(model.state_dict(), checkpoint))
                _, reload_seconds = self.timed(lambda: torch.load(checkpoint, map_location="cpu", weights_only=True))
            seconds = max(training_times)
            samples.append({"condition": condition, "full_inner_epoch_seconds": seconds, "pairs": len(inner),
                            "full_pairs": len(fitting), "inner_training_epoch_seconds": max(training_times),
                            "validation_epoch_seconds": max(validation_times), "full_refit_epoch_seconds": max(refit_times),
                            "inner_epoch_samples_seconds": training_times, "validation_samples_seconds": validation_times,
                            "refit_epoch_samples_seconds": refit_times, "replay_seconds": replay_seconds + reload_seconds,
                            "checkpoint_seconds": checkpoint_seconds,
                            "pairs_per_second": len(inner) / seconds,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else None,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if self.device.type == "cuda" else None})
            del model, optimizer, rows
        # Timing only: constant predictions on released CV training rows, not pilot system scores.
        from .metrics import score
        _, score_seconds = self.timed(lambda: score(self.train, self.train[data.KEYS].assign(p_positive=.5)))
        from .exp023_evaluation import paired_interval
        # Time the original bootstrap kernel on neutral synthetic differences; no
        # held-out system performance is computed or used for runtime selection.
        _, bootstrap_seconds = self.timed(lambda: paired_interval(np.zeros(self.train.painting.nunique()),
                                            self.config["bootstrap_replicates"], self.config["bootstrap_seed"]))
        self.scoring_breakdown = {"single_full_population_score_seconds": score_seconds,
                                  "single_bootstrap_seconds": bootstrap_seconds,
                                  "scoring_calls_allowance": 87, "bootstrap_calls": 3}
        self.scoring_seconds = max(self.operations["minimum_final_scoring_seconds"], 87 * score_seconds + 3 * bootstrap_seconds)
        return self.forecast(samples, target_seconds)

    def forecast(self, samples, target_seconds, *, reused=False):
        self.timing_samples, self.target_seconds = samples, target_seconds
        completed, remaining = self.completed_timings()
        scoring = getattr(self, "scoring_seconds", self.operations["minimum_final_scoring_seconds"])
        overhead = 12 * target_seconds + scoring + self.operations["minimum_export_seconds"]
        if self.execution_scope == COMPETITION_READY:
            overhead += 3 * self.operations["baseline_component_seconds"]
        estimate = forecast_work(samples, remaining, completed, self.config, self.operations,
                                 fit_pairs=len(self.train), target_seconds=target_seconds, overhead=overhead)
        result = {"status": "BENCHMARK_ONLY_NOT_HELD_OUT_PERFORMANCE", "identity": self.identity,
                  "reused_verified_timing": reused, "epochs_per_condition": self.operations["benchmark_epochs"], "samples": samples,
                  "target_seconds": target_seconds, "scoring_seconds": scoring, **estimate,
                  "scoring_breakdown": getattr(self, "scoring_breakdown", {}),
                  "persistence_reserve_seconds": self.budget.reserve,
                  "remaining_authorized_seconds": self.budget.remaining(),
                  "admission_policy": "One complete fit plus verification/persistence and cleanup reserve; global forecast is diagnostic only"}
        write_json(self.run_dir / "benchmark.json", result)
        with (self.run_dir / "forecast_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": time.time(), **result}) + "\n")
        self.forecast_remaining = estimate["estimated_remaining_seconds"]
        self.forecast_basis = "Same-condition completed-fit means with max-epoch fallback for unseen conditions" if completed else "No completed fits yet; max-epoch planning proxy, not measured expected duration"
        print(json.dumps({"stage": "Forecast", "theoretical_worst_case_seconds": estimate["theoretical_worst_case_seconds"],
                          "measured_expected_remaining_seconds": estimate["measured_expected_remaining_seconds"],
                          "verified_fit_observations": len(completed), "remaining_fits": len(remaining),
                          "remaining_authorized_seconds": self.budget.remaining(), "basis": self.forecast_basis}), flush=True)
        self.budget.check()  # Never reject all CV because its theoretical total exceeds this session.
        return result

    def timed(self, operation):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        value = operation()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return value, time.perf_counter() - tick

    def completed_timings(self):
        completed, remaining = [], []
        folds = list(range(1, 6))
        if self.execution_scope == COMPETITION_READY and self.authorization["AUTHORIZE_GUARDED_CONFIRMATION"]:
            folds.append(0)
        for fold in folds:
            for condition in self.config["conditions"]:
                for seed in self.config["seeds"]:
                    path = self.run_dir / "models" / f"fold_{fold}_{condition}_{seed}"
                    if not path.exists():
                        remaining.append((fold, condition, seed))
                        continue
                    verify_model(path, self.model_identity(fold, condition, seed))
                    info = json.loads((path / "fit.json").read_text())
                    verified = json.loads((path / "initial_replay.json").read_text())
                    if verified.get("status") != "PASS" or verified["checkpoint_sha256"] != sha256(path / "checkpoint.pt"):
                        raise StopRun("FAILED_VERIFICATION", "Completed model has no valid initial independent replay")
                    completed.append({**info, "fold": fold, "condition": condition, "seed": seed,
                                      "end_to_end_seconds": info["fit_seconds"] + info["predict_seconds"]
                                      + info["checkpoint_write_seconds"] + verified["seconds"]})
        return completed, remaining

    def admit_fit(self, fold, condition, seed, fit_pairs):
        if self.timing_samples is None:
            raise StopRun("FAILED_VERIFICATION", "A matching representative timing benchmark is required before a fit")
        completed, _ = self.completed_timings()
        sample = next(r for r in self.timing_samples if r["condition"] == condition)
        cost = condition_cost(sample, self.config["max_epochs"], fit_pairs, self.operations,
                              [r for r in completed if r["condition"] == condition])
        write_json(self.run_dir / "admission_history" / f"{time.time_ns()}.json",
                   {"fold": fold, "condition": condition, "seed": seed, "remaining_seconds": self.budget.remaining(),
                    "persistence_reserve_seconds": self.budget.reserve, **cost})
        self.budget.check(cost["admission_seconds"])

    def model_identity(self, fold, condition, seed):
        fitting, held, inner, stopping = self.partition(fold)
        return {**self.identity, "fold": fold, "condition": condition, "seed": seed,
                "partitions": {key: sorted(frame.painting.unique()) for key, frame in
                               (("fit", fitting), ("held", held), ("inner", inner), ("stopping", stopping))}}

    def fit_one(self, fold, condition, seed, prepared, baseline, held):
        if fold == 0:
            self.require_competition_scope()
        path = self.run_dir / "models" / f"fold_{fold}_{condition}_{seed}"
        identity = self.model_identity(fold, condition, seed)
        fitting, _, inner, stopping = self.partition(fold)
        if not path.exists():
            self.admit_fit(fold, condition, seed, len(fitting))
            self.progress("Fit", fold=fold, condition=condition, seed=seed)
        def writer(stage):
            timings = {"selection_training_seconds": 0., "refit_training_seconds": 0.}
            phase = ["selection"]
            def progress(which, epoch):
                phase[0] = which
                self.progress(which, fold=fold, condition=condition, seed=seed, epoch=epoch)
            def measured_epoch(*args):
                value, seconds = self.timed(lambda: train_epoch(*args))
                timings[phase[0] + "_training_seconds"] += seconds
                return value
            model, curves, info = fit_student(fitting, inner, stopping, self.visual, prepared.get(condition), seed,
                                             self.config, self.device,
                                             progress=progress, epoch_function=measured_epoch)
            timings["validation_stopping_and_setup_seconds"] = max(0., info["fit_seconds"] - sum(timings.values()))
            raw, predict_seconds = self.timed(lambda: predict_student(model, held[data.INPUT_COLUMNS], self.visual, baseline, self.device, self.config["batch_size"]))
            tick = time.perf_counter()
            torch.save({"state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                        "languages": list(model.languages)}, stage / "checkpoint.pt")
            np.save(stage / "raw_predictions.npy", raw, allow_pickle=False)
            held[data.KEYS].to_csv(stage / "held_keys.csv", index=False, lineterminator="\n")
            pd.DataFrame(curves).to_csv(stage / "curves.csv", index=False, lineterminator="\n")
            checkpoint_seconds = time.perf_counter() - tick
            self.budget.check()
            (_, replay), replay_seconds = self.timed(lambda: self.replay_payload(stage, baseline, held))
            write_json(stage / "initial_replay.json", {**replay, "seconds": replay_seconds})
            write_json(stage / "fit.json", {**info, "timings": timings, "predict_seconds": predict_seconds,
                                             "checkpoint_write_seconds": checkpoint_seconds})
        publish_model(path, identity, writer)
        # Reused complete fits must retain their original verified replay metadata.
        completed, remaining = self.completed_timings()
        write_json(self.run_dir / "cv_completion_state.json", {"status": "OPERATIONALLY_INCOMPLETE" if any(r[0] != 0 for r in remaining) else "ALL_CV_FITS_COMPLETE",
                   "verified_CV_models": sum(r["fold"] != 0 for r in completed), "required_CV_models": 45,
                   "remaining_CV_models": sum(r[0] != 0 for r in remaining)})
        if self.timing_samples is not None:
            self.forecast(self.timing_samples, self.target_seconds, reused=True)
        return path

    def replay(self, path, baseline, held, identity):
        seal = verify_model(path, identity)
        values, receipt = self.replay_payload(path, baseline, held)
        return values, {**receipt, "model_seal_sha256": sha256(path / "seal.json")}

    def replay_payload(self, path, baseline, held):
        saved_keys = read_csv(path / "held_keys.csv")
        if not saved_keys.equals(held[data.KEYS].reset_index(drop=True)):
            raise StopRun("FAILED_VERIFICATION", "Held-out painting/language identities differ")
        payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
        model = CaptionStudent(payload["languages"], self.config).to(self.device)
        model.load_state_dict(payload["state_dict"], strict=True)
        if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
            raise StopRun("FAILED_VERIFICATION", "Nonfinite checkpoint")
        q = predict_student(model, held[data.INPUT_COLUMNS], self.visual, baseline, self.device, self.config["batch_size"])
        raw = np.load(path / "raw_predictions.npy", allow_pickle=False)
        if not np.array_equal(q, raw):
            raise StopRun("FAILED_VERIFICATION", "Independent checkpoint replay differs; prior successful decision is invalid")
        return q, {"checkpoint_sha256": sha256(path / "checkpoint.pt"),
                   "max_abs_error": 0., "rows": len(q), "status": "PASS"}

    def fit_cv(self):
        for fold in range(1, 6):
            _, held, _, _ = self.partition(fold)
            prepared, _ = self.targets(fold)
            baseline = held[data.KEYS].merge(self.r0[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
            for condition in self.config["conditions"]:
                for seed in self.config["seeds"]:
                    self.fit_one(fold, condition, seed, prepared, baseline, held)
        return {"completed": 45}

    def cv_placeholder(self):
        completed, _ = self.completed_timings()
        if sum(r["fold"] != 0 for r in completed) != 45:
            raise StopRun("INCOMPLETE_CV", "Fewer than 45 verified fits: operationally incomplete, not scientific NO-GO")
        return {"status": "FITS_COMPLETE_PENDING_INDEPENDENT_REPLAY"}

    def verify_cv(self):
        self.cv_placeholder()
        expected = 10 * self.target_seconds + sum(r["replay_seconds"] for r in self.timing_samples) * 15
        self.budget.check(expected * self.operations["forecast_safety_factor"])
        write_json(self.run_dir / "cv_verification.json", {"status": "RUNNING", "previous_pass_revoked": True})
        records = {key: [] for key in ("R0", "C0", "C1", "C2", "EC0", "EC1", "EC2")}
        checks, seed_metrics = [], []
        for fold in range(1, 6):
            _, held, _, _ = self.partition(fold)
            self.targets(fold)
            baseline = held[data.KEYS].merge(self.r0[data.KEYS + ["p_positive"]], on=data.KEYS, validate="one_to_one")
            records["R0"].append(baseline)
            for condition in self.config["conditions"]:
                values = []
                for seed in self.config["seeds"]:
                    self.progress("Verify", fold=fold, condition=condition, seed=seed)
                    path = self.run_dir / "models" / f"fold_{fold}_{condition}_{seed}"
                    q, receipt = self.replay(path, baseline, held, self.model_identity(fold, condition, seed))
                    values.append(q)
                    checks.append({"fold": fold, "condition": condition, "seed": seed, **receipt})
                    from .metrics import score
                    metric = score(held, held[data.KEYS].assign(p_positive=np.round(q, 6)))
                    seed_metrics.append({"fold": fold, "condition": condition, "seed": seed,
                                         "contrast": metric["cultural_contrast"], "macro_log_loss": metric["log_loss"]})
                student, ensemble = aggregate(held, values, baseline)
                records[condition].append(student)
                records["E" + condition].append(ensemble)
        self.predictions = {name: pd.concat(parts, ignore_index=True) for name, parts in records.items()}
        for name, pred in self.predictions.items():
            pred.assign(model=name).to_csv(self.run_dir / f"cv_{name}.csv", index=False, lineterminator="\n")
        pd.DataFrame(seed_metrics).to_csv(self.run_dir / "seed_metrics.csv", index=False, lineterminator="\n")
        self.verified = len(checks)
        result = {"status": "PASS", "models_replayed": self.verified, "checks": checks}
        write_json(self.run_dir / "cv_verification.json", result)
        return result

    def decision(self):
        if self.verified != 45:
            raise StopRun("INCOMPLETE_CV", "Scientific decisions require all 45 independently replayed fits")
        self.budget.check(getattr(self, "scoring_seconds", self.operations["minimum_final_scoring_seconds"]))
        analysis, folds, paintings, languages = analyze(self.train, self.predictions, self.config)
        for name, frame in (("folds", folds), ("paintings", paintings), ("languages", languages)):
            frame.to_csv(self.run_dir / f"cv_{name}.csv", index=False, lineterminator="\n")
        write_json(self.run_dir / "cv_analysis.json", analysis)
        self.diagnostics(paintings, languages)
        result = scientific_gate(analysis, folds, self.config, verified_models=self.verified, inputs_verified=True)
        self.cv_decision = result
        scoped = scoped_cv_decision(result, self.execution_scope)
        write_json(self.run_dir / "scientific_cv_decision.json", result)
        write_json(self.run_dir / "decision.json", scoped)
        return scoped

    def diagnostics(self, paintings, languages):
        # Fixed subgroups are descriptive only; never a selection input.
        from PIL import Image
        unique = self.train[["painting", "image"]].drop_duplicates()
        aspect = {}
        for row in unique.itertuples():
            with Image.open(safe_image_path(self.inputs / "artwork", row.image)) as image:
                w, h = image.size
            aspect[row.painting] = max(w / h, h / w)
        coverage = self.train.groupby("painting").size()
        table = paintings.assign(aspect_ratio=paintings.painting.map(aspect), languages=paintings.painting.map(coverage))
        table["aspect_group"] = pd.cut(table.aspect_ratio, [0, 1.25, 1.75, np.inf], labels=["up_to_1.25", "1.25_to_1.75", "above_1.75"])
        table["coverage_group"] = pd.cut(table.languages, [0, 2, 9, np.inf], labels=["below_3", "3_to_9", "10_or_more"])
        for column in ("aspect_group", "coverage_group"):
            table.groupby(["model", column], observed=False).contrast.agg(["count", "mean"]).to_csv(self.run_dir / f"subgroup_{column}.csv")
        pivot = languages.pivot(index="language", columns="model", values="log_loss")
        for right in ("R0", "EC0", "EC1"):
            pivot["EC2_minus_" + right] = pivot.EC2 - pivot[right]
        pivot.to_csv(self.run_dir / "language_loss_changes.csv")
        leave_out = {}
        for fold in range(1, 6):
            names = set(self.train.loc[self.train.fold.ne(fold), "painting"])
            leave_out[str(fold)] = paintings[paintings.painting.isin(names)].groupby("model").contrast.mean().to_dict()
        write_json(self.run_dir / "leave_one_fold_out_aggregation.json", leave_out)

    def final_baseline(self, held):
        self.require_competition_scope()
        languages = sorted(self.train.language.unique())
        predictions = []
        for name, bank, width in zip(("clip", "siglip", "large"), self.banks, (512, 768, 768)):
            directory = self.run_dir / "R0_1335" / name
            identity = {**self.identity, "kind": "R0_1335", "encoder": name, "fit_paintings": sorted(self.train.painting.unique())}
            def writer(stage, bank=bank):
                self.budget.check(self.operations["baseline_component_seconds"])
                self.progress("R0_1335", encoder=name)
                model = fit_frozen_head(self.train, bank, languages, 2027)
                np.savez(stage / "head.npz", coef=model.coef_, intercept=model.intercept_)
                write_json(stage / "languages.json", languages)
            publish_model(directory, identity, writer)
            verify_model(directory, identity)
            with np.load(directory / "head.npz", allow_pickle=False) as a:
                model = head_from_arrays(a["coef"], a["intercept"], width, languages)
            predictions.append(predict_lr(model, held, bank, languages))
        from .fusion import blend
        return blend(blend(predictions[0], predictions[1], .5), predictions[2], .5)

    def confirmation(self):
        self.require_competition_scope()
        if self.cv_decision["status"] != "CV_GO" or self.verified != 45:
            raise StopRun("FAILED_VERIFICATION", "Verified complete CV_GO is required before final fits or confirmation")
        if not self.authorization["AUTHORIZE_GUARDED_CONFIRMATION"]:
            raise StopRun("BLOCKED_CONFIRMATION", "Confirmation was not included in advance authorization")
        # Use training-only smoke inputs before any confirmation label is opened.
        smoke = self.train.groupby("painting", sort=False).head(1).iloc[:64]
        baseline = self.final_baseline(smoke)
        prepared, _ = self.targets(0)
        frozen = {}
        for condition in self.config["conditions"]:
            for seed in self.config["seeds"]:
                path = self.fit_one(0, condition, seed, prepared, baseline, smoke)
                _, receipt = self.replay(path, baseline, smoke, self.model_identity(0, condition, seed))
                frozen[f"{condition}_{seed}"] = receipt["checkpoint_sha256"]
        self.final_verified = len(frozen)
        write_json(self.run_dir / "final_model_identities.json", frozen)
        ledger = self.run_dir / "confirmation_access.json"
        if ledger.exists():
            saved = json.loads(ledger.read_text())
            if saved.get("final_model_identities") != frozen or saved.get("completed") is not True:
                raise StopRun("FAILED_VERIFICATION", "Confirmation already opened without a complete same-identity decision; no second access")
            self.confirm_decision = json.loads((self.run_dir / "confirmation_decision.json").read_text())
            return self.confirm_decision
        write_json(ledger, {"final_model_identities": frozen, "completed": False, "opened_once": True})
        gold = read_confirmation(self.inputs, self.protocol, cv_verified=self.cv_decision["status"] == "CV_GO" and self.verified == 45,
                                 authorized=True, final_models_verified=self.final_verified == 9, execution_scope=self.execution_scope)
        r0 = self.final_baseline(gold[data.INPUT_COLUMNS])
        predictions = {"R0": r0}
        for condition in self.config["conditions"]:
            values = self.final_predict(condition, gold[data.INPUT_COLUMNS], r0)
            predictions[condition], predictions["E" + condition] = aggregate(gold, values, r0)
        analysis, folds, paintings, languages = analyze(gold, predictions, self.config, has_folds=False)
        self.confirm_decision = scientific_gate(analysis, folds, self.config, verified_models=self.final_verified, inputs_verified=True, confirmation=True)
        write_json(self.run_dir / "confirmation_analysis.json", analysis)
        write_json(self.run_dir / "confirmation_decision.json", self.confirm_decision)
        for name, pred in predictions.items():
            pred.to_csv(self.run_dir / f"confirmation_{name}.csv", index=False, lineterminator="\n")
        paintings.to_csv(self.run_dir / "confirmation_paintings.csv", index=False, lineterminator="\n")
        languages.to_csv(self.run_dir / "confirmation_languages.csv", index=False, lineterminator="\n")
        write_json(ledger, {"final_model_identities": frozen, "completed": True, "opened_once": True})
        return self.confirm_decision

    def final_predict(self, condition, inputs, baseline):
        self.require_competition_scope()
        values = []
        for seed in self.config["seeds"]:
            path = self.run_dir / "models" / f"fold_0_{condition}_{seed}"
            verify_model(path, self.model_identity(0, condition, seed))
            payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            model = CaptionStudent(payload["languages"], self.config).to(self.device)
            model.load_state_dict(payload["state_dict"], strict=True)
            values.append(predict_student(model, inputs[data.INPUT_COLUMNS], self.visual, baseline, self.device, self.config["batch_size"]))
        return values

    def final_model(self):
        self.require_competition_scope()
        expected = json.loads((self.run_dir / "final_model_identities.json").read_text())
        for condition in self.config["conditions"]:
            for seed in self.config["seeds"]:
                path = self.run_dir / "models" / f"fold_0_{condition}_{seed}"
                seal = verify_model(path, self.model_identity(0, condition, seed))
                if seal["files"]["checkpoint.pt"] != expected[f"{condition}_{seed}"]:
                    raise StopRun("FAILED_VERIFICATION", "A final model changed after confirmation")
        return {"status": "PASS", "models": expected, "refitted_after_confirmation": False}

    def inference(self):
        self.require_competition_scope()
        if not self.authorization["REQUIRE_SUBMISSION_ZIP"]:
            raise StopRun("CONFIRMATION_GO_NO_SUBMISSION_REQUESTED", "Science complete; no submission requested")
        r0 = self.final_baseline(self.eval_inputs)
        _, prediction = aggregate(self.eval_inputs, self.final_predict("C2", self.eval_inputs, r0), r0)
        prediction.to_csv(self.run_dir / "predictions.csv", index=False, float_format="%.6f", lineterminator="\n")
        return {"rows": len(prediction), "input": "image_pixels_and_target_language_via_checksum_verified_frozen_features"}

    def validate_submission(self):
        self.require_competition_scope()
        submission.pack(self.run_dir / "predictions.csv", self.run_dir / "submission.zip", self.eval_inputs)
        actual = submission.validate_archive(self.run_dir / "submission.zip", self.eval_inputs)
        if not actual[data.KEYS].equals(self.eval_inputs[data.KEYS]):
            raise StopRun("FAILED_SUBMISSION", "Submission ordering differs")
        digest = sha256(self.run_dir / "submission.zip")
        (self.run_dir / "submission.sha256").write_text(digest + "  submission.zip\n")
        return {"status": "PASS", "rows": len(actual), "sha256": digest, "automatic_promotion": False}

    def require_competition_scope(self):
        if getattr(self, "execution_scope", None) != COMPETITION_READY or not getattr(self, "competition_eligible", False):
            raise StopRun("BLOCKED_SCOPE", "Research-only or eligibility-unverified execution cannot enter competition stages")

    def run(self):
        callbacks = dict(zip(("Validate", "Benchmark", "Experiment", "CV", "Verify", "Decision", "GuardedConfirmation", "FinalModel", "Inference", "SubmissionValidation"),
                             (self.validate, self.benchmark, self.fit_cv, self.cv_placeholder, self.verify_cv, self.decision,
                              self.confirmation, self.final_model, self.inference, self.validate_submission)))
        failures = {"Validate": "BLOCKED_INPUT", "Benchmark": "FAILED_TRAINING", "Experiment": "FAILED_TRAINING",
                    "CV": "FAILED_VERIFICATION", "Verify": "FAILED_VERIFICATION", "Decision": "FAILED_VERIFICATION",
                    "GuardedConfirmation": "FAILED_VERIFICATION", "FinalModel": "FAILED_VERIFICATION",
                    "Inference": "FAILED_SUBMISSION", "SubmissionValidation": "FAILED_SUBMISSION"}
        def guarded(stage, callback):
            def invoke():
                try:
                    return callback()
                except StopRun:
                    raise
                except Exception as exc:
                    raise StopRun(failures[stage], f"{stage}: {type(exc).__name__}: {exc}") from exc
            return invoke
        return execute_stages({stage: guarded(stage, callback) for stage, callback in callbacks.items()}, self.budget, self.progress,
                              scope=self.execution_scope)
