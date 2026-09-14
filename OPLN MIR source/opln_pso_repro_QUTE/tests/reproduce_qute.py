"""Run the fixed QUTE regression protocol without changing the program framework.

Example: python -m tests.reproduce_qute --device cuda --min-domain-um 0
Each stage is an ordinary invocation of the existing run_pso.py entry point.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--min-domain-um", type=float, default=0.0)
    parser.add_argument("--stop-after", type=int, choices=[500, 1000, 1500, 1700],
                        default=1700, help="Total PSO iterations in the saved benchmark protocol")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.output or project / "output" / f"qute_protocol_min_{args.min_domain_um:g}um"
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Choose an empty output directory to preserve existing results: {root}")
    root.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(project / "run_pso.py"),
              "--pump-um", "1.75", "--signal-um", "3.5",
              "--crystal-length-mm", "30", "--target-sigma-mm", "6",
              "--qpm-period-um", "15.497", "--length-rounding", "floor",
              "--min-domain-um", str(args.min_domain_um), "--device", args.device,
              "--objective", "legacy_intensity", "--sampling", "wavelength",
              "--q-points", "601", "--optimization-span-nm", "60",
              "--jsa-span-nm", "60", "--jsa-grid", "200", "--jsa-tick-nm", "20",
              "--pump-scan-min-nm", "0.5", "--pump-scan-max-nm", "8",
              "--pump-coarse-step-nm", "0.1", "--pump-fine-step-nm", "0.01",
              "--particles", "24", "--iterations", "100", "--cycles", "5",
              "--noise-decay", "0.6", "--seed", "7", "--no-symmetric",
              "--polish-iterations", "0"]
    env = os.environ.copy()
    # Small 200x200 SVDs do not benefit from dozens of BLAS threads. This
    # limits only the child process; no user's system environment is changed.
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    previous = None
    previous_iterations = 0
    schedule = [(500, 0.06, 5), (1000, 0.005, 5), (1500, 0.002, 5), (1700, 0.001, 2)]
    schedule = [stage for stage in schedule if stage[0] <= args.stop_after]
    for total, noise, cycles in schedule:
        destination = root / f"stage_{total:04d}"
        command = common + ["--initial-noise", str(noise), "--cycles", str(cycles),
                            "--output", str(destination)]
        if total != schedule[-1][0]:
            command += ["--skip-jsa"]
        if previous is not None:
            command += ["--initial-duties-npy", str(previous / "optimized_duties.npy"),
                        "--prior-history-npy", str(previous / "pso_history.npy"),
                        "--prior-pso-iterations", str(previous_iterations)]
        subprocess.run(command, cwd=project, env=env, check=True)
        previous = destination
        previous_iterations = total
    with (previous / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    actual = summary["structures"]["optimized"]["purity"]
    target = 0.997571
    print(f"Py_JSI |PMF| purity, 60 nm window: {actual:.8f}; table reference: {target:.6f}")
    print("Magnitude-only benchmark met" if actual >= target else "Magnitude-only benchmark NOT met")
    print("This single-seed result does not establish full-phase or wider-window paper reproduction.")


if __name__ == "__main__":
    main()
