#!/usr/bin/env python3
"""
Rank the solver + preconditioner combinations in a run_solver_study.sh results
directory by time to solution, for each example, device and assembly level.

Time to solution is the sum of three parts, taken from the level 2 perf graph:
  form    EquationSystemProblemOperator::FormSystem: bilinear form assembly and
          forming the linear system
  setop   EquationSystemProblemOperator::SolveSystem minus the Mult below: setting the
          operator on the solver, where MatrixFreeAMS and p-multigrid set themselves up
  mult    ProblemOperatorBase::SolveWithOperator::Mult: the Krylov solve, including the
          setup of hypre preconditioners, which hypre runs on their first application

Only runs whose outer solver converged to the right solution are ranked. 'dev' is the
relative deviation of the run's solution functional (diffusion: solution_l2_norm;
curlcurl: e_error, the L2 error against the exact solution) from the median over the
example's converged runs. The solvers measure their residual in different norms, and a
solver can report convergence with a preconditioner it cannot handle, so a run whose
deviation exceeds MAX_DEV is reported as a wrong solution.

Usage: summarize_solver_study.py <results_dir> [--tol TOL]
"""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOG_NAME_RE = re.compile(
    r"^(?P<tag>[^_]+)_(?P<device>[^_]+)_(?P<asm>[^_]+)_(?P<combo>[^_]+)"
    r"_p(?P<p>\d+)_n(?P<n>\d+)_r(?P<r>\d+)\.log$"
)
SECTIONS = {
    "form": "EquationSystemProblemOperator::FormSystem",
    "solve": "EquationSystemProblemOperator::SolveSystem",
    "mult": "ProblemOperatorBase::SolveWithOperator::Mult",
}
FUNCTIONALS = {"diffusion": "solution_l2_norm", "curlcurl": "e_error"}
# Converged runs agree to about 1e-11 at small sizes; MFEM GMRES with MatrixFreeAMS,
# which reports convergence to a wrong solution, deviates by a factor of 30.
MAX_DEV = 1e-6


def parse_log(path: Path, tol: float):
    """Return a dict describing one run, with 'error' set if it produced no result."""
    # The logs contain NUL bytes.
    text = path.read_bytes().replace(b"\0", b"").decode(errors="replace")
    run = {"dofs": None, "its": None, "error": None}
    m = re.search(r"total_true_dofs=(\d+)", text)
    if m:
        run["dofs"] = int(m.group(1))
    if "TIMED OUT after" in text:
        run["error"] = "timed out"
        return run
    if "Performance Graph:" not in text:
        run["error"] = "failed"
        return run

    # The tree table prints each section once; cols[7] is its Total(s) column.
    for line in text.splitlines():
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 7:
            for key, section in SECTIONS.items():
                if cols[1] == section and key not in run:
                    run[key] = float(cols[7])
    if any(key not in run for key in SECTIONS):
        run["error"] = "no level 2 perf graph"
        return run
    run["setop"] = run["solve"] - run["mult"]
    run["total"] = run["form"] + run["solve"]

    # MFEM solvers print "PCG: Number of iterations: N" and, unless converged,
    # "PCG: No convergence!". MFEM's hypre wrappers print "PCG Iterations = N" and the
    # final relative residual norm, which hypre compares with the tolerance. MOOSE's
    # progress dots can land inside MFEM's line: "PCG: Number of iterations: .599".
    mfem_its = re.findall(r"(?:PCG|GMRES): Number of iterations: [.\s]*(\d+)", text)
    hypre_its = re.findall(r"(?:PCG|GMRES) Iterations = (\d+)", text)
    hypre_res = re.findall(r"Final (?:PCG|GMRES) Relative Residual Norm = (\S+)", text)
    if hypre_its and hypre_res:
        run["its"] = int(hypre_its[-1])
        converged = float(hypre_res[-1]) <= tol
    elif mfem_its:
        run["its"] = int(mfem_its[-1])
        converged = "No convergence!" not in text
    else:
        converged = False
    if not converged:
        run["error"] = "not converged"
    return run


def read_functional(results_dir: Path, name: str, column: str):
    """Last value of `column` in the run's postprocessor CSV, or None."""
    path = results_dir / "csv" / f"{name}.csv"
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1][column]) if rows and column in rows[-1] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--tol", type=float, default=1e-12,
                        help="relative tolerance the outer solvers were given")
    args = parser.parse_args()

    cases = defaultdict(dict)
    for log in sorted(args.results_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        run = parse_log(log, args.tol)
        run["functional"] = read_functional(args.results_dir, log.stem, FUNCTIONALS.get(m["tag"]))
        run["label"] = f"p{m['p']}, {m['n']} ranks, parallel_refine={m['r']}"
        cases[(m["tag"], m["device"], m["asm"])][m["combo"]] = run
    if not cases:
        raise SystemExit(f"No run_solver_study.sh logs in {args.results_dir}")

    medians = {}
    for tag in {key[0] for key in cases}:
        values = [run["functional"] for key, runs in cases.items() if key[0] == tag
                  for run in runs.values() if not run["error"] and run["functional"] is not None]
        medians[tag] = statistics.median(values) if values else None
    for (tag, _, _), runs in cases.items():
        for run in runs.values():
            run["dev"] = None
            if run["functional"] is not None and medians[tag]:
                run["dev"] = abs(run["functional"] / medians[tag] - 1)
                if not run["error"] and run["dev"] > MAX_DEV:
                    run["error"] = f"wrong solution (dev {run['dev']:.1e})"

    fastest = {}
    for (tag, device, asm), runs in sorted(cases.items()):
        dofs = next((run["dofs"] for run in runs.values() if run["dofs"]), None)
        label = next(iter(runs.values()))["label"]
        dofs_text = f", {dofs:,} true DoFs" if dofs else ""
        print(f"{tag}, {device}, {asm} assembly ({label}{dofs_text})")
        print(f"  {'combination':16s} {'its':>6s} {'form':>8s} {'setop':>8s} {'mult':>8s} "
              f"{'total':>8s} {'x best':>7s} {'dev':>8s}")
        ranked = sorted((run["total"], combo) for combo, run in runs.items() if not run["error"])
        best = ranked[0][0] if ranked else None
        for total, combo in ranked:
            run = runs[combo]
            dev = "-" if run["dev"] is None else f"{run['dev']:.1e}"
            print(f"  {combo:16s} {run['its']:6d} {run['form']:8.2f} {run['setop']:8.2f} "
                  f"{run['mult']:8.2f} {total:8.2f} {total / best:7.2f} {dev:>8s}")
        for combo, run in sorted(runs.items()):
            if run["error"]:
                its = f" after {run['its']} iterations" if run["its"] else ""
                print(f"  {combo:16s} {run['error']}{its}")
        if ranked:
            fastest[(tag, device, asm)] = (ranked[0][1], best)
        print()

    print("Fastest combination per case (times in s)")
    for (tag, device, asm), (combo, total) in sorted(fastest.items()):
        print(f"  {tag:10s} {device:9s} {asm:8s} {combo:16s} {total:8.2f}")


if __name__ == "__main__":
    main()
