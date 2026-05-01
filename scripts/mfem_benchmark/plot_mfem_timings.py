#!/usr/bin/env python3
"""
Parse MOOSE-MFEM benchmark logs produced by run_benchmark.sh and generate
log-log plots of timing vs. number of true DoFs for each example.

Each log must contain:
  - a [MFEM_DOFS] total_true_dofs=N line (added in EquationSystem::Init / ComplexEquationSystem::Init)
  - a Performance Graph table from Outputs/perf_graph=true

For each example, one PNG is written into the results directory with one
loglog curve per timed section that fired during that run.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Section name printed by TIME_SECTION  ->  short label used in the legend
SECTIONS = {
    "MFEM::EquationSystemProblemOperator::Solve::Mult": "Mult (steady)",
    "MFEM::TimeDependentEquationSystemProblemOperator::ImplicitSolve::Mult": "Mult (transient)",
    "MFEM::EquationSystem::FormSystemOperator::FormLinearSystem": "FormLinearSystem (op)",
    "MFEM::EquationSystem::FormSystemMatrix::FormLinearSystem": "FormLinearSystem (diag)",
    "MFEM::EquationSystem::FormSystemMatrix::FormRectangularLinearSystem": "FormRectangularLinearSystem",
    "MFEM::ComplexEquationSystem::FormSystemOperator::FormLinearSystem": "FormLinearSystem (cmplx op)",
    "MFEM::ComplexEquationSystem::FormSystemMatrix::FormLinearSystem": "FormLinearSystem (cmplx diag)",
    "MFEM::MFEMDataCollection::output::Save": "Save",
}

DOFS_RE = re.compile(r"\[MFEM_DOFS\]\s+total_true_dofs=(\d+)")
LOG_NAME_RE = re.compile(r"^(?P<tag>[A-Za-z0-9]+)_r(?P<r>\d+)\.log$")


def parse_log(path: Path):
    """Return (dofs, {section_name: total_seconds}) parsed from one log file."""
    text = path.read_text(errors="replace")

    dofs_match = DOFS_RE.search(text)
    dofs = int(dofs_match.group(1)) if dofs_match else None

    timings: dict[str, float] = {}
    # The full PerfGraph tree table has 10 columns:
    #   Section | Calls | Self(s) | Avg(s) | % | Mem | Total(s) | Avg(s) | % | Mem
    # After splitting on '|', total time is at index 7 (index 0 is empty before
    # the leading '|'). The "Heaviest Sections" table has fewer columns so we
    # ignore it via the len(parts) check.
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 12:  # tree table only
            continue
        section_name = parts[1]
        if section_name not in SECTIONS:
            continue
        try:
            total_s = float(parts[7])
        except ValueError:
            continue
        # Tree table prints each section once; first hit wins.
        timings.setdefault(section_name, total_s)
    return dofs, timings


def collect(results_dir: Path):
    """Group runs by example tag.  -> {tag: [(refinement, dofs, timings), ...]}"""
    runs: dict[str, list[tuple[int, int, dict[str, float]]]] = {}
    for log in sorted(results_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        tag = m.group("tag")
        refinement = int(m.group("r"))
        dofs, timings = parse_log(log)
        if dofs is None:
            print(f"  warning: {log.name} has no [MFEM_DOFS] line — skipped")
            continue
        if not timings:
            print(f"  warning: {log.name} has no MFEM timer rows — skipped")
            continue
        runs.setdefault(tag, []).append((refinement, dofs, timings))
    for tag in runs:
        runs[tag].sort()
    return runs


def plot_example(tag: str, runs, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    plotted_any = False
    for full_name, label in SECTIONS.items():
        xs, ys = [], []
        for _, dofs, timings in runs:
            t = timings.get(full_name)
            if t is None or t <= 0:
                continue
            xs.append(dofs)
            ys.append(t)
        if not xs:
            continue
        ax.loglog(xs, ys, "o-", label=label)
        plotted_any = True

    if not plotted_any:
        print(f"  {tag}: no usable timing data — skipping plot")
        plt.close(fig)
        return

    ax.set_xlabel("Number of true DoFs")
    ax.set_ylabel("Cumulative time [s]")
    ax.set_title(f"{tag}: MFEM section timing vs. DoFs")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <results_dir>", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    runs = collect(results_dir)
    if not runs:
        print("No usable logs found.", file=sys.stderr)
        sys.exit(1)

    for tag, run_list in runs.items():
        plot_example(tag, run_list, results_dir / f"{tag}_timings.png")


if __name__ == "__main__":
    main()
