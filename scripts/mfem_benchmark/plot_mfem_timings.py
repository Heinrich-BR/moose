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

# Registered section name (regex)  ->  short label used in the legend.
# The prefix "MFEM::" comes from PerfGraphInterface(perf_graph, "MFEM") /
# PerfGraphInterface(&problem, "MFEM"). Save lives in MFEMDataCollection-derived
# MooseObjects, whose prefix is the registered type name.
SECTIONS = [
    (re.compile(r"^MFEM::EquationSystemProblemOperator::Solve::Mult$"),
     "Mult (steady)"),
    (re.compile(r"^MFEM::TimeDependentEquationSystemProblemOperator::ImplicitSolve::Mult$"),
     "Mult (transient)"),
    (re.compile(r"^MFEM::EquationSystem::FormSystemOperator::FormLinearSystem$"),
     "FormLinearSystem (op)"),
    (re.compile(r"^MFEM::EquationSystem::FormSystemMatrix::FormLinearSystem$"),
     "FormLinearSystem (diag)"),
    (re.compile(r"^MFEM::EquationSystem::FormSystemMatrix::FormRectangularLinearSystem$"),
     "FormRectangularLinearSystem"),
    (re.compile(r"^MFEM::ComplexEquationSystem::FormSystemOperator::FormLinearSystem$"),
     "FormLinearSystem (cmplx op)"),
    (re.compile(r"^MFEM::ComplexEquationSystem::FormSystemMatrix::FormLinearSystem$"),
     "FormLinearSystem (cmplx diag)"),
    (re.compile(r"^MFEM\w*DataCollection::Save$"),
     "Save"),
]

DOFS_RE = re.compile(r"\[MFEM_DOFS\]\s+total_true_dofs=(\d+)")
LOG_NAME_RE = re.compile(
    r"^(?P<tag>[A-Za-z0-9]+)_(?P<device>[A-Za-z0-9-]+)_r(?P<r>\d+)\.log$"
)

# Per-device line style. Marker varies too so devices stay distinguishable
# when printed in greyscale.
DEVICE_STYLES = {
    "cpu":      {"linestyle": "-",  "marker": "o"},
    "ceed-cpu": {"linestyle": "-.", "marker": "D"},
    "hip":      {"linestyle": "--", "marker": "s"},
    "ceed-hip": {"linestyle": ":",  "marker": "^"},
}
DEFAULT_STYLE = {"linestyle": (0, (3, 1, 1, 1, 1, 1)), "marker": "x"}


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
        label = None
        for pattern, lbl in SECTIONS:
            if pattern.match(section_name):
                label = lbl
                break
        if label is None:
            continue
        try:
            total_s = float(parts[7])
        except ValueError:
            continue
        # Tree table prints each section once; first hit wins.
        timings.setdefault(label, total_s)
    return dofs, timings


def collect(results_dir: Path):
    """Group runs by (example tag, device).
    Returns {tag: {device: [(refinement, dofs, timings), ...]}}.
    """
    runs: dict[str, dict[str, list[tuple[int, int, dict[str, float]]]]] = {}
    for log in sorted(results_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        tag = m.group("tag")
        device = m.group("device")
        refinement = int(m.group("r"))
        dofs, timings = parse_log(log)
        if dofs is None:
            print(f"  warning: {log.name} has no [MFEM_DOFS] line — skipped")
            continue
        if not timings:
            print(f"  warning: {log.name} has no MFEM timer rows — skipped")
            continue
        runs.setdefault(tag, {}).setdefault(device, []).append(
            (refinement, dofs, timings)
        )
    for tag in runs:
        for device in runs[tag]:
            runs[tag][device].sort()
    return runs


def plot_example(tag: str, by_device, out_path: Path):
    """`by_device` is {device: [(refinement, dofs, timings), ...]}."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # One color per section label, shared across devices, taken from matplotlib's
    # default property cycle. Only sections that have data get a color so the
    # legend stays compact.
    used_labels: list[str] = []
    for _, label in SECTIONS:
        for runs in by_device.values():
            if any(label in t for _r, _d, t in runs):
                used_labels.append(label)
                break
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    section_color = {
        label: color_cycle[i % len(color_cycle)]
        for i, label in enumerate(used_labels)
    }

    plotted_any = False
    # Stable device order: known devices first, then any extras alphabetically.
    devices_present = sorted(
        by_device.keys(),
        key=lambda d: (list(DEVICE_STYLES).index(d) if d in DEVICE_STYLES else 99, d),
    )

    for device in devices_present:
        style = DEVICE_STYLES.get(device, DEFAULT_STYLE)
        runs = by_device[device]
        for label in used_labels:
            xs, ys = [], []
            for _r, dofs, timings in runs:
                t = timings.get(label)
                if t is None or t <= 0:
                    continue
                xs.append(dofs)
                ys.append(t)
            if not xs:
                continue
            ax.loglog(
                xs,
                ys,
                color=section_color[label],
                linestyle=style["linestyle"],
                marker=style["marker"],
            )
            plotted_any = True

    if not plotted_any:
        print(f"  {tag}: no usable timing data — skipping plot")
        plt.close(fig)
        return

    # Two-part legend: section -> color, device -> linestyle.
    from matplotlib.lines import Line2D

    section_handles = [
        Line2D([0], [0], color=section_color[label], linestyle="-", marker="o", label=label)
        for label in used_labels
    ]
    device_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DEVICE_STYLES.get(d, DEFAULT_STYLE)["linestyle"],
            marker=DEVICE_STYLES.get(d, DEFAULT_STYLE)["marker"],
            label=d,
        )
        for d in devices_present
    ]
    section_legend = ax.legend(
        handles=section_handles, title="Section", loc="upper left", fontsize=9
    )
    ax.add_artist(section_legend)
    ax.legend(handles=device_handles, title="Device", loc="lower right", fontsize=9)

    ax.set_xlabel("Number of true DoFs")
    ax.set_ylabel("Cumulative time [s]")
    ax.set_title(f"{tag}: MFEM section timing vs. DoFs")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
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

    for tag, by_device in runs.items():
        plot_example(tag, by_device, results_dir / f"{tag}_timings.png")


if __name__ == "__main__":
    main()
