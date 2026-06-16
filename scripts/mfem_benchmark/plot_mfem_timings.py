#!/usr/bin/env python3
"""
Parse MOOSE-MFEM benchmark logs produced by run_benchmark.sh and generate
log-log plots of timing vs. number of true DoFs for each example.

Each log must contain:
  - a [MFEM_DOFS] total_true_dofs=N line (added in EquationSystem::Init / ComplexEquationSystem::Init)
  - a Performance Graph table from Outputs/perf_graph=true

For each example, several PNGs are written into the results directory. Timed
sections are bucketed into three logical groups, "BuildBilinearForms",
"FormLinearSystem" and "Mult", and for each group (plus their sum) both a
raw-time plot and a per-DoF plot are produced. Within a plot, color denotes the
polynomial order, linestyle the device family (cpu/hip) and marker the assembly
level.

Runs are folded by device family, so the libCEED ceed-cpu/ceed-hip backends
(which is where matrix-free "none" assembly runs) are drawn under cpu/hip. The
assembled-matrix levels are collapsed into a single "legacy/full" series that
uses the legacy results on CPU and the full results on GPU.

An optional second argument "<cpu_ranks>,<hip_ranks>" restricts the plots to a
single MPI-rank count per hardware family, so a CPU run at one rank count can be
compared against a GPU run at another. For example "32,4" keeps only cpu/ceed-cpu
logs with np=32 and hip/ceed-hip logs with np=4. The chosen counts are appended
to the device legend labels and the output filenames.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Timed PerfGraph sections are bucketed into logical groups. The prefix "MFEM::"
# comes from PerfGraphInterface(perf_graph, "MFEM") / PerfGraphInterface(&problem,
# "MFEM"). Within a single run at most one pattern per group fires (the op/diag and
# steady/transient code paths are mutually exclusive and never co-occur), so a
# run's group time is just the sum of whatever matched.
SECTION_GROUPS = {
    # Local (bi/sesqui)linear form assembly: steady, transient and complex paths
    # each have their own BuildBilinearForms override; only one fires per run.
    "BuildBilinearForms": [
        re.compile(r"^MFEM::EquationSystem::BuildBilinearForms$"),
        re.compile(r"^MFEM::TimeDependentEquationSystem::BuildBilinearForms$"),
        re.compile(r"^MFEM::ComplexEquationSystem::BuildBilinearForms$"),
    ],
    # FormSystemOperator (non-legacy) and FormSystemMatrix (legacy diagonal block)
    # are different paths representing the same work; real and complex are folded
    # together too since each example is either real or complex, never both.
    "FormLinearSystem": [
        re.compile(r"^MFEM::EquationSystem::FormSystemOperator::FormLinearSystem$"),
        re.compile(r"^MFEM::EquationSystem::FormSystemMatrix::FormLinearSystem$"),
        re.compile(r"^MFEM::ComplexEquationSystem::FormSystemOperator::FormLinearSystem$"),
        re.compile(r"^MFEM::ComplexEquationSystem::FormSystemMatrix::FormLinearSystem$"),
    ],
    # Steady (Solve), transient (ImplicitSolve) and complex (Solve) residual
    # evaluations folded into one "Mult" group.
    "Mult": [
        re.compile(r"^MFEM::EquationSystemProblemOperator::Solve::Mult$"),
        re.compile(r"^MFEM::TimeDependentEquationSystemProblemOperator::ImplicitSolve::Mult$"),
        re.compile(r"^MFEM::ComplexEquationSystemProblemOperator::Solve::Mult$"),
    ],
}

# One output plot per entry: (title label, group(s) summed for the y value,
# filename slug). For multi-group kinds a run contributes a point only if every
# listed group fired in that run.
PLOT_KINDS = [
    ("BuildBilinearForms", ["BuildBilinearForms"], "buildbilinearforms"),
    ("FormLinearSystem", ["FormLinearSystem"], "formlinearsystem"),
    ("Mult", ["Mult"], "mult"),
    (
        "BuildBilinearForms + FormLinearSystem + Mult",
        ["BuildBilinearForms", "FormLinearSystem", "Mult"],
        "total",
    ),
]

DOFS_RE = re.compile(r"\[MFEM_DOFS\]\s+total_true_dofs=(\d+)")
# The _p<P> order and _n<R> MPI-rank components are optional for backward
# compatibility with logs produced before those sweeps were added; such logs are
# treated as order 1. Rank count is parsed but not currently a plotting
# dimension, so mixing rank counts in one results dir overlays them.
LOG_NAME_RE = re.compile(
    r"^(?P<tag>[A-Za-z0-9]+)_(?P<device>[A-Za-z0-9-]+)_(?P<asm>[A-Za-z]+)"
    r"(?:_p(?P<p>\d+))?(?:_n(?P<n>\d+))?_r(?P<r>\d+)\.log$"
)

# Polynomial order -> color (primary categorical dimension, Tableau palette).
ORDER_COLORS = {
    1: "tab:red",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:blue",
}
# Device -> linestyle (second categorical dimension).
DEVICE_LINESTYLES = {
    "cpu":      "-",
    "hip":      "--",
}
DEFAULT_COLOR = "black"
DEFAULT_LINESTYLE = (0, (3, 1, 1, 1, 1, 1))


def device_family(device: str) -> str:
    """Map a device name to its hardware family, 'cpu' or 'hip'."""
    return "hip" if device in ("hip", "ceed-hip") else "cpu"


def merged_assembly(family: str, asm: str):
    """Collapse the assembled-matrix levels into one 'legacy/full' series.

    The fastest assembled-matrix path differs by hardware — legacy on CPU, full on
    GPU — so they are plotted as a single series picking the device-appropriate
    one. Returns the assembly label to plot under, or None to drop the run (full on
    CPU, legacy on GPU — the device-inappropriate variant). partial and none pass
    through unchanged.
    """
    if asm == "legacy":
        return "legacy/full" if family == "cpu" else None
    if asm == "full":
        return "legacy/full" if family == "hip" else None
    return asm

# Per-assembly-level marker (third categorical dimension). legacy and full are
# collapsed into one "legacy/full" series by merged_assembly().
ASSEMBLY_MARKERS = {
    "legacy/full": "o",
    "partial":     "^",
    "none":        "D",
}
DEFAULT_MARKER = "x"


def parse_log(path: Path):
    """Return (dofs, {group_name: total_seconds}) parsed from one log file."""
    text = path.read_text(errors="replace")

    dofs_match = DOFS_RE.search(text)
    dofs = int(dofs_match.group(1)) if dofs_match else None

    group_times: dict[str, float] = {}
    seen: set[str] = set()
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
        if section_name in seen:  # tree table prints each section once; first wins
            continue
        group = None
        for gname, patterns in SECTION_GROUPS.items():
            if any(p.match(section_name) for p in patterns):
                group = gname
                break
        if group is None:
            continue
        try:
            total_s = float(parts[7])
        except ValueError:
            continue
        seen.add(section_name)
        group_times[group] = group_times.get(group, 0.0) + total_s
    return dofs, group_times


def collect(results_dir: Path, rank_filter: dict[str, int] | None = None):
    """Group runs by (example tag, (device_family, assembly, order)).
    Returns {tag: {(family, assembly, order): [(refinement, dofs, group_times), ...]}}.

    Devices are folded to their family (cpu/hip), so ceed-cpu/ceed-hip share a
    series with cpu/hip; legacy/full are merged per merged_assembly(). If
    `rank_filter` is given (a {family: rank_count} mapping), only logs whose
    MPI-rank count matches the requirement for their device family are kept.
    """
    runs: dict[
        str, dict[tuple[str, str, int], list[tuple[int, int, dict[str, float]]]]
    ] = {}
    for log in sorted(results_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        tag = m.group("tag")
        family = device_family(m.group("device"))
        ranks = int(m.group("n")) if m.group("n") else None
        if rank_filter is not None and ranks != rank_filter[family]:
            continue
        assembly = merged_assembly(family, m.group("asm"))
        if assembly is None:
            continue
        order = int(m.group("p")) if m.group("p") else 1
        key = (family, assembly, order)
        refinement = int(m.group("r"))
        dofs, group_times = parse_log(log)
        if dofs is None:
            print(f"  warning: {log.name} has no [MFEM_DOFS] line — skipped")
            continue
        if not group_times:
            print(f"  warning: {log.name} has no MFEM timer rows — skipped")
            continue
        runs.setdefault(tag, {}).setdefault(key, []).append(
            (refinement, dofs, group_times)
        )
    for tag in runs:
        for key in runs[tag]:
            # Sort by (refinement, dofs) only; the trailing group_times dict is not
            # orderable, and several runs (e.g. different rank counts) can share the
            # same refinement under one folded key.
            runs[tag][key].sort(key=lambda entry: (entry[0], entry[1]))
    return runs


def kind_value(group_times: dict[str, float], components: list[str]):
    """Sum the named groups for one run, or None if any is missing/non-positive."""
    vals = [group_times.get(g) for g in components]
    if any(v is None or v <= 0 for v in vals):
        return None
    return sum(vals)


def plot_kind(tag: str, by_run, kind_label: str, components, out_path: Path,
              normalize: bool = False, rank_filter: dict[str, int] | None = None):
    """Plot one section-kind for one example.

    `by_run` is {(device, assembly, order): [(refinement, dofs, group_times), ...]}.
    `components` lists which section groups are summed for the y value. If
    `normalize` is True, the y-axis is divided by the global number of true DoFs.
    If `rank_filter` is given, the per-family rank counts are shown in the device
    legend labels.
    """
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Stable order: known orders first, then devices, then assembly levels.
    keys_present = sorted(
        by_run.keys(),
        key=lambda k: (
            list(ORDER_COLORS).index(k[2]) if k[2] in ORDER_COLORS else 99,
            list(DEVICE_LINESTYLES).index(k[0]) if k[0] in DEVICE_LINESTYLES else 99,
            list(ASSEMBLY_MARKERS).index(k[1]) if k[1] in ASSEMBLY_MARKERS else 99,
            k,
        ),
    )

    plotted_keys = []
    for device, asm, order in keys_present:
        color = ORDER_COLORS.get(order, DEFAULT_COLOR)
        linestyle = DEVICE_LINESTYLES.get(device, DEFAULT_LINESTYLE)
        marker = ASSEMBLY_MARKERS.get(asm, DEFAULT_MARKER)
        xs, ys = [], []
        for _r, dofs, group_times in by_run[(device, asm, order)]:
            v = kind_value(group_times, components)
            if v is None:
                continue
            xs.append(dofs)
            ys.append(v / dofs if normalize else v)
        # A lone point draws no line, so the device (linestyle) is indistinguishable;
        # require at least two points before plotting a series.
        if len(xs) < 2:
            continue
        ax.loglog(xs, ys, color=color, linestyle=linestyle, marker=marker)
        plotted_keys.append((device, asm, order))

    if not plotted_keys:
        print(f"  {tag} [{kind_label}]: no series with >=2 points — skipping plot")
        plt.close(fig)
        return

    # Three-part legend: order -> color, device -> linestyle, assembly -> marker.
    from matplotlib.lines import Line2D

    orders_present = [p for p in ORDER_COLORS if any(k[2] == p for k in plotted_keys)]
    orders_present += sorted({k[2] for k in plotted_keys} - set(orders_present))
    devices_present = [d for d in DEVICE_LINESTYLES if any(k[0] == d for k in plotted_keys)]
    devices_present += sorted({k[0] for k in plotted_keys} - set(devices_present))
    assemblies_present = [a for a in ASSEMBLY_MARKERS if any(k[1] == a for k in plotted_keys)]
    assemblies_present += sorted({k[1] for k in plotted_keys} - set(assemblies_present))

    order_handles = [
        Line2D(
            [0],
            [0],
            color=ORDER_COLORS.get(p, DEFAULT_COLOR),
            linestyle="-",
            label=f"p={p}",
        )
        for p in orders_present
    ]
    device_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=DEVICE_LINESTYLES.get(d, DEFAULT_LINESTYLE),
            label=(f"{d} (np={rank_filter[device_family(d)]})" if rank_filter else d),
        )
        for d in devices_present
    ]
    assembly_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="none",
            marker=ASSEMBLY_MARKERS.get(a, DEFAULT_MARKER),
            label=a,
        )
        for a in assemblies_present
    ]
    order_legend = ax.legend(
        handles=order_handles, title="Order", loc="upper left", fontsize=9
    )
    ax.add_artist(order_legend)
    device_legend = ax.legend(
        handles=device_handles, title="Device", loc="lower left", fontsize=9
    )
    ax.add_artist(device_legend)
    ax.legend(handles=assembly_handles, title="Assembly", loc="lower right", fontsize=9)

    ax.set_xlabel("Number of true DoFs")
    if normalize:
        ax.set_ylabel(f"Cumulative time per DoF [s/DoF]")
        ax.set_title(f"{tag}: {kind_label} time per DoF")
    else:
        ax.set_ylabel(f"Cumulative time [s]")
        ax.set_title(f"{tag}: {kind_label} time")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} <results_dir> [cpu_ranks,hip_ranks]", file=sys.stderr)
        sys.exit(1)
    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rank_filter = None
    suffix = ""
    if len(sys.argv) == 3:
        try:
            cpu_ranks, hip_ranks = (int(x) for x in sys.argv[2].split(","))
        except ValueError:
            print(
                f"Invalid rank pair '{sys.argv[2]}': expected <cpu_ranks>,<hip_ranks>",
                file=sys.stderr,
            )
            sys.exit(1)
        rank_filter = {"cpu": cpu_ranks, "hip": hip_ranks}
        suffix = f"_cpu{cpu_ranks}_hip{hip_ranks}"

    runs = collect(results_dir, rank_filter)
    if not runs:
        print("No usable logs found.", file=sys.stderr)
        sys.exit(1)

    for tag, by_run in runs.items():
        for kind_label, components, slug in PLOT_KINDS:
            plot_kind(
                tag, by_run, kind_label, components,
                results_dir / f"{tag}_{slug}_timings{suffix}.png",
                rank_filter=rank_filter,
            )
            plot_kind(
                tag, by_run, kind_label, components,
                results_dir / f"{tag}_{slug}_timings_per_dof{suffix}.png",
                normalize=True, rank_filter=rank_filter,
            )


if __name__ == "__main__":
    main()
