#!/usr/bin/env python3
"""
Parse the paired GPU-aware-MPI on/off logs produced by
run_gpu_aware_mpi_benchmark.sh and plot, for each timed section group (and their
sum), the speedup obtained by enabling MFEM's GPU-aware MPI vs. the number of
true DoFs.

For every (example, device, assembly, order, ranks, refinement) there are two
logs: one with GPU-aware MPI on (gam1) and one with it off (gam0). The speedup is

    speedup = time(gam0) / time(gam1)

i.e. how much faster the section runs with GPU-aware MPI enabled (> 1 is faster).
The section grouping, log parsing and the order/device/assembly visual encoding
are reused from plot_mfem_timings.py. Within a plot, color denotes the polynomial
order, linestyle the device family (cpu/hip) and marker the assembly level; a
dotted line at speedup = 1 marks break-even.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_mfem_timings as base

# Same filename scheme as the timing benchmark plus the _gam<0|1> suffix.
LOG_NAME_RE = re.compile(
    r"^(?P<tag>[A-Za-z0-9]+)_(?P<device>[A-Za-z0-9-]+)_(?P<asm>[A-Za-z]+)"
    r"(?:_p(?P<p>\d+))?(?:_n(?P<n>\d+))?_r(?P<r>\d+)_gam(?P<gam>[01])\.log$"
)

ORDER_COLORS = base.ORDER_COLORS
DEVICE_LINESTYLES = base.DEVICE_LINESTYLES
DEFAULT_COLOR = base.DEFAULT_COLOR
DEFAULT_LINESTYLE = base.DEFAULT_LINESTYLE
DEFAULT_MARKER = base.DEFAULT_MARKER
device_family = base.device_family

# Raw assembly-level markers. Unlike the timing plot, legacy and full are NOT
# merged here: GPU-aware MPI affects each assembly's communication differently, so
# every level is kept as its own series.
ASSEMBLY_MARKERS = {
    "legacy":  "o",
    "full":    "s",
    "element": "P",
    "partial": "^",
    "none":    "D",
}


def collect(results_dir: Path):
    """Pair gam0/gam1 logs and return per-example speedup series.

    Returns {tag: {(device, assembly, order): [(refinement, dofs, gt_off, gt_on), ...]}}
    where gt_off/gt_on are the {group: seconds} dicts for GPU-aware MPI off/on.
    """
    # raw[(tag, device, asm, order, ranks, r)] = {gam: (dofs, group_times)}
    raw: dict = {}
    for log in sorted(results_dir.glob("*.log")):
        m = LOG_NAME_RE.match(log.name)
        if not m:
            continue
        order = int(m.group("p")) if m.group("p") else 1
        ranks = int(m.group("n")) if m.group("n") else 1
        key = (m.group("tag"), m.group("device"), m.group("asm"), order, ranks,
               int(m.group("r")))
        dofs, group_times = base.parse_log(log)
        if dofs is None:
            print(f"  warning: {log.name} has no [MFEM_DOFS] line — skipped")
            continue
        if not group_times:
            print(f"  warning: {log.name} has no MFEM timer rows — skipped")
            continue
        raw.setdefault(key, {})[int(m.group("gam"))] = (dofs, group_times)

    runs: dict = {}
    for (tag, device, asm, order, _ranks, r), gams in raw.items():
        if 0 not in gams or 1 not in gams:
            continue  # need both on and off to form a ratio
        dofs_off, gt_off = gams[0]
        _dofs_on, gt_on = gams[1]
        runs.setdefault(tag, {}).setdefault((device, asm, order), []).append(
            (r, dofs_off, gt_off, gt_on)
        )
    for tag in runs:
        for k in runs[tag]:
            runs[tag][k].sort(key=lambda e: (e[0], e[1]))
    return runs


def plot_kind(tag: str, by_run, kind_label: str, components, out_path: Path):
    """Plot the GPU-aware MPI speedup for one section-kind of one example."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Stable order: known orders first, then device, then assembly level.
    keys_present = sorted(
        by_run.keys(),
        key=lambda k: (
            list(ORDER_COLORS).index(k[2]) if k[2] in ORDER_COLORS else 99,
            list(DEVICE_LINESTYLES).index(device_family(k[0])),
            list(ASSEMBLY_MARKERS).index(k[1]) if k[1] in ASSEMBLY_MARKERS else 99,
            k,
        ),
    )

    plotted_keys = []
    for device, asm, order in keys_present:
        color = ORDER_COLORS.get(order, DEFAULT_COLOR)
        linestyle = DEVICE_LINESTYLES.get(device_family(device), DEFAULT_LINESTYLE)
        marker = ASSEMBLY_MARKERS.get(asm, DEFAULT_MARKER)
        xs, ys = [], []
        for _r, dofs, gt_off, gt_on in by_run[(device, asm, order)]:
            v_off = base.kind_value(gt_off, components)
            v_on = base.kind_value(gt_on, components)
            if v_off is None or v_on is None:
                continue
            xs.append(dofs)
            ys.append(v_off / v_on)  # speedup from enabling GPU-aware MPI
        # A lone point draws no line, so the device (linestyle) is indistinguishable.
        if len(xs) < 2:
            continue
        ax.semilogx(xs, ys, color=color, linestyle=linestyle, marker=marker)
        plotted_keys.append((device, asm, order))

    if not plotted_keys:
        print(f"  {tag} [{kind_label}]: no paired on/off series with >=2 points — skipping plot")
        plt.close(fig)
        return

    # Break-even reference: speedup = 1 means GPU-aware MPI made no difference.
    ax.axhline(1.0, color="gray", linewidth=1, linestyle=":", zorder=0)

    # Three-part legend: order -> color, device -> linestyle, assembly -> marker.
    orders_present = [p for p in ORDER_COLORS if any(k[2] == p for k in plotted_keys)]
    orders_present += sorted({k[2] for k in plotted_keys} - set(orders_present))
    devices_present = sorted({k[0] for k in plotted_keys},
                             key=lambda d: (list(DEVICE_LINESTYLES).index(device_family(d)), d))
    assemblies_present = [a for a in ASSEMBLY_MARKERS if any(k[1] == a for k in plotted_keys)]
    assemblies_present += sorted({k[1] for k in plotted_keys} - set(assemblies_present))

    order_handles = [
        Line2D([0], [0], color=ORDER_COLORS.get(p, DEFAULT_COLOR), linestyle="-",
               label=f"p={p}")
        for p in orders_present
    ]
    device_handles = [
        Line2D([0], [0], color="black",
               linestyle=DEVICE_LINESTYLES.get(device_family(d), DEFAULT_LINESTYLE),
               label=d)
        for d in devices_present
    ]
    assembly_handles = [
        Line2D([0], [0], color="black", linestyle="none",
               marker=ASSEMBLY_MARKERS.get(a, DEFAULT_MARKER), label=a)
        for a in assemblies_present
    ]
    order_legend = ax.legend(handles=order_handles, title="Order", loc="upper left", fontsize=9)
    ax.add_artist(order_legend)
    device_legend = ax.legend(handles=device_handles, title="Device", loc="lower left", fontsize=9)
    ax.add_artist(device_legend)
    ax.legend(handles=assembly_handles, title="Assembly", loc="lower right", fontsize=9)

    ax.set_xlabel("Number of true DoFs")
    ax.set_ylabel("GPU-aware MPI speedup (time off / time on)")
    ax.set_title(f"{tag}: {kind_label} GPU-aware MPI speedup")
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
        print("No paired GPU-aware-MPI on/off logs found.", file=sys.stderr)
        sys.exit(1)

    for tag, by_run in runs.items():
        for kind_label, components, slug in base.PLOT_KINDS:
            plot_kind(
                tag, by_run, kind_label, components,
                results_dir / f"{tag}_{slug}_gpu_aware_mpi_speedup.png",
            )


if __name__ == "__main__":
    main()
