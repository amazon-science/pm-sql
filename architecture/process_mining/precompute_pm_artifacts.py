"""
Pre-compute process mining artifacts from the XES event log using pm4py.

Generates text files in pm_data/ that the get_process_info tool reads at runtime.
Run this once before running the Step 5 experiment.
"""

import pm4py
from pathlib import Path
from collections import Counter

OUTPUT_DIR = Path(__file__).parent / "pm_data"
XES_PATH = Path(__file__).resolve().parents[2] / "BPIC2020_InternationalDeclarations.xes"


def compute_dfg(log):
    """Full DFG with frequency and mean performance (seconds)."""
    dfg_freq, start_activities, end_activities = pm4py.discover_dfg(log)
    dfg_perf, _, _ = pm4py.discover_performance_dfg(log)

    lines = []
    lines.append("# Directly-Follows Graph (DFG)")
    lines.append("# Each line: SourceActivity -> TargetActivity | frequency | mean_time_seconds")
    lines.append(f"# Total edges: {len(dfg_freq)}")
    lines.append("")

    lines.append("## Start Activities (first activity in a case)")
    for act, freq in sorted(start_activities.items(), key=lambda x: -x[1]):
        lines.append(f"  {act}: {freq} cases")
    lines.append("")

    lines.append("## End Activities (last activity in a case)")
    for act, freq in sorted(end_activities.items(), key=lambda x: -x[1]):
        lines.append(f"  {act}: {freq} cases")
    lines.append("")

    lines.append("## Directly-Follows Edges (sorted by frequency)")
    sorted_edges = sorted(dfg_freq.items(), key=lambda x: -x[1])
    for (src, tgt), freq in sorted_edges:
        mean_time = dfg_perf.get((src, tgt), 0)
        if isinstance(mean_time, dict):
            mean_time = mean_time.get('mean', 0)
        days = mean_time / 86400 if mean_time else 0
        lines.append(f"  {src} -> {tgt} | freq={freq} | mean={days:.1f} days")

    return "\n".join(lines)


def compute_variants(log):
    """Top process variants with frequencies."""
    raw_variants = pm4py.get_variants(log)
    # pm4py returns {tuple_of_activities: list_of_traces} — convert to counts
    variants = {k: len(v) for k, v in raw_variants.items()}
    total_cases = sum(variants.values())
    sorted_vars = sorted(variants.items(), key=lambda x: -x[1])

    lines = []
    lines.append("# Process Variants (most common execution paths)")
    lines.append(f"# Total unique variants: {len(variants)}")
    lines.append(f"# Total cases: {total_cases}")
    lines.append("")

    cumulative = 0
    for i, (variant, count) in enumerate(sorted_vars[:25], 1):
        cumulative += count
        pct = count / total_cases * 100
        cum_pct = cumulative / total_cases * 100
        activities = variant if isinstance(variant, str) else " -> ".join(variant)
        lines.append(f"Variant {i} ({count} cases, {pct:.1f}%, cumulative {cum_pct:.1f}%):")
        lines.append(f"  {activities}")
        lines.append("")

    return "\n".join(lines)


def compute_activities(log):
    """Activity statistics: frequency, which cases have them, rework rates."""
    from pm4py.statistics.attributes.log import get as attr_get
    activities = attr_get.get_attribute_values(log, "concept:name")

    # Compute rework: cases where an activity appears more than once
    activity_rework = Counter()
    activity_cases = Counter()
    for trace in log:
        case_activities = Counter()
        for event in trace:
            case_activities[event["concept:name"]] += 1
        for act, cnt in case_activities.items():
            activity_cases[act] += 1
            if cnt > 1:
                activity_rework[act] += 1

    total_cases = len(log)

    lines = []
    lines.append("# Activity Statistics")
    lines.append(f"# Total distinct activities: {len(activities)}")
    lines.append(f"# Total events: {sum(activities.values())}")
    lines.append(f"# Total cases: {total_cases}")
    lines.append("")
    lines.append("| Activity | Events | Cases | Rework Cases | Rework % |")
    lines.append("|---|---|---|---|---|")

    for act, freq in sorted(activities.items(), key=lambda x: -x[1]):
        cases = activity_cases[act]
        rework = activity_rework.get(act, 0)
        rework_pct = rework / cases * 100 if cases > 0 else 0
        lines.append(f"| {act} | {freq} | {cases} | {rework} | {rework_pct:.1f}% |")

    return "\n".join(lines)


def compute_case_stats(log):
    """Case-level statistics: duration, event counts, lifecycle."""

    durations = []
    event_counts = []
    first_activities = Counter()
    last_activities = Counter()

    for trace in log:
        events = sorted(trace, key=lambda e: e["time:timestamp"])
        event_counts.append(len(events))
        first_activities[events[0]["concept:name"]] += 1
        last_activities[events[-1]["concept:name"]] += 1
        start = events[0]["time:timestamp"]
        end = events[-1]["time:timestamp"]
        dur = (end - start).total_seconds() / 86400  # days
        durations.append(dur)

    lines = []
    lines.append("# Case Statistics")
    lines.append(f"# Total cases: {len(log)}")
    lines.append("")
    lines.append("## Duration (days)")
    lines.append(f"  Mean: {sum(durations)/len(durations):.1f}")
    lines.append(f"  Median: {sorted(durations)[len(durations)//2]:.1f}")
    lines.append(f"  Min: {min(durations):.1f}")
    lines.append(f"  Max: {max(durations):.1f}")
    lines.append("")
    lines.append("## Events per case")
    lines.append(f"  Mean: {sum(event_counts)/len(event_counts):.1f}")
    lines.append(f"  Median: {sorted(event_counts)[len(event_counts)//2]}")
    lines.append(f"  Min: {min(event_counts)}")
    lines.append(f"  Max: {max(event_counts)}")
    lines.append("")
    lines.append("## First activity in cases")
    for act, cnt in sorted(first_activities.items(), key=lambda x: -x[1]):
        lines.append(f"  {act}: {cnt} cases ({cnt/len(log)*100:.1f}%)")
    lines.append("")
    lines.append("## Last activity in cases")
    for act, cnt in sorted(last_activities.items(), key=lambda x: -x[1]):
        lines.append(f"  {act}: {cnt} cases ({cnt/len(log)*100:.1f}%)")

    return "\n".join(lines)


def compute_rework(log):
    """Detailed rework analysis: which activities repeat, self-distances."""
    activity_rework_cases = Counter()
    total_rework_cases = set()

    for trace in log:
        case_id = trace.attributes.get("concept:name", "")
        case_activities = Counter()
        for event in trace:
            case_activities[event["concept:name"]] += 1
        has_rework = False
        for act, cnt in case_activities.items():
            if cnt > 1:
                activity_rework_cases[act] += 1
                has_rework = True
        if has_rework:
            total_rework_cases.add(case_id)

    # Consecutive repetitions (same activity immediately followed by itself)
    consecutive_rework = Counter()
    for trace in log:
        events = sorted(trace, key=lambda e: e["time:timestamp"])
        for i in range(len(events) - 1):
            if events[i]["concept:name"] == events[i+1]["concept:name"]:
                consecutive_rework[events[i]["concept:name"]] += 1

    lines = []
    lines.append("# Rework Analysis")
    lines.append(f"# Cases with any rework: {len(total_rework_cases)} / {len(log)} ({len(total_rework_cases)/len(log)*100:.1f}%)")
    lines.append("")
    lines.append("## Activities with rework (repeated within same case)")
    lines.append("| Activity | Cases with Rework |")
    lines.append("|---|---|")
    for act, cnt in sorted(activity_rework_cases.items(), key=lambda x: -x[1]):
        lines.append(f"| {act} | {cnt} |")
    lines.append("")
    lines.append("## Consecutive repetitions (same activity immediately repeated)")
    lines.append("| Activity | Consecutive Repetitions |")
    lines.append("|---|---|")
    for act, cnt in sorted(consecutive_rework.items(), key=lambda x: -x[1]):
        lines.append(f"| {act} | {cnt} |")

    return "\n".join(lines)


def compute_resources(log):
    """Resource statistics."""
    from pm4py.statistics.attributes.log import get as attr_get
    resources = attr_get.get_attribute_values(log, "org:resource")

    # Resource per activity
    resource_activities = {}
    for trace in log:
        for event in trace:
            res = event.get("org:resource", "UNKNOWN")
            act = event["concept:name"]
            if res not in resource_activities:
                resource_activities[res] = Counter()
            resource_activities[res][act] += 1

    # Cases per resource
    resource_cases = Counter()
    for trace in log:
        case_resources = set()
        for event in trace:
            case_resources.add(event.get("org:resource", "UNKNOWN"))
        for res in case_resources:
            resource_cases[res] += 1

    lines = []
    lines.append("# Resource Statistics")
    lines.append(f"# Total distinct resources: {len(resources)}")
    lines.append("")
    lines.append("| Resource | Events | Cases | Top Activity |")
    lines.append("|---|---|---|---|")
    for res, freq in sorted(resources.items(), key=lambda x: -x[1])[:20]:
        cases = resource_cases[res]
        top_act = resource_activities[res].most_common(1)[0][0] if res in resource_activities else "N/A"
        lines.append(f"| {res} | {freq} | {cases} | {top_act} |")

    return "\n".join(lines)


def compute_process_tree(log):
    """Process tree from inductive miner — shows process structure."""
    tree = pm4py.discover_process_tree_inductive(log)
    tree_str = str(tree)

    lines = []
    lines.append("# Process Tree (Inductive Miner)")
    lines.append("# Operators: -> (sequence), X (exclusive choice), + (parallel), * (loop)")
    lines.append("# tau = silent/invisible transition (skip)")
    lines.append("")
    lines.append(tree_str)

    return "\n".join(lines)


def main():
    print(f"Loading XES file: {XES_PATH}")
    log = pm4py.read_xes(str(XES_PATH), return_legacy_log_object=True)
    print(f"Loaded {len(log)} cases")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "dfg.txt": compute_dfg,
        "variants.txt": compute_variants,
        "activities.txt": compute_activities,
        "case_stats.txt": compute_case_stats,
        "rework.txt": compute_rework,
        "resources.txt": compute_resources,
        "process_tree.txt": compute_process_tree,
    }

    for filename, compute_fn in artifacts.items():
        print(f"Computing {filename}...")
        try:
            text = compute_fn(log)
            output_path = OUTPUT_DIR / filename
            output_path.write_text(text)
            print(f"  Saved: {output_path} ({len(text)} chars, ~{len(text)//4} tokens)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone. All artifacts saved to pm_data/")


if __name__ == "__main__":
    main()
