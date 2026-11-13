"""
Process mining artifacts tool for Step 5.

Provides pre-computed process mining knowledge to help the agent
understand process structure, activity relationships, and domain concepts.
"""

from pathlib import Path

from strands import tool

PM_DATA_DIR = Path(__file__).parent / "pm_data"

VALID_ASPECTS = {
    "dfg": "dfg.txt",
    "variants": "variants.txt",
    "activities": "activities.txt",
    "case_stats": "case_stats.txt",
    "rework": "rework.txt",
    "resources": "resources.txt",
    "process_tree": "process_tree.txt",
}


@tool
def get_process_info(aspect: str) -> str:
    """Get process mining information about the event log to help write correct SQL.

    Use this tool BEFORE writing SQL when:
    - The question uses domain terminology (e.g., "rework", "happy path", "bottleneck")
    - You need to know which activities follow which (directly-follows relationships)
    - You need to understand the overall process structure
    - You need exact activity names or their frequencies

    Args:
        aspect: One of:
            - "dfg": Directly-follows graph — which activities follow which, with frequencies and mean time between them
            - "variants": Top 25 most common process execution paths (activity sequences)
            - "activities": All 34 activities with event counts, case counts, and rework rates
            - "case_stats": Case duration stats, events per case, first/last activities
            - "rework": Which activities are repeated within cases, consecutive repetitions
            - "resources": Resource statistics (event counts, cases, top activities)
            - "process_tree": Hierarchical process model showing sequence, choice, parallelism, and loops

    Returns:
        Pre-computed process mining text for the requested aspect.
    """
    if aspect not in VALID_ASPECTS:
        return f"ERROR: Invalid aspect '{aspect}'. Valid options: {', '.join(sorted(VALID_ASPECTS.keys()))}"

    filepath = PM_DATA_DIR / VALID_ASPECTS[aspect]
    if not filepath.exists():
        return f"ERROR: Data file not found for aspect '{aspect}'. Run precompute_pm_artifacts.py first."

    return filepath.read_text()
