"""Multi-agent predictive analytics orchestrator powered by ``ChatGPTAgent``.

This module extends :mod:`agent` with a small framework that coordinates
multiple specialised agents around a shared dataset profile.  It is optimised
for exploratory analysis of large CSV-style datasets where a human operator
wants to bootstrap hypotheses, surface predictive indicators, and cross-reference
signals across different domain experts.

The design focuses on three pillars:

* **Data profiling** – Efficiently summarise large datasets without loading the
  entire file into memory.  Only lightweight statistics and a small sample of
  rows are collected locally before handing context to the language model.
* **Role-specialised agents** – Instantiate multiple ``ChatGPTAgent`` objects
  with targeted system prompts so that each agent contributes a unique
  perspective (profiling, indicator discovery, risk evaluation, synthesis).
* **Cross-referenced reporting** – Collate all intermediate agent outputs into a
  final report that emphasises alignment and disagreement between predictive
  indicators.

Example
-------

.. code-block:: bash

    python multi_agent.py data/marketing.csv \\
        --indicators "conversion_rate" "customer_lifetime_value" \\
        --model gpt-4.1-mini

The command above profiles ``marketing.csv`` and asks the multi-agent system to
prioritise the ``conversion_rate`` and ``customer_lifetime_value`` columns when
hunting for predictive signals.

The module exposes a :class:`MultiAgentPredictiveSystem` class that can be
integrated into larger applications.  Invoke ``python multi_agent.py --help``
for CLI usage information.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, MutableMapping, Optional, Sequence

from agent import ChatGPTAgent


@dataclass
class NumericColumnSummary:
    """Streaming statistics for a numeric column."""

    name: str
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the current mean.
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.minimum is None or value < self.minimum:
            self.minimum = value
        if self.maximum is None or value > self.maximum:
            self.maximum = value

    @property
    def variance(self) -> Optional[float]:
        if self.count < 2:
            return None
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> Optional[float]:
        variance = self.variance
        if variance is None:
            return None
        return math.sqrt(variance)


@dataclass
class DatasetProfile:
    """Aggregated view of the dataset consumed by the agent network."""

    row_count: int
    column_types: Dict[str, str]
    numeric_columns: Dict[str, NumericColumnSummary]
    categorical_top_values: Dict[str, Counter]
    sample_rows: List[Dict[str, str]]
    missing_counts: Dict[str, int]

    def render_markdown(self, max_categories: int = 5, sample_limit: int = 5) -> str:
        """Convert the profile into a Markdown table for LLM context."""

        lines: List[str] = []
        lines.append(f"Total rows analysed: **{self.row_count}**")
        lines.append("")
        lines.append("### Column Overview")
        lines.append("| Column | Type | Missing | Details |")
        lines.append("| --- | --- | --- | --- |")
        for column, column_type in self.column_types.items():
            missing = self.missing_counts.get(column, 0)
            detail: str
            if column_type == "numeric":
                stats = self.numeric_columns[column]
                detail_parts = [
                    f"count={stats.count}",
                    f"mean={stats.mean:.3f}" if stats.count else "mean=n/a",
                ]
                if stats.stddev is not None:
                    detail_parts.append(f"std={stats.stddev:.3f}")
                if stats.minimum is not None and stats.maximum is not None:
                    detail_parts.append(
                        f"range=({stats.minimum:.3f}, {stats.maximum:.3f})"
                    )
                detail = ", ".join(detail_parts)
            else:
                categories = self.categorical_top_values.get(column, Counter())
                top_items = categories.most_common(max_categories)
                if not top_items:
                    detail = "No non-missing values"
                else:
                    pretty = ", ".join(
                        f"{value} ({count})" for value, count in top_items
                    )
                    detail = f"Top values: {pretty}"
            lines.append(
                f"| {column} | {column_type} | {missing} | {detail} |"
            )

        if self.sample_rows:
            lines.append("")
            lines.append("### Sample Rows")
            lines.append("```")
            for row in self.sample_rows[:sample_limit]:
                flattened = ", ".join(f"{k}={v}" for k, v in row.items())
                lines.append(flattened)
            lines.append("```")

        return "\n".join(lines)


def _detect_column_types(
    first_row: MutableMapping[str, str]
) -> Dict[str, Dict[str, bool]]:
    """Initialise per-column flags used during type detection."""

    flags: Dict[str, Dict[str, bool]] = {}
    for column in first_row:
        flags[column] = {"numeric_candidate": True, "seen_numeric": False}
    return flags


def _profile_dataset(
    path: Path,
    *,
    sample_rows: int = 15,
    max_rows: Optional[int] = None,
) -> DatasetProfile:
    """Stream the dataset and compute summary statistics.

    Parameters
    ----------
    path:
        Path to a CSV-like file.  Files are read using :class:`csv.DictReader`
        so delimiters other than commas are supported when the header row is
        separated by the same delimiter throughout the file.
    sample_rows:
        Number of rows to keep for contextual examples passed to the language
        model.
    max_rows:
        Optional hard limit on the number of rows to analyse.  ``None`` means
        "read the entire file".
    """

    with path.open("r", newline="", encoding="utf-8") as handle:
        # Attempt to sniff the dialect for better compatibility with TSV and
        # pipe-separated datasets.
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(handle, dialect=dialect)
        try:
            first_row = next(reader)
        except StopIteration as exc:  # pragma: no cover - empty dataset guard.
            raise ValueError(f"Dataset '{path}' does not contain any rows.") from exc

        column_flags = _detect_column_types(first_row)
        numeric_columns = {
            column: NumericColumnSummary(column)
            for column in first_row.keys()
        }
        categorical_counters: Dict[str, Counter] = defaultdict(Counter)
        missing_counts: Dict[str, int] = defaultdict(int)
        captured_rows: List[Dict[str, str]] = []

        def process_row(row: MutableMapping[str, str]) -> None:
            nonlocal captured_rows
            if len(captured_rows) < sample_rows:
                captured_rows.append(dict(row))

            for column, raw_value in row.items():
                value = raw_value.strip() if raw_value is not None else ""
                if value == "":
                    missing_counts[column] += 1
                    continue

                flags = column_flags[column]
                if flags["numeric_candidate"]:
                    try:
                        numeric_value = float(value)
                    except ValueError:
                        flags["numeric_candidate"] = False
                        categorical_counters[column][value] += 1
                    else:
                        flags["seen_numeric"] = True
                        numeric_columns[column].update(numeric_value)
                else:
                    categorical_counters[column][value] += 1

        row_count = 1
        process_row(first_row)

        for row in reader:
            row_count += 1
            process_row(row)
            if max_rows is not None and row_count >= max_rows:
                break

    column_types: Dict[str, str] = {}
    for column, flags in column_flags.items():
        if flags["numeric_candidate"] and flags["seen_numeric"]:
            column_types[column] = "numeric"
        else:
            column_types[column] = "categorical"
            numeric_columns.pop(column, None)

    # Clean up categorical counters that correspond to numeric columns.
    for column in list(categorical_counters.keys()):
        if column_types.get(column) == "numeric":
            categorical_counters.pop(column)

    return DatasetProfile(
        row_count=row_count,
        column_types=column_types,
        numeric_columns=numeric_columns,
        categorical_top_values=dict(categorical_counters),
        sample_rows=captured_rows,
        missing_counts=dict(missing_counts),
    )


@dataclass
class MultiAgentConfig:
    """Configuration for the predictive multi-agent workflow."""

    model: str = "gpt-4o-mini"
    base_temperature: float = 0.4
    indicator_focus: Sequence[str] = field(default_factory=list)
    max_rows: Optional[int] = None


class MultiAgentPredictiveSystem:
    """Coordinate a cohort of specialised ``ChatGPTAgent`` instances."""

    def __init__(self, dataset: Path, config: Optional[MultiAgentConfig] = None):
        self.dataset = dataset
        self.config = config or MultiAgentConfig()
        self._profile: Optional[DatasetProfile] = None

        focus_hint = (
            "Focus on these business KPIs: "
            + ", ".join(self.config.indicator_focus)
            if self.config.indicator_focus
            else "Prioritise statistically meaningful signals."
        )

        self.profiling_agent = ChatGPTAgent(
            model=self.config.model,
            system_prompt=textwrap.dedent(
                f"""
                You are a senior data analyst.  Summarise the supplied dataset
                profile, highlight data quality concerns, and surface important
                distributions that merit deeper investigation.  {focus_hint}
                """
            ).strip(),
        )
        self.indicator_agent = ChatGPTAgent(
            model=self.config.model,
            system_prompt=textwrap.dedent(
                f"""
                You are a predictive signal researcher.  Use the dataset profile
                and analyst notes to propose leading indicators, lagging
                indicators, and supporting features that could power forecasting
                models.  {focus_hint}  Explain why each indicator is relevant.
                """
            ).strip(),
        )
        self.risk_agent = ChatGPTAgent(
            model=self.config.model,
            system_prompt=textwrap.dedent(
                """
                You are a risk and bias auditor.  Examine the proposed
                predictive indicators for potential pitfalls such as data leakage,
                sampling bias, or operational concerns.  Provide concrete
                mitigation strategies when issues are identified.
                """
            ).strip(),
        )
        self.synthesis_agent = ChatGPTAgent(
            model=self.config.model,
            system_prompt=textwrap.dedent(
                """
                You are an executive insights partner.  Combine analyst,
                indicator, and risk perspectives into an actionable briefing.
                Emphasise cross-referenced evidence and next steps for model
                development.
                """
            ).strip(),
        )

    @property
    def profile(self) -> DatasetProfile:
        if self._profile is None:
            self._profile = _profile_dataset(
                self.dataset,
                max_rows=self.config.max_rows,
            )
        return self._profile

    def _build_context(self) -> str:
        return self.profile.render_markdown()

    def run(self) -> str:
        """Execute the multi-agent workflow and return the final report."""

        context = self._build_context()

        profiling_notes = self.profiling_agent.ask(
            textwrap.dedent(
                f"""
                Dataset profile:

                {context}

                Provide a concise analysis in Markdown.
                """
            ).strip(),
            temperature=self.config.base_temperature,
        )

        indicator_plan = self.indicator_agent.ask(
            textwrap.dedent(
                f"""
                Dataset profile:

                {context}

                Analyst notes:
                {profiling_notes}

                Derive predictive indicators and supportive feature sets.  Cite
                specific columns or combinations of columns when relevant.
                """
            ).strip(),
            temperature=self.config.base_temperature,
        )

        risk_review = self.risk_agent.ask(
            textwrap.dedent(
                f"""
                Dataset profile:

                {context}

                Proposed predictive indicators:
                {indicator_plan}

                Audit the plan for risks, leakage, and operational challenges.
                Provide mitigations or validation tactics for each concern.
                """
            ).strip(),
            temperature=self.config.base_temperature,
        )

        final_report = self.synthesis_agent.ask(
            textwrap.dedent(
                f"""
                Dataset profile summary:
                {context}

                Analyst findings:
                {profiling_notes}

                Predictive indicator plan:
                {indicator_plan}

                Risk audit:
                {risk_review}

                Compose a final cross-referenced report that:
                - Aligns overlapping insights.
                - Highlights conflicting views with resolution strategies.
                - Recommends concrete next actions for data scientists and
                  business stakeholders.
                Present the response using Markdown headings and bullet lists.
                """
            ).strip(),
            temperature=max(0.2, self.config.base_temperature - 0.1),
        )

        return final_report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the CSV dataset to analyse.",
    )
    parser.add_argument(
        "--indicator",
        "--indicators",
        dest="indicators",
        nargs="*",
        default=[],
        help="Optional columns or KPIs to prioritise when searching for predictive indicators.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model identifier used for every agent (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Base sampling temperature for agent responses (default: %(default)s)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows to profile (useful for very large files).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the dataset profile instead of engaging the language model.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        parser.error(f"Dataset '{args.dataset}' does not exist.")

    config = MultiAgentConfig(
        model=args.model,
        base_temperature=args.temperature,
        indicator_focus=args.indicators,
        max_rows=args.max_rows,
    )

    system = MultiAgentPredictiveSystem(args.dataset, config=config)

    if args.preview:
        print(system.profile.render_markdown())
        return 0

    report = system.run()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
