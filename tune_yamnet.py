#!/usr/bin/env python3
"""
Tune ScreamFinder's YAMNet vocalization detector against labeled test folders.

Expected folder layout:
  test_files/testN/
    Labels 1.txt
    <one media file>

Each label line should be:
  <start_sec>\t<end_sec>\t<label1,label2,...>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from screamfinder import MEDIA_EXTENSIONS, analyze_yamnet_streaming


ANNOTATION_TAG_ALIASES = {
    "scream": "vocal_peak",
    "screaming": "vocal_peak",
    "moan": "vocal_sustain",
    "moaning": "vocal_sustain",
    "groan": "vocal_sustain",
    "crying": "vocal_distress",
    "whimper": "vocal_distress",
    "whimpering": "vocal_distress",
    "gasp": "vocal_breath",
    "breathing": "vocal_breath",
    "gagged": "vocal_breath",
    "speech": "speech",
    "female": "speech",
    "male": "speech",
    "laugh": "laughter",
    "laughing": "laughter",
    "music": "music",
    "slap": "impact",
    "slapping": "impact",
    "spank": "impact",
    "spanking": "impact",
    "hands": "impact",
    "street noise": "ambient_noise",
    "street sounds": "ambient_noise",
    "buzzing": "ambient_noise",
    "clipped": "artifact",
}

POSITIVE_ANNOTATION_CATEGORIES = {
    "vocal_peak",
    "vocal_sustain",
    "vocal_distress",
    "vocal_breath",
}

YAMNET_LABEL_ALIASES = {
    "Screaming": "vocal_peak",
    "Shout": "vocal_peak",
    "Yell": "vocal_peak",
    "Bellow": "vocal_peak",
    "Wail, moan": "vocal_sustain",
    "Groan": "vocal_sustain",
    "Grunt": "vocal_sustain",
    "Crying, sobbing": "vocal_distress",
    "Whimper": "vocal_distress",
    "Baby cry, infant cry": "vocal_distress",
    "Gasp": "vocal_breath",
    "Breathing": "vocal_breath",
    "Pant": "vocal_breath",
    "Sigh": "vocal_breath",
    "Speech": "speech",
    "Conversation": "speech",
    "Narration, monologue": "speech",
    "Child speech, kid speaking": "speech",
    "Babbling": "speech",
    "Hubbub, speech noise, speech babble": "speech",
    "Laughter": "laughter",
    "Giggle": "laughter",
    "Snicker": "laughter",
    "Chuckle, chortle": "laughter",
    "Belly laugh": "laughter",
    "Baby laughter": "laughter",
    "Music": "music",
    "Background music": "music",
    "Song": "music",
    "Singing": "music",
    "Vocal music": "music",
    "Whoop": "crowd_noise",
    "Cheering": "crowd_noise",
    "Crowd": "crowd_noise",
    "Hands": "impact",
    "Slap, smack": "impact",
    "Whack, thwack": "impact",
    "Burst, pop": "impact",
    "Whip": "impact",
    "Buzz": "ambient_noise",
    "Mains hum": "ambient_noise",
    "Outside, urban or manmade": "ambient_noise",
    "Traffic noise, roadway noise": "ambient_noise",
    "Vehicle": "ambient_noise",
    "Dog": "animal_noise",
    "Domestic animals, pets": "animal_noise",
    "Animal": "animal_noise",
    "Whimper (dog)": "animal_noise",
    "Yip": "animal_noise",
    "Bow-wow": "animal_noise",
    "Howl": "animal_noise",
}


@dataclass
class Annotation:
    start: float
    end: float
    tags: Tuple[str, ...]

    @property
    def is_positive(self) -> bool:
        return any(category in POSITIVE_ANNOTATION_CATEGORIES for category in self.categories)

    @property
    def categories(self) -> Tuple[str, ...]:
        return tuple(sorted({canonicalize_annotation_tag(tag) for tag in self.tags}))


@dataclass
class TestCase:
    name: str
    media_path: Path
    labels_path: Path
    annotations: List[Annotation]


@dataclass
class WindowEval:
    case_name: str
    start: float
    end: float
    score: float
    predicted: bool
    truth: str
    overlap_tags: Tuple[str, ...]
    overlap_categories: Tuple[str, ...]
    top_labels: Tuple[str, ...]
    top_label_categories: Tuple[str, ...]


def canonicalize_annotation_tag(tag: str) -> str:
    return ANNOTATION_TAG_ALIASES.get(tag, tag)


def canonicalize_yamnet_label(label: str) -> str:
    return YAMNET_LABEL_ALIASES.get(label, label)


def categories_for_annotation_tags(tags: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({canonicalize_annotation_tag(tag) for tag in tags}))


def categories_for_yamnet_labels(labels: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted({canonicalize_yamnet_label(label) for label in labels}))


@dataclass
class ReviewRegion:
    start: float
    end: float
    peak_score: float
    window_count: int
    top_labels: Tuple[str, ...]


def parse_labels_file(path: Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise ValueError(f"{path}:{lineno}: expected start/end/tags separated by tabs")
        start = float(parts[0])
        end = float(parts[1])
        tags = tuple(tag.strip().lower() for tag in parts[2].split(",") if tag.strip())
        annotations.append(Annotation(start=start, end=end, tags=tags))
    return annotations


def load_test_cases(root: Path) -> List[TestCase]:
    cases: List[TestCase] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        labels = sorted(child.glob("Labels*.txt"))
        media = sorted(p for p in child.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
        if len(labels) != 1 or len(media) != 1:
            continue
        cases.append(
            TestCase(
                name=child.name,
                media_path=media[0],
                labels_path=labels[0],
                annotations=parse_labels_file(labels[0]),
            )
        )
    return cases


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def window_truth(
    annotations: Sequence[Annotation],
    start: float,
    end: float,
    min_overlap_ratio: float,
) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    window_len = max(1e-9, end - start)
    tags: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    positive = False
    negative = False
    for ann in annotations:
        ov = overlap_seconds(start, end, ann.start, ann.end)
        if ov <= 0:
            continue
        for tag in ann.tags:
            tags[tag] += 1
        for category in ann.categories:
            categories[category] += 1
        if (ov / window_len) >= min_overlap_ratio:
            if ann.is_positive:
                positive = True
            else:
                negative = True
    if positive:
        return "positive", tuple(sorted(tags)), tuple(sorted(categories))
    if negative:
        return "negative", tuple(sorted(tags)), tuple(sorted(categories))
    return "unknown", tuple(sorted(tags)), tuple(sorted(categories))


def collect_case_windows(
    case: TestCase,
    model_handle: str,
    top_k: int,
    score_threshold: float,
    min_window_rms: float,
    context_rms_seconds: float,
    context_rms_ratio: float,
    min_overlap_ratio: float,
) -> Tuple[dict, List[WindowEval]]:
    analysis = analyze_yamnet_streaming(
        case.media_path,
        model_handle=model_handle,
        score_threshold=score_threshold,
        min_window_rms=min_window_rms,
        context_rms_seconds=context_rms_seconds,
        context_rms_ratio=context_rms_ratio,
        collect_debug=True,
        top_k=top_k,
    )
    debug = analysis.get("yamnet_debug") or {}
    windows = debug.get("windows") or []
    rows: List[WindowEval] = []
    for window in windows:
        truth, overlap_tags, overlap_categories = window_truth(
            case.annotations,
            float(window["start"]),
            float(window["end"]),
            min_overlap_ratio=min_overlap_ratio,
        )
        score = float(window["vocalization_score"])
        top_labels = tuple(item["label"] for item in window.get("top_labels", []))
        rows.append(
            WindowEval(
                case_name=case.name,
                start=float(window["start"]),
                end=float(window["end"]),
                score=score,
                predicted=score >= score_threshold,
                truth=truth,
                overlap_tags=overlap_tags,
                overlap_categories=overlap_categories,
                top_labels=top_labels,
                top_label_categories=categories_for_yamnet_labels(top_labels),
            )
        )
    return analysis, rows


def collect_dataset(
    test_root: Path,
    model_handle: str,
    top_k: int,
    score_threshold: float,
    min_window_rms: float,
    context_rms_seconds: float,
    context_rms_ratio: float,
    min_overlap_ratio: float,
) -> Tuple[List[TestCase], List[dict], List[WindowEval]]:
    cases = load_test_cases(test_root)
    if not cases:
        raise RuntimeError(f"No labeled test folders found under {test_root}")

    analyses: List[dict] = []
    windows: List[WindowEval] = []
    for case in cases:
        analysis, case_rows = collect_case_windows(
            case,
            model_handle=model_handle,
            top_k=top_k,
            score_threshold=score_threshold,
            min_window_rms=min_window_rms,
            context_rms_seconds=context_rms_seconds,
            context_rms_ratio=context_rms_ratio,
            min_overlap_ratio=min_overlap_ratio,
        )
        analyses.append({
            "case": case.name,
            "media_path": str(case.media_path),
            "labels_path": str(case.labels_path),
            "annotation_count": len(case.annotations),
            "window_count": len(case_rows),
            "analysis_duration": round(float(analysis["duration"]), 3),
        })
        windows.extend(case_rows)
    return cases, analyses, windows


def evaluate_threshold(windows: Sequence[WindowEval], threshold: float) -> Dict[str, float]:
    tp = fp = fn = tn = 0
    ignored = 0
    for row in windows:
        if row.truth == "unknown":
            ignored += 1
            continue
        pred = row.score >= threshold
        actual = row.truth == "positive"
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ignored": float(ignored),
    }


def threshold_grid(start: float, stop: float, step: float) -> List[float]:
    vals: List[float] = []
    cur = start
    while cur <= stop + 1e-9:
        vals.append(round(cur, 4))
        cur += step
    return vals


def summarize_labels(windows: Iterable[WindowEval], limit: int = 12) -> List[Tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in windows:
        counts.update(row.top_labels)
    return counts.most_common(limit)


def summarize_mapped_label_categories(
    windows: Iterable[WindowEval],
    limit: int = 12,
) -> List[Tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in windows:
        counts.update(row.top_label_categories)
    return counts.most_common(limit)


def summarize_overlap_tags(windows: Iterable[WindowEval], limit: int = 12) -> List[Tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in windows:
        counts.update(row.overlap_tags)
    return counts.most_common(limit)


def summarize_overlap_categories(windows: Iterable[WindowEval], limit: int = 12) -> List[Tuple[str, int]]:
    counts: Counter[str] = Counter()
    for row in windows:
        counts.update(row.overlap_categories)
    return counts.most_common(limit)


def case_summary(rows: Sequence[WindowEval], threshold: float) -> List[dict]:
    out: List[dict] = []
    by_case: Dict[str, List[WindowEval]] = {}
    for row in rows:
        by_case.setdefault(row.case_name, []).append(row)
    for case_name, items in sorted(by_case.items()):
        stats = evaluate_threshold(items, threshold)
        out.append({
            "case": case_name,
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "f1": round(stats["f1"], 4),
            "tp": int(stats["tp"]),
            "fp": int(stats["fp"]),
            "fn": int(stats["fn"]),
            "tn": int(stats["tn"]),
        })
    return out


def build_report(
    test_root: Path,
    model_handle: str,
    top_k: int,
    score_threshold: float,
    min_window_rms: float,
    context_rms_seconds: float,
    context_rms_ratio: float,
    min_overlap_ratio: float,
    sweep_start: float,
    sweep_stop: float,
    sweep_step: float,
) -> Tuple[dict, List[TestCase], List[WindowEval]]:
    cases, analyses, windows = collect_dataset(
        test_root=test_root,
        model_handle=model_handle,
        top_k=top_k,
        score_threshold=score_threshold,
        min_window_rms=min_window_rms,
        context_rms_seconds=context_rms_seconds,
        context_rms_ratio=context_rms_ratio,
        min_overlap_ratio=min_overlap_ratio,
    )
    sweep = [
        evaluate_threshold(windows, thr)
        for thr in threshold_grid(sweep_start, sweep_stop, sweep_step)
    ]
    best = max(sweep, key=lambda item: (item["f1"], item["precision"], item["recall"]))

    tp_rows = [row for row in windows if row.truth == "positive" and row.score >= best["threshold"]]
    fp_rows = [row for row in windows if row.truth == "negative" and row.score >= best["threshold"]]
    fn_rows = [row for row in windows if row.truth == "positive" and row.score < best["threshold"]]
    unlabeled_high_rows = [
        row for row in windows if row.truth == "unknown" and row.score >= best["threshold"]
    ]

    return {
        "test_root": str(test_root),
        "model_handle": model_handle,
        "positive_annotation_categories": sorted(POSITIVE_ANNOTATION_CATEGORIES),
        "annotation_tag_aliases": ANNOTATION_TAG_ALIASES,
        "yamnet_label_aliases": YAMNET_LABEL_ALIASES,
        "top_k": top_k,
        "yamnet_min_window_rms": min_window_rms,
        "yamnet_context_rms_seconds": context_rms_seconds,
        "yamnet_context_rms_ratio": context_rms_ratio,
        "min_overlap_ratio": min_overlap_ratio,
        "case_count": len(cases),
        "window_count": len(windows),
        "cases": analyses,
        "best_threshold": round(best["threshold"], 4),
        "best_metrics": {
            "precision": round(best["precision"], 4),
            "recall": round(best["recall"], 4),
            "f1": round(best["f1"], 4),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "fn": int(best["fn"]),
            "tn": int(best["tn"]),
            "ignored_windows": int(best["ignored"]),
        },
        "threshold_sweep": [
            {
                "threshold": round(item["threshold"], 4),
                "precision": round(item["precision"], 4),
                "recall": round(item["recall"], 4),
                "f1": round(item["f1"], 4),
                "tp": int(item["tp"]),
                "fp": int(item["fp"]),
                "fn": int(item["fn"]),
                "tn": int(item["tn"]),
                "ignored": int(item["ignored"]),
            }
            for item in sweep
        ],
        "case_metrics": case_summary(windows, best["threshold"]),
        "top_labels_true_positive": summarize_labels(tp_rows),
        "top_label_categories_true_positive": summarize_mapped_label_categories(tp_rows),
        "top_labels_false_positive": summarize_labels(fp_rows),
        "top_label_categories_false_positive": summarize_mapped_label_categories(fp_rows),
        "annotation_tags_false_positive": summarize_overlap_tags(fp_rows),
        "annotation_categories_false_positive": summarize_overlap_categories(fp_rows),
        "annotation_tags_false_negative": summarize_overlap_tags(fn_rows),
        "annotation_categories_false_negative": summarize_overlap_categories(fn_rows),
        "top_labels_unlabeled_high_score": summarize_labels(unlabeled_high_rows),
        "top_label_categories_unlabeled_high_score": summarize_mapped_label_categories(unlabeled_high_rows),
        "false_positive_examples": [
            {
                "case": row.case_name,
                "start": round(row.start, 3),
                "end": round(row.end, 3),
                "score": round(row.score, 4),
                "overlap_tags": list(row.overlap_tags),
                "overlap_categories": list(row.overlap_categories),
                "top_labels": list(row.top_labels),
                "top_label_categories": list(row.top_label_categories),
            }
            for row in fp_rows[:25]
        ],
        "false_negative_examples": [
            {
                "case": row.case_name,
                "start": round(row.start, 3),
                "end": round(row.end, 3),
                "score": round(row.score, 4),
                "overlap_tags": list(row.overlap_tags),
                "overlap_categories": list(row.overlap_categories),
                "top_labels": list(row.top_labels),
                "top_label_categories": list(row.top_label_categories),
            }
            for row in fn_rows[:25]
        ],
        "unlabeled_high_score_examples": [
            {
                "case": row.case_name,
                "start": round(row.start, 3),
                "end": round(row.end, 3),
                "score": round(row.score, 4),
                "top_labels": list(row.top_labels),
                "top_label_categories": list(row.top_label_categories),
            }
            for row in unlabeled_high_rows[:50]
        ],
    }, cases, windows


def sanitize_review_label(label: str) -> str:
    return label.replace(", ", "/").replace(",", "/")


def summarize_region_top_labels(rows: Sequence[WindowEval], limit: int) -> Tuple[str, ...]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(row.top_labels)
    return tuple(label for label, _count in counts.most_common(limit))


def merge_review_regions(
    rows: Sequence[WindowEval],
    threshold: float,
    merge_gap: float,
    top_label_count: int,
) -> List[ReviewRegion]:
    candidate_rows = sorted(
        (
            row for row in rows
            if row.truth == "unknown" and row.score >= threshold
        ),
        key=lambda row: (row.start, row.end),
    )
    regions: List[ReviewRegion] = []
    current_rows: List[WindowEval] = []
    current_start = 0.0
    current_end = 0.0
    current_peak = 0.0

    def flush_region() -> None:
        nonlocal current_rows, current_start, current_end, current_peak
        if not current_rows:
            return
        regions.append(
            ReviewRegion(
                start=current_start,
                end=current_end,
                peak_score=current_peak,
                window_count=len(current_rows),
                top_labels=summarize_region_top_labels(current_rows, top_label_count),
            )
        )
        current_rows = []
        current_start = 0.0
        current_end = 0.0
        current_peak = 0.0

    for row in candidate_rows:
        if not current_rows:
            current_rows = [row]
            current_start = row.start
            current_end = row.end
            current_peak = row.score
            continue
        if row.start <= current_end + merge_gap:
            current_rows.append(row)
            current_end = max(current_end, row.end)
            current_peak = max(current_peak, row.score)
            continue
        flush_region()
        current_rows = [row]
        current_start = row.start
        current_end = row.end
        current_peak = row.score
    flush_region()
    return regions


def write_review_labels(
    cases: Sequence[TestCase],
    windows: Sequence[WindowEval],
    threshold: float,
    output_name: str,
    merge_gap: float,
    top_label_count: int,
) -> List[Tuple[str, Path, int]]:
    by_case: Dict[str, List[WindowEval]] = {}
    for row in windows:
        by_case.setdefault(row.case_name, []).append(row)

    outputs: List[Tuple[str, Path, int]] = []
    for case in cases:
        regions = merge_review_regions(
            by_case.get(case.name, []),
            threshold=threshold,
            merge_gap=merge_gap,
            top_label_count=top_label_count,
        )
        output_path = case.labels_path.parent / output_name
        lines = []
        for region in regions:
            labels = " | ".join(sanitize_review_label(label) for label in region.top_labels)
            label_text = (
                f"review score={region.peak_score:.3f} "
                f"windows={region.window_count}"
            )
            if labels:
                label_text += f" labels={labels}"
            lines.append(f"{region.start:.6f}\t{region.end:.6f}\t{label_text}")
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        outputs.append((case.name, output_path, len(regions)))
    return outputs


def print_summary(report: dict) -> None:
    best = report["best_metrics"]
    print(f"Cases: {report['case_count']}  Windows: {report['window_count']}")
    print(
        "Best threshold: "
        f"{report['best_threshold']:.2f}  "
        f"precision={best['precision']:.3f}  "
        f"recall={best['recall']:.3f}  "
        f"f1={best['f1']:.3f}  "
        f"ignored={best['ignored_windows']}"
    )
    print("\nPer-case metrics:")
    for item in report["case_metrics"]:
        print(
            f"  {item['case']}: "
            f"precision={item['precision']:.3f} "
            f"recall={item['recall']:.3f} "
            f"f1={item['f1']:.3f} "
            f"(tp={item['tp']} fp={item['fp']} fn={item['fn']})"
        )
    print("\nTop raw labels in true positives:")
    for label, count in report["top_labels_true_positive"]:
        print(f"  {label}: {count}")
    print("\nTop mapped label categories in true positives:")
    for label, count in report["top_label_categories_true_positive"]:
        print(f"  {label}: {count}")
    print("\nTop raw labels in false positives:")
    for label, count in report["top_labels_false_positive"]:
        print(f"  {label}: {count}")
    print("\nTop mapped label categories in false positives:")
    for label, count in report["top_label_categories_false_positive"]:
        print(f"  {label}: {count}")
    print("\nTop annotation categories in false positives:")
    for label, count in report["annotation_categories_false_positive"]:
        print(f"  {label}: {count}")
    print("\nTop raw labels in unlabeled high-score windows:")
    for label, count in report["top_labels_unlabeled_high_score"]:
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ScreamFinder's YAMNet vocalization detector against labeled test folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test-root",
        default="test_files",
        help="Root directory containing test1/test2/... folders with one media file and one Labels*.txt file each",
    )
    parser.add_argument(
        "--yamnet-model",
        default="https://tfhub.dev/google/yamnet/1",
        help="TensorFlow Hub handle or local SavedModel path for YAMNet",
    )
    parser.add_argument(
        "--yamnet-top-k",
        type=int,
        default=12,
        help="How many top raw AudioSet labels to capture per analysis window",
    )
    parser.add_argument(
        "--yamnet-min-window-rms",
        type=float,
        default=0.005,
        help="Absolute RMS backstop for YAMNet windows before scoring",
    )
    parser.add_argument(
        "--yamnet-context-rms-seconds",
        type=float,
        default=12.0,
        help="Amount of preceding audio context used by the adaptive YAMNet energy gate",
    )
    parser.add_argument(
        "--yamnet-context-rms-ratio",
        type=float,
        default=0.0,
        help="Require each YAMNet window to be at least this fraction of the recent context RMS reference",
    )
    parser.add_argument(
        "--initial-threshold",
        type=float,
        default=0.05,
        help="Threshold used while generating the underlying YAMNet debug pass",
    )
    parser.add_argument(
        "--sweep-start",
        type=float,
        default=0.05,
        help="Threshold sweep start",
    )
    parser.add_argument(
        "--sweep-stop",
        type=float,
        default=0.95,
        help="Threshold sweep stop",
    )
    parser.add_argument(
        "--sweep-step",
        type=float,
        default=0.05,
        help="Threshold sweep step",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.2,
        help="Minimum fraction of a YAMNet window that must overlap a positive annotation to count as positive ground truth",
    )
    parser.add_argument(
        "--output-json",
        default="yamnet-tuning-report.json",
        help="Where to write the tuning report JSON",
    )
    parser.add_argument(
        "--write-review-labels",
        action="store_true",
        help="Write Audacity-ready label files for unlabeled high-score windows in each test folder",
    )
    parser.add_argument(
        "--review-label-name",
        default="Review Labels.txt",
        help="Filename to use for generated Audacity review labels inside each test folder",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Threshold for review labels. Defaults to the best threshold found by the sweep",
    )
    parser.add_argument(
        "--review-merge-gap",
        type=float,
        default=0.25,
        help="Merge adjacent review windows separated by at most this many seconds",
    )
    parser.add_argument(
        "--review-top-labels",
        type=int,
        default=4,
        help="How many top raw labels to include in each generated review label",
    )

    args = parser.parse_args()
    report, cases, windows = build_report(
        test_root=Path(args.test_root),
        model_handle=args.yamnet_model,
        top_k=args.yamnet_top_k,
        score_threshold=args.initial_threshold,
        min_window_rms=args.yamnet_min_window_rms,
        context_rms_seconds=args.yamnet_context_rms_seconds,
        context_rms_ratio=args.yamnet_context_rms_ratio,
        min_overlap_ratio=args.min_overlap_ratio,
        sweep_start=args.sweep_start,
        sweep_stop=args.sweep_stop,
        sweep_step=args.sweep_step,
    )
    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_summary(report)
    print(f"\nReport JSON: {output_path.resolve()}")
    if args.write_review_labels:
        review_threshold = (
            float(args.review_threshold)
            if args.review_threshold is not None
            else float(report["best_threshold"])
        )
        outputs = write_review_labels(
            cases=cases,
            windows=windows,
            threshold=review_threshold,
            output_name=args.review_label_name,
            merge_gap=args.review_merge_gap,
            top_label_count=args.review_top_labels,
        )
        print("\nReview label files:")
        for case_name, path, count in outputs:
            print(f"  {case_name}: {count} region(s) -> {path}")


if __name__ == "__main__":
    main()
