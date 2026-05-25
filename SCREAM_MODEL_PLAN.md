# ScreamFinder scream/moan detector plan

## Recommended first model

Wire in **PANNs `Cnn14_DecisionLevelMax`** first.

Why this is the best first implementation:

- It is trained on **AudioSet**, which includes the kinds of labels we care about: `Screaming`, `Yell`, `Shout`, `Groan`, `Wail, moan`, `Crying, sobbing`, and related non-speech vocal events.
- The official repo supports **frame-wise sound event detection**, not just whole-clip tagging.
- It is easier to integrate into this Python script than BEATs or HTS-AT, and more localization-friendly than a simple whole-clip classifier.
- It has a lighter path for inference than a custom training stack, including the separate `panns_inference` package.

## Why not start with the others

- **BEATs** is probably the strongest research option overall, but it is a heavier first integration and less attractive for a quick CPU-first utility.
- **HTS-AT** is very strong for classification and localization, but the repo is a bigger lift than PANNs for this project.
- **YAMNet** is the easiest fallback, but its AudioSet metrics are weaker and it is more of a baseline than a best first choice here.

## What to change in the current code

Current detector logic is centered on band-energy heuristics:

- [screamfinder.py](/Users/ned/src/OEngineer/screamfinder/screamfinder.py:1143)
- [screamfinder.py](/Users/ned/src/OEngineer/screamfinder/screamfinder.py:1269)
- [screamfinder.py](/Users/ned/src/OEngineer/screamfinder/screamfinder.py:1449)

That logic should become a **pluggable detector pipeline**:

1. Keep `extract_audio(...)` as the common decode path.
2. Replace `analyze_vocalizations(...)` with a detector entry point that can dispatch to:
   - `heuristic`
   - `panns`
3. Return segment-level event data, not only two percentages.
4. Derive final table metrics from those segments.

## Proposed CLI shape

Add these options:

- `--detector heuristic|panns`
- `--target-labels screaming,groan,wail_moan,...`
- `--min-event-score 0.35`
- `--window-sec 2.0`
- `--hop-sec 0.5`
- `--merge-gap-sec 0.4`
- `--segment-min-duration 0.35`

Keep `--clip-duration` and parallel worker behavior as-is.

## Scoring model

Score windows using weighted AudioSet labels instead of speech detection.

Suggested positive weights:

- `Screaming`: `1.00`
- `Wail, moan`: `0.95`
- `Groan`: `0.85`
- `Yell`: `0.75`
- `Shout`: `0.70`
- `Crying, sobbing`: `0.55`
- `Gasp`: `0.45`
- `Pant`: `0.35`

Suggested negative weights:

- `Speech`: `-0.35`
- `Conversation`: `-0.40`
- `Narration, monologue`: `-0.40`
- `Music`: `-0.70`
- `Background music`: `-0.80`
- `Inside, small room` and other ambience-only labels: `0.0`

Window score:

```text
score = clamp(sum(label_probability * label_weight), 0.0, 1.0)
```

Segment rule:

1. Mark each analysis window positive when `score >= min_event_score`.
2. Merge adjacent positives when the gap is below `merge_gap_sec`.
3. Drop merged segments shorter than `segment_min_duration`.

## Output metrics to compute

Instead of `female_pct` and `male_pct`, compute:

- `scream_pct`: percentage of analyzed time dominated by high-intensity labels such as `Screaming`, `Yell`, `Shout`
- `moan_pct`: percentage of analyzed time dominated by `Wail, moan`, `Groan`, `Crying, sobbing`
- `vocal_event_pct`: percentage of analyzed time with any positive non-speech vocal event
- `event_count`
- `avg_event_duration`
- `max_event_duration`
- `peak_event_score`

For each detected segment, also compute frequency features from the raw waveform:

- duration
- RMS
- spectral centroid
- spectral rolloff
- dominant frequency
- band energy in configurable ranges

That preserves your original "analyze by frequency range and duration" goal, but only after a better first-stage event detector has isolated likely screams/moans.

## Suggested result structure

Worker output should move from:

```python
{"duration": duration, "female_pct": x, "male_pct": y}
```

to something like:

```python
{
    "duration": duration,
    "scream_pct": 12.4,
    "moan_pct": 28.1,
    "vocal_event_pct": 31.6,
    "event_count": 7,
    "avg_event_duration": 1.24,
    "max_event_duration": 4.88,
    "peak_event_score": 0.93,
    "segments": [
        {
            "start": 12.5,
            "end": 14.1,
            "duration": 1.6,
            "score": 0.82,
            "labels": {"Wail, moan": 0.78, "Groan": 0.31},
            "dominant_freq": 512.0,
            "spectral_centroid": 1380.4,
        }
    ],
}
```

## Implementation order

1. Add a detector abstraction and keep the current heuristic as a fallback mode.
2. Add the PANNs inference path and map AudioSet class names to weights.
3. Change worker output and cache keys to include detector settings.
4. Update HTML columns from `Female %` and `Male %` to `Scream %` and `Moan %`.
5. Add an optional segment-details panel or JSON export for timestamps and frequency summaries.

## References

- PANNs repo: <https://github.com/qiuqiangkong/audioset_tagging_cnn>
- PANNs paper: <https://arxiv.org/abs/1912.10211>
- PANNs inference package: <https://github.com/qiuqiangkong/panns_inference>
- AudioSet ontology: <https://research.google.com/audioset/ontology/>
- YAMNet repo: <https://github.com/tensorflow/models/tree/master/research/audioset/yamnet>
- HTS-AT repo: <https://github.com/RetroCirce/HTS-Audio-Transformer>
- HTS-AT paper: <https://arxiv.org/abs/2202.00874>
- BEATs repo: <https://github.com/microsoft/unilm/tree/master/beats>
- BEATs paper: <https://arxiv.org/abs/2212.09058>
