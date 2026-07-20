# TattooAudit

Artifact repository for the paper:

**TattooAudit: A Benchmark-Grounded Interactive Expert-Review System for Open-Set Tattoo Analysis**

## Overview

TattooAudit combines a controlled Vision–Language Model benchmark with an interactive expert-review workflow for open-set tattoo analysis.

The repository contains the experimental pipeline, normalized model outputs, evaluation tables, case-level audit-priority indicators, and the TattooAudit Streamlit application.

The system supports two complementary research goals:

1. evaluating the semantic naming behavior of general-purpose Vision–Language Models under original-image and reduced-context conditions; and
2. transforming benchmark discrepancies into a transparent, structured, and traceable expert-review workflow.

TattooAudit does not treat model predictions as final authority. The benchmark reference annotation remains authoritative, while model outputs provide supporting evidence for expert inspection.

## Dataset

The study uses the publicly available test sets of the TSSD2023 benchmark:

https://github.com/Brilhador/tssd2023

TSSD2023 provides original tattoo images, pixel-level reference masks, and semantic labels for closed-set and open-set evaluation.

This repository should contain only data and derived artifacts that can be publicly redistributed. The non-public TSSD2023 training and validation images are not included.

## Experimental scope

The study evaluates zero-shot multi-label semantic naming under oracle-segmentation conditions.

The reference masks are treated as oracle segmentations, allowing the experiment to isolate semantic naming behavior without conflating it with mask-prediction errors.

Each case is evaluated under three visual conditions:

- **Original image:** the unmodified full image;
- **GT-crop black:** a class-specific crop derived from the reference mask, rendered on a black background;
- **GT-crop white:** the same class-specific foreground rendered on a white background.

For images containing multiple reference classes, the pipeline processes each class-specific crop independently and aggregates the normalized crop-level predictions at the image level.

## Evaluated models

- Gemma3:12B
- Qwen2.5-VL-7B
- LLaMA 3.2 Vision 11B

The models operate under a controlled 35-label vocabulary that excludes background and includes the `unknown` label.

## Prediction normalization

Model outputs are normalized into sets of valid vocabulary labels.

For each image:

- `GT` is the set of reference labels;
- `Pred` is the normalized set of predicted labels.

The evaluation computes:

```text
TP = |GT ∩ Pred|
FP = |Pred \ GT|
FN = |GT \ Pred|
