# Automatic FAQ Generation from University Examination Regulations

This repository contains the code and experiments developed as part of a Bachelor thesis.
The goal is to automatically generate high-quality FAQ question–answer pairs from university
examination regulations using local large language models.

## Pipeline Overview
1. PDF parsing and text extraction
2. Text cleaning and chunking
3. Question–answer generation using instruction-tuned LLMs
4. Dataset filtering and deduplication
5. Automatic and human-centered evaluation

## Models
- FLAN-T5 (baseline)
- Planned: Qwen 2.5 / LLaMA-based models

## Evaluation
- Exact Match Accuracy
- Token-level F1 Score
- Accuracy Success
- Textual Entailment

## Gold Dataset
A small manually curated gold dataset is used for evaluation and comparison.
