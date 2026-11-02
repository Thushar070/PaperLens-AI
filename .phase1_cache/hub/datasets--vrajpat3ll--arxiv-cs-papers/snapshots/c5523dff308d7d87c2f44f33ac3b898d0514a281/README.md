---
language:
  - en
license: cc-by-4.0
task_categories:
  - text-classification
task_ids:
  - multi-class-classification
tags:
  - arxiv
  - scientific-abstracts
  - computer-science
pretty_name: ArXiv CS Abstracts
size_categories:
  - 10K < n < 100K
---

# ArXiv CS Abstracts

A corpus of CS paper abstracts from the [arXiv API](https://arxiv.org/help/api/index),
organised into four thematic labels.

Collected for NLP Lab (CS-5-45(MO)), IIT Jammu, Semester 2, 2025-2026.

## Label Schema

The `label` column contains numeric indices corresponding to the following categories:

| (Index) Label     | arXiv Categories           | Theme                           |
| ----------------- | -------------------------- | ------------------------------- |
| (0) Algorithms    | cs.DS, cs.CC, cs.GT, cs.CG | Theory & foundations            |
| (1) NLP/AI        | cs.CL, cs.AI, cs.LG, cs.IR | Language, reasoning, & learning |
| (2) Systems/Infra | cs.NI, cs.DC, cs.OS, cs.SY | Networks, systems, & infra      |
| (3) Vision        | cs.CV, cs.GR, cs.MM        | Visual computing & multimedia   |

Label assignment is based on the **primary category** (the first arXiv tag). Cross-listed papers are not duplicated.

## Fields

| Column        | Description                     |
| ------------- | ------------------------------- |
| `arxiv_id`    | Unique arXiv paper ID           |
| `title`       | Paper title                     |
| `abstract`    | Raw abstract (no preprocessing) |
| `primary_cat` | Primary arXiv category          |
| `all_cats`    | All categories (pipe-separated) |
| `label`       | Thematic label (numeric index)  |

## Collection details

- Source: [arxiv.py](https://github.com/lukasschwab/arxiv.py)
- Sort: most recently submitted first
- Deduplication: by `arxiv_id`
- Class balancing: random undersample, seed = 42
- Collection date: 2026-03-12
