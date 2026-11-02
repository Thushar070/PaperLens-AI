"""Generate Phase 1 data artifacts from the Hugging Face dataset."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from datasets import load_dataset
from ml.data.pipeline import FIELDS, clean_records, make_splits, problems, validate_schema

SOURCE = 'vrajpat3ll/arxiv-cs-papers'
REVISION = 'c5523dff308d7d87c2f44f33ac3b898d0514a281'


def load_source(loader=load_dataset):
    try:
        return loader(SOURCE, revision=REVISION)
    except Exception as exc:
        raise RuntimeError(f'Unable to load {SOURCE} at {REVISION}: {exc}') from exc


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    def percentile(p):
        return values[round((len(values)-1)*p)]
    return {'min': values[0], 'median': statistics.median(values), 'mean': round(statistics.mean(values), 2), 'p90': percentile(.9), 'p99': percentile(.99), 'max': values[-1]}


def process(dataset, output, seed=42, plots=True):
    if set(dataset) != {'train'}:
        raise ValueError(f'Unexpected source splits: {list(dataset)}')
    source = dataset['train']
    labels = validate_schema(source.features)
    rows = list(source)
    selected, excluded, duplicates = clean_records(rows, labels)
    splits = make_splits(rows, selected, labels, seed)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    counts = lambda subset: {name: sum(isinstance(rows[i], dict) and type(rows[i].get('label')) is int and rows[i]['label'] == j for i in subset) for j, name in enumerate(labels)}
    lengths = {}
    for field in ('title', 'abstract'):
        texts = [r[field] for r in rows if isinstance(r, dict) and isinstance(r.get(field), str)]
        lengths[field] = {'characters': stats([len(t) for t in texts]), 'words': stats([len(t.split()) for t in texts])}
    issues = Counter(reason for row in rows for reason in problems(row, labels))
    reasons = Counter(reason for reasons in excluded.values() for reason in reasons)
    samples = {name: next((rows[i] for i in selected if rows[i]['label'] == j), None) for j, name in enumerate(labels)}
    metadata = {'source': SOURCE, 'revision': REVISION, 'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'license': 'cc-by-4.0', 'source_splits': {'train': len(rows)}, 'features': {k: str(v) for k, v in source.features.items()}, 'label_mapping': dict(enumerate(labels)), 'seed': seed, 'split_method': 'per-class deterministic shuffle, approximately 80/10/10', 'total_records': len(rows), 'valid_before_deduplication': len(rows)-sum(bool(problems(r, labels)) for r in rows), 'selected_records': len(selected), 'excluded_records': len(excluded), 'validation_issues': dict(issues), 'exclusion_reasons': dict(reasons), 'duplicates': duplicates, 'source_class_distribution': counts(range(len(rows))), 'selected_class_distribution': counts(selected), 'split_sizes': {k: len(v) for k,v in splits.items()}, 'split_class_distribution': {k: counts(v) for k,v in splits.items()}, 'length_statistics': lengths, 'samples': samples}
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    (output/'exclusions.json').write_text(json.dumps([{'source_index': i, 'arxiv_id': rows[i].get('arxiv_id') if isinstance(rows[i], dict) else None, 'reasons': why} for i, why in sorted(excluded.items())], indent=2), encoding='utf-8')
    for name, indices in splits.items():
        with (output/f'{name}.jsonl').open('w', encoding='utf-8') as f:
            for i in indices:
                f.write(json.dumps({'source_index': i, **{k: rows[i][k] for k in FIELDS}}, ensure_ascii=False) + '\n')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].bar(labels, list(counts(selected).values()))
        axes[0].tick_params(axis='x', rotation=30)
        axes[0].set_title('Selected records by class')
        for ax, field in zip(axes[1:], ('title', 'abstract')):
            ax.hist([len(rows[i][field].split()) for i in selected], bins=40)
            ax.set_title(f'{field.title()} word counts')
            ax.set_xlabel('Words')
        fig.tight_layout()
        fig.savefig(output/'eda.png', dpi=150)
        plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/data/phase1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata = process(load_source(), args.output, args.seed)
    print(json.dumps({k: metadata[k] for k in ('total_records','selected_records','excluded_records','validation_issues','exclusion_reasons','duplicates','selected_class_distribution','split_sizes','split_class_distribution','length_statistics')}, indent=2))


if __name__ == '__main__':
    main()
