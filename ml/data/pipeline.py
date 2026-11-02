"""Phase 1 dataset validation, deduplication, and reproducible splits."""
from collections import defaultdict
import random
import unicodedata

FIELDS = ('arxiv_id', 'title', 'abstract', 'primary_cat', 'all_cats', 'label')


def validate_schema(features):
    missing = set(FIELDS) - set(features)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    names = getattr(features['label'], 'names', None)
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected a unique, nonempty ClassLabel mapping')
    return list(names)


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def problems(row, labels):
    if not isinstance(row, dict):
        return ['malformed_record']
    reasons = []
    for field in FIELDS[:-1]:
        value = row.get(field)
        if not isinstance(value, str):
            reasons.append(f'{field}_missing_or_invalid')
        elif not value.strip():
            reasons.append(f'{field}_empty')
    label = row.get('label')
    if type(label) is not int or not 0 <= label < len(labels):
        reasons.append('invalid_label')
    return reasons


def clean_records(rows, labels):
    if not rows:
        raise ValueError('Dataset is empty')
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    seen_id, seen_text = {}, {}
    duplicate_id = duplicate_text = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        identifier = row.get('arxiv_id')
        if isinstance(identifier, str) and identifier.strip():
            key = normalized(identifier)
            if key in seen_id:
                duplicate_id += 1
                union(i, seen_id[key])
            else:
                seen_id[key] = i
        title, abstract = row.get('title'), row.get('abstract')
        if all(isinstance(v, str) and v.strip() for v in (title, abstract)):
            key = (normalized(title), normalized(abstract))
            if key in seen_text:
                duplicate_text += 1
                union(i, seen_text[key])
            else:
                seen_text[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    excluded, selected = {}, []
    for group in groups.values():
        valid = []
        for i in group:
            reasons = problems(rows[i], labels)
            if reasons:
                excluded[i] = reasons
            else:
                valid.append(i)
        if len({rows[i]['label'] for i in valid}) > 1:
            for i in valid:
                excluded[i] = ['duplicate_label_conflict']
        elif valid:
            selected.append(valid[0])
            for i in valid[1:]:
                excluded[i] = ['duplicate']
    return sorted(selected), excluded, {'duplicate_id_rows': duplicate_id, 'duplicate_text_rows': duplicate_text, 'duplicate_groups': sum(len(g) > 1 for g in groups.values())}


def make_splits(rows, indices, labels, seed=42):
    by_label = defaultdict(list)
    for i in indices:
        by_label[rows[i]['label']].append(i)
    if not indices or any(len(by_label[i]) < 3 for i in range(len(labels))):
        raise ValueError('Each class requires at least three records for stratified splits')
    rng = random.Random(seed)
    result = {'train': [], 'validation': [], 'test': []}
    for label in range(len(labels)):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * .1))
        n_val = max(1, round(n * .1))
        result['test'].extend(group[:n_test])
        result['validation'].extend(group[n_test:n_test+n_val])
        result['train'].extend(group[n_test+n_val:])
    for values in result.values():
        values.sort()
    return result
