import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from datasets import ClassLabel, Features, Value
from ml.data.pipeline import clean_records, make_splits, normalized, problems, validate_schema
from ml.data.exploration import load_source, process

FEATURES = Features({'arxiv_id': Value('string'), 'title': Value('string'), 'abstract': Value('string'), 'primary_cat': Value('string'), 'all_cats': Value('string'), 'label': ClassLabel(names=['a', 'b'])})

def fixture():
    return [dict(arxiv_id=str(i), title=f'title {i}', abstract=f'abstract {i}', primary_cat='cs.AI', all_cats='cs.AI', label=i % 2) for i in range(20)]

class Fixture(list):
    features = FEATURES

class DataTests(unittest.TestCase):
    def test_schema_and_missing_columns(self):
        self.assertEqual(validate_schema(FEATURES), ['a', 'b'])
        for field in ('title', 'abstract', 'label'):
            features = dict(FEATURES)
            del features[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_schema(features)
        with self.assertRaises(ValueError):
            validate_schema({**FEATURES, 'label': Value('int64')})

    def test_validation(self):
        row = fixture()[0]
        for field in ('title', 'abstract'):
            self.assertIn(field+'_empty', problems({**row, field: '  '}, ['a', 'b']))
            self.assertIn(field+'_missing_or_invalid', problems({**row, field: None}, ['a', 'b']))
        self.assertIn('invalid_label', problems({**row, 'label': 7}, ['a', 'b']))
        self.assertIn('invalid_label', problems({**row, 'label': True}, ['a', 'b']))
        self.assertEqual(problems(None, ['a', 'b']), ['malformed_record'])
        with self.assertRaises(ValueError):
            clean_records([], ['a', 'b'])

    def test_duplicates_transitive_and_conflicting_labels(self):
        rows = fixture()[:4]
        rows[1] = {**rows[0], 'arxiv_id': 'other', 'title': ' TITLE  0 '}
        rows[2] = {**rows[1], 'title': 'new', 'abstract': 'new'}
        rows[3] = {**rows[2], 'arxiv_id': 'third', 'label': 1}
        rows[3]['title'] = ' NEW '
        indices, excluded, info = clean_records(rows, ['a', 'b'])
        self.assertEqual(indices, [])
        self.assertEqual(len(excluded), 4)
        self.assertEqual(info['duplicate_groups'], 1)
        self.assertEqual(normalized('  HELLO  World '), 'hello world')

    def test_clean_and_split(self):
        rows = fixture()
        original = copy.deepcopy(rows)
        rows.append(copy.deepcopy(rows[0]))
        rows.append({**rows[1], 'title': ''})
        selected, excluded, _ = clean_records(rows, ['a', 'b'])
        self.assertEqual(excluded[20], ['duplicate'])
        self.assertEqual(excluded[21], ['title_empty'])
        splits = make_splits(rows, selected, ['a', 'b'])
        self.assertEqual(splits, make_splits(rows, selected, ['a', 'b']))
        self.assertEqual(set().union(*map(set, splits.values())), set(selected))
        self.assertEqual(sum(map(len, splits.values())), len(selected))
        self.assertEqual(rows[:20], original)
        with self.assertRaises(ValueError):
            make_splits(rows, [0, 1], ['a', 'b'])

    def test_load_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'network down'):
            load_source(Mock(side_effect=ConnectionError('network down')))

    def test_integration(self):
        rows = fixture()
        source = Fixture(rows)
        ds = {'train': source}
        loader = Mock(return_value=ds)
        self.assertIs(load_source(loader), ds)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'output'
            meta = process(ds, out, plots=False)
            self.assertEqual(meta['selected_records'], 20)
            self.assertEqual(json.loads((out/'metadata.json').read_text(encoding='utf-8'))['split_sizes'], meta['split_sizes'])
            seen_ids, seen_text = set(), set()
            for split in ('train','validation','test'):
                for line in (out/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    self.assertFalse(problems(row, ['a','b']))
                    self.assertNotIn(row['arxiv_id'], seen_ids)
                    key = (normalized(row['title']), normalized(row['abstract']))
                    self.assertNotIn(key, seen_text)
                    seen_ids.add(row['arxiv_id']); seen_text.add(key)
            self.assertEqual(list(source), rows)
            with self.assertRaises(FileExistsError):
                process(ds, out, plots=False)

if __name__ == '__main__':
    unittest.main()
