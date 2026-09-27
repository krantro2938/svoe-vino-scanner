import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.merge_wine_catalog import merge_catalog, safe_image


class MergeTests(unittest.TestCase):
    def test_preserves_organizer_identity_and_blocks_conflicting_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official_images = root / 'official-images'
            website = root / 'website'
            official_images.mkdir()
            website.mkdir()
            (official_images/'a.webp').write_bytes(b'reference-a')
            (website/'b.webp').write_bytes(b'reference-b')
            (website/'c.webp').write_bytes(b'reference-a')
            official = root/'official.jsonl'
            snapshot = root/'snapshot.jsonl'
            original = {'slug':'a','name':'Original name','photo_name':'a.webp','index_ready':True}
            official.write_text(json.dumps(original)+'\n')
            snapshot.write_text('\n'.join(json.dumps(r) for r in [
                {'slug':'a','name':'New website name'},
                {'slug':'b','name':'B','image_path':'b.webp'},
                {'slug':'c','name':'C','image_path':'c.webp'},
                {'slug':'missing','name':'Missing','image_path':'absent.webp'},
            ]))
            report = merge_catalog(official,snapshot,official_images,website,root/'output')
            records = {r['slug']:r for r in map(json.loads,(root/'output/catalog.jsonl').read_text().splitlines())}
            self.assertEqual(records['a']['name'],'Original name')
            self.assertEqual(records['a']['website_snapshot']['name'],'New website name')
            self.assertFalse(records['a']['index_ready'])
            self.assertFalse(records['c']['index_ready'])
            self.assertTrue(records['b']['index_ready'])
            self.assertFalse(records['missing']['index_ready'])
            self.assertEqual(report['expanded_records'],4)
            self.assertEqual(report['added_count'],3)
            self.assertEqual(json.loads(official.read_text()),original)
            with self.assertRaises(ValueError):
                merge_catalog(official,snapshot,official_images,website,root/'output')

    def test_alternate_views_share_slug_and_producer_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'original').mkdir()
            (root/'web').mkdir()
            (root/'original/a.webp').write_bytes(b'original')
            (root/'web/new.webp').write_bytes(b'updated-view')
            (root/'official.jsonl').write_text(json.dumps({'slug':'a','name':'A','winery':'Producer',
                'photo_name':'a.webp','index_ready':True}))
            (root/'snapshot.jsonl').write_text(json.dumps({'slug':'a','name':'Website A',
                'image_path':'new.webp','index_ready':True,'source_url':'https://example.org/a'}))
            report=merge_catalog(root/'official.jsonl',root/'snapshot.jsonl',root/'original',root/'web',root/'out')
            refs=list(map(json.loads,(root/'out/training_references.jsonl').read_text().splitlines()))
            self.assertEqual(report['training_references'],2)
            self.assertEqual({r['slug'] for r in refs},{'a'})
            self.assertEqual({r['leakage_group'] for r in refs},{'Producer'})
            self.assertEqual({r['source_kind'] for r in refs},{'organizer','website'})

    def test_shared_upload_path_quarantines_resized_different_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'original').mkdir()
            (root/'web').mkdir()
            (root/'original/a.webp').write_bytes(b'original')
            (root/'web/b.webp').write_bytes(b'resized')
            (root/'official.jsonl').write_text(json.dumps({'slug':'a','name':'A',
                'photo_name':'a.webp','media_path':'archive/uploads/same.webp','index_ready':True}))
            (root/'snapshot.jsonl').write_text(json.dumps({'slug':'b','name':'B',
                'image_path':'b.webp','source_image_path':'/uploads/same.webp','index_ready':True}))
            report=merge_catalog(root/'official.jsonl',root/'snapshot.jsonl',root/'original',root/'web',root/'out')
            self.assertEqual(report['training_references'],0)
            self.assertEqual(report['index_ready_records'],0)
            self.assertEqual(len(report['shared_source_assets']),1)

    def test_strict_website_recovery_preserves_identity_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'original').mkdir()
            (root/'web').mkdir()
            payload = b'verified-new-reference'
            (root/'web/a.webp').write_bytes(payload)
            original = {'slug':'a','name':' Wine A ','winery':'Producer',
                        'photo_name':'absent.webp','index_ready':False,'media_status':'ambiguous',
                        'description':'Official description'}
            website = {'slug':'a','name':'wine a','winery':' producer ', 'image_path':'a.webp',
                       'image_sha256':hashlib.sha256(payload).hexdigest(), 'index_ready':True,
                       'source_image_path':'/uploads/unique.webp','source_url':'https://example.org/a'}
            (root/'official.jsonl').write_text(json.dumps(original))
            (root/'snapshot.jsonl').write_text(json.dumps(website))
            report = merge_catalog(root/'official.jsonl',root/'snapshot.jsonl',root/'original',root/'web',root/'out')
            row = json.loads((root/'out/catalog.jsonl').read_text())
            reference = json.loads((root/'out/training_references.jsonl').read_text())
            self.assertEqual(report['recovered_slugs'], ['a'])
            self.assertEqual(report['recovered_count'], 1)
            self.assertEqual(row['catalog_origin'], 'organizer')
            self.assertEqual(row['reference_origin'], 'website')
            self.assertEqual(row['media_status'], 'ambiguous')
            self.assertEqual(row['name'], original['name'])
            self.assertEqual(row['description'], original['description'])
            self.assertTrue(row['index_ready'])
            self.assertEqual(reference['source_kind'], 'website')
            self.assertEqual(reference['source_url'], website['source_url'])
            self.assertEqual(reference['source_archive_path'], '')
            self.assertEqual(json.loads((root/'official.jsonl').read_text()), original)

    def test_recovery_refuses_metadata_mismatch_missing_hash_and_cross_identity_assets(self):
        for variation in ('name', 'winery', 'missing_hash', 'bytes', 'upload', 'resolved_gate'):
            with self.subTest(variation=variation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'original').mkdir()
                (root/'web').mkdir()
                payload = b'candidate'
                (root/'web/a.webp').write_bytes(payload)
                original = {'slug':'a','name':'Wine, A','winery':'Producer','index_ready':False,
                            'media_status':'unresolved','photo_name':'absent.webp'}
                website = {'slug':'a','name':'Wine, A','winery':'Producer','index_ready':True,
                           'image_path':'a.webp','image_sha256':hashlib.sha256(payload).hexdigest(),
                           'source_image_path':'/uploads/unique.webp'}
                originals = [original]
                if variation == 'name': website['name'] = 'Wine A'
                if variation == 'winery': website['winery'] = 'Other Producer'
                if variation == 'missing_hash': website.pop('image_sha256')
                if variation == 'resolved_gate': original['media_status'] = 'resolved'
                if variation in ('bytes', 'upload'):
                    (root/'original/b.webp').write_bytes(payload if variation == 'bytes' else b'original-size')
                    originals.append({'slug':'b','name':'B','photo_name':'b.webp','index_ready':False,
                                      'media_path':'archive/uploads/unique.webp' if variation == 'upload' else ''})
                (root/'official.jsonl').write_text('\n'.join(map(json.dumps, originals)))
                (root/'snapshot.jsonl').write_text(json.dumps(website))
                report = merge_catalog(root/'official.jsonl',root/'snapshot.jsonl',root/'original',root/'web',root/'out')
                self.assertEqual(report['recovered_count'], 0)
                self.assertEqual(report['training_references'], 0)

    def test_image_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'images').mkdir()
            (root/'secret').write_bytes(b'x')
            self.assertIsNone(safe_image(root/'images','../secret'))

if __name__=='__main__':
    unittest.main()
