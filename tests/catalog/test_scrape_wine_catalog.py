import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts' / 'scrape_wine_catalog.py'
spec = importlib.util.spec_from_file_location('scrape_wine_catalog', SCRIPT)
scraper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scraper)


class ListingTests(unittest.TestCase):
    def test_decodes_nuxt_references_without_confusing_scalar_numbers(self):
        values = [{'currentPage': 1, 'totalItems': 2, 'totalPages': 1, 'items': 3},
                  1, 2069, [4], {'slug': 5, 'title': 6}, 'wine', 'Вино']
        html = '<script id="__NUXT_DATA__" type="application/json">' + json.dumps(values) + '</script>'
        result = scraper.parse_listing(html)
        self.assertEqual(result['totalItems'], 2069)
        self.assertEqual(result['items'], [{'slug': 'wine', 'title': 'Вино'}])

    def test_listing_is_not_misrepresented_as_original_image_or_full_metadata(self):
        row = scraper.normalize({'slug': 'wine', 'title': 'Вино', 'image': {'url': '/uploads/wine.webp'}}, 2, 'today')
        self.assertEqual(row['image_url'], 'https://api.vino-svoe.ru/v1/img/str-api/910/910/resize/uploads/wine.webp')
        self.assertFalse(row['index_ready'])
        self.assertEqual(row['metadata_scope'], 'listing')
        self.assertEqual(row['source_page'], 2)

    def test_rejects_path_traversal_slug(self):
        with self.assertRaises(ValueError):
            scraper.normalize({'slug': '../wine'}, 1, 'today')

    def test_rejects_unexpected_html(self):
        with self.assertRaises(ValueError):
            scraper.parse_listing('<html>blocked</html>')

    def test_rejects_non_image_riff_and_cached_html(self):
        for payload in (b'RIFF1234WAVE', b'<html>error</html>'):
            with self.assertRaises(ValueError):
                scraper.validate_image_signature(payload)
        scraper.validate_image_signature(b'RIFF1234WEBP')
