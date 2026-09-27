import unittest
from evaluation.build_confusion_clusters import make_clusters, without_year


def row(slug, name, winery='Producer', **extra):
    return dict(slug=slug, name=name, winery=winery, index_ready=True, **extra)


class ConfusionTests(unittest.TestCase):
    def test_deterministic_vintage_neighbors_preserve_exact_labels(self):
        rows = [row('wine-2021','Wine Reserve 2021'), row('wine-2022','Wine Reserve 2022')]
        first = make_clusters(rows)
        self.assertEqual(first, make_clusters(list(reversed(rows))))
        self.assertEqual(first[2]['covered_classes'], 2)
        self.assertEqual(first[1][0]['slug'], 'wine-2021')
        self.assertEqual(first[1][0]['neighbors'][0]['slug'], 'wine-2022')
        self.assertIn('same_name_without_vintage',first[1][0]['neighbors'][0]['reasons'])

    def test_excludes_cross_winery_and_blocked_labels(self):
        rows = [row('a','Reserve Wine 2021'), row('b','Reserve Wine 2022','Other'),
                dict(row('blocked','Reserve Wine 2023'), index_ready=False)]
        clusters, neighbors, report = make_clusters(rows)
        self.assertEqual(clusters, [])
        self.assertEqual(report['eligible_classes'], 2)
        self.assertTrue(all(not r['neighbors'] for r in neighbors))

    def test_series_uses_only_supplied_grape_tokens(self):
        rows = [row('a','Northern Valley Chardonnay',grapes='Chardonnay', category='White'),
                row('b','Northern Valley Riesling',grapes='Riesling', category='White')]
        clusters, _, _ = make_clusters(rows)
        self.assertIn('same_series_tokens', {c['reason'] for c in clusters})
        self.assertEqual(rows[0]['grapes'], 'Chardonnay')

    def test_bounded_neighbors_and_oversize_groups(self):
        rows = [row(f'wine-{i}', 'Reserve', grapes='Grape', category='White') for i in range(8)]
        _, neighbors, report = make_clusters(rows, max_neighbors=2)
        self.assertTrue(all(len(r['neighbors']) == 2 for r in neighbors))
        self.assertEqual(report['directed_neighbor_pairs'],16)
        _, _, small = make_clusters(rows, max_group=3)
        self.assertEqual(small['covered_classes'],0)
        self.assertTrue(small['oversized_groups_excluded'])

    def test_only_explicit_four_digit_vintages_are_removed(self):
        self.assertEqual(without_year('Reserve 2019 24 9'), ('reserve','24','9'))

    def test_duplicate_labels_rejected(self):
        with self.assertRaises(ValueError):
            make_clusters([row('a','A'), row('a','B')])
