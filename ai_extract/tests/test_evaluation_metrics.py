from django.test import SimpleTestCase
from ai_extract.services.evaluation_metrics_service import ocr_counts, ranking_metrics, field_metrics


class EvaluationMetricsTests(SimpleTestCase):
    def test_ocr_counts_keep_accents_and_ignore_layout_whitespace(self):
        result = ocr_counts('Điện\n an toàn', 'Dien an toàn')
        self.assertEqual(result['word_errors'], 1)
        self.assertGreater(result['character_errors'], 0)
        self.assertEqual(ocr_counts('a\nb', 'a b')['character_errors'], 0)

    def test_rank_deduplicates_hits_and_uses_all_relevant_as_recall_denominator(self):
        result = ranking_metrics(['a', 'a', 'b', 'c'], ['b', 'c'], 2)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['reciprocal_rank'], .5)
        self.assertLess(result['ndcg'], 1)
        self.assertIsNone(ranking_metrics([], [], 2)['recall'])
        with self.assertRaises(ValueError):
            ranking_metrics(['a'], ['a'], 0)

    def test_wrong_value_counts_as_false_positive_and_false_negative(self):
        result = field_metrics({'price': '120', 'penalty': ''}, {'price': '90', 'penalty': '5'})
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 2, 1))
        self.assertEqual(result['f1'], 0)
        self.assertEqual(field_metrics({'price': '120'}, {'price': '120'})['f1'], 1)
