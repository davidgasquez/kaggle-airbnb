from kairbnb.metrics import dcg_score
from kairbnb.metrics import ndcg_score


class TestMetrics(object):

    def test_dcg_score(self):
        y_true = [0, 1, 0]
        y_score = [0.2, 0.6, 0.2]
        assert dcg_score(y_true, y_score) == 1.0

    def test_ndcg_score(self):
        ground_truth = [1, 0, 2]
        predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
        score = ndcg_score(ground_truth, predictions)
        assert score == 1.0
