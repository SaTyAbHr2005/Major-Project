from hospital_client.training.metrics import compute_classification_metrics


def test_perfect_predictions_score_100_percent():
    class_mapping = {"cat": 0, "dog": 1}
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, loss=0.1, class_mapping=class_mapping)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]


def test_per_class_metrics_reflect_class_specific_behavior():
    class_mapping = {"normal": 0, "disease": 1}
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 0]  # model never predicts the minority "disease" class
    metrics = compute_classification_metrics(y_true, y_pred, loss=0.5, class_mapping=class_mapping)
    assert metrics.per_class["disease"]["recall"] == 0.0
    assert metrics.per_class["normal"]["recall"] == 1.0
    # Overall accuracy looks fine (4/6) despite total failure on the minority class -
    # this is exactly why per-class metrics matter for imbalanced medical data.
    assert metrics.accuracy > 0.5
    assert metrics.per_class["disease"]["f1"] == 0.0


def test_class_order_matches_class_mapping_indices():
    class_mapping = {"z_class": 0, "a_class": 1}
    metrics = compute_classification_metrics([0, 1], [0, 1], 0.0, class_mapping)
    assert metrics.class_order == ["z_class", "a_class"]


def test_confusion_matrix_shape_matches_num_classes():
    class_mapping = {"a": 0, "b": 1, "c": 2}
    metrics = compute_classification_metrics([0, 1, 2, 0], [0, 1, 1, 2], 0.0, class_mapping)
    assert len(metrics.confusion_matrix) == 3
    assert all(len(row) == 3 for row in metrics.confusion_matrix)


def test_to_dict_contains_all_expected_fields():
    metrics = compute_classification_metrics([0, 1], [0, 0], 0.2, {"a": 0, "b": 1})
    d = metrics.to_dict()
    for key in ("loss", "accuracy", "precision", "recall", "f1", "per_class", "confusion_matrix", "sample_count"):
        assert key in d
