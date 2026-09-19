from metrics import ece, k_bucket, risk_coverage_accuracy, spearman


def test_k_bucket():
    assert k_bucket(2) == "2"
    assert k_bucket(4) == "3-5"
    assert k_bucket(10) == "6-15"
    assert k_bucket(77) == ">15"


def test_ece_perfect():
    conf = [0.0, 0.0, 1.0, 1.0]
    corr = [0, 0, 1, 1]
    assert ece(conf, corr, n_bins=10) < 0.05
    # 50% accurate at confidence 0.5 is also calibrated.
    assert ece([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], n_bins=10) < 0.05


def test_risk_coverage():
    corr = [1, 1, 0, 0]
    conf = [0.9, 0.8, 0.2, 0.1]
    assert risk_coverage_accuracy(corr, conf, 0.5) == 1.0


def test_spearman_monotonic():
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) > 0.99
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) < -0.99
