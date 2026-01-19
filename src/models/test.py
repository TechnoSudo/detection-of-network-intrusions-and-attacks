import numpy as np

from supervised.random_forest import RFModel
from supervised.svm import SVMModel
from supervised.xgboost import XGBModel
from supervised.ensemble import WeightedSoftVotingEnsemble

from anomaly_detection.isolation_forest import IsolationForestDetector
from anomaly_detection.one_class_svm import OneClassSVMDetector

from drift_detection.adwin import ADWINDriftDetector, ADWINConfig
from drift_detection.page_hinkley import PageHinkleyDriftDetector, PageHinkleyConfig

from meta_decision_layer import MetaDecisionLayer, MetaDecisionConfig


def run_sanity_check():
    np.random.seed(42)

    # Fake dataset
    X = np.random.randn(200, 20)
    y = np.random.choice(["normal", "attack"], size=200)

    # Split
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    # Train base models
    rf = RFModel().fit(X_train, y_train)
    svm = SVMModel().fit(X_train, y_train)
    xgb = XGBModel().fit(X_train, y_train)

    ensemble = WeightedSoftVotingEnsemble(
        models={
            "rf": rf,
            "svm": svm,
            "xgb": xgb
        }
    )
    ensemble.fit(X_train, y_train)

    # Train anomaly detectors
    normal_idx = y_train == "normal"
    iforest = IsolationForestDetector().fit(X_train[normal_idx])
    ocsvm = OneClassSVMDetector().fit(X_train[normal_idx])

    # Meta Decision Layer
    mdl = MetaDecisionLayer(
        supervised_model=ensemble,
        isolation_forest=iforest,
        one_class_svm=ocsvm,
        config=MetaDecisionConfig(confidence_threshold=0.6)
    )

    output = mdl.predict(X_test)

    print("Predictions:", output["label"][:10])
    print("Confidence:", output["confidence"][:10])
    print("Anomaly flags:", output["anomaly"][:10])

    print(" Sanity check passed!")

def test_adwin_drift():
    np.random.seed(42)

    det = ADWINDriftDetector(ADWINConfig(delta=0.002))

    # Phase 1: low error (stable)
    errors_1 = (np.random.rand(800) < 0.05).astype(float)  # ~5% error

    # Phase 2: higher error (drift)
    errors_2 = (np.random.rand(800) < 0.25).astype(float)  # ~25% error

    stream = np.concatenate([errors_1, errors_2])

    drift_points = []
    for i, e in enumerate(stream, start=1):
        if det.update(e):
            drift_points.append(i)

    print("ADWIN drift points:", drift_points[:10], "..." if len(drift_points) > 10 else "")
    print("Total drift signals:", len(drift_points))

def test_page_hinkley_drift():
    np.random.seed(42)

    det = PageHinkleyDriftDetector(PageHinkleyConfig(
        min_instances=30,
        delta=0.005,
        threshold=50.0,
        alpha=1.0
    ))

    # Phase 1: low error
    errors_1 = (np.random.rand(800) < 0.05).astype(float)

    # Phase 2: higher error (drift)
    errors_2 = (np.random.rand(800) < 0.25).astype(float)

    stream = np.concatenate([errors_1, errors_2])

    drift_points = []
    for i, e in enumerate(stream, start=1):
        if det.update(e):
            drift_points.append(i)

    print("Page-Hinkley drift points:", drift_points[:10], "..." if len(drift_points) > 10 else "")
    print("Total drift signals:", len(drift_points))


if __name__ == "__main__":
    # run_sanity_check()
    # test_adwin_drift()
    test_page_hinkley_drift()