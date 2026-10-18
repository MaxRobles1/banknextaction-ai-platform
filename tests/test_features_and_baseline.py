"""Feature transformation (PDF §5.1.2 'una transformación'), baseline contract and threshold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from banknextaction.config import Config
from banknextaction.data.load import load_dataset
from banknextaction.evaluation.metrics import choose_threshold_f1, evaluate, f1_by_threshold
from banknextaction.features.preprocess import build_preprocessor, contact_history, signed_log1p
from banknextaction.models import available_models, create_model, model_class
from tests.conftest import make_raw_frame


@pytest.fixture
def xy(tmp_path: Path, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    path = tmp_path / "raw.csv"
    make_raw_frame(n_rows=600, seed=1).to_csv(path, sep=";", index=False)
    return load_dataset(path, cfg.data, cfg.features)


def test_contact_history_decodes_pdays_sentinel() -> None:
    out = contact_history(np.array([-1, 0, 9]))

    np.testing.assert_array_equal(out[:, 0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 0.0, np.log1p(9)])


def test_signed_log1p_keeps_sign_and_compresses_tails() -> None:
    out = signed_log1p(np.array([-100.0, 0.0, 100.0]))

    assert out[0] == pytest.approx(-out[2])
    assert out[1] == 0.0
    assert out[2] < 100.0


def test_preprocessor_output_is_numeric_finite_and_fit_on_train_only(
    xy: tuple[pd.DataFrame, pd.Series], cfg: Config
) -> None:
    features, _ = xy
    train, other = features.iloc[:400], features.iloc[400:]
    pre = build_preprocessor(cfg.features).fit(train)

    out = pre.transform(other)

    assert out.shape[0] == len(other)
    assert out.shape[1] == len(pre.get_feature_names_out())
    assert np.isfinite(out).all()
    scaler = pre.named_transformers_["num"]
    np.testing.assert_allclose(scaler.mean_, [train["age"].mean()])


def test_preprocessor_ignores_unseen_categories(
    xy: tuple[pd.DataFrame, pd.Series], cfg: Config
) -> None:
    features, _ = xy
    pre = build_preprocessor(cfg.features).fit(features)
    unseen = features.iloc[:2].copy()
    unseen["job"] = unseen["job"].cat.add_categories(["astronaut"])
    unseen.loc[unseen.index[0], "job"] = "astronaut"

    assert np.isfinite(pre.transform(unseen)).all()


def test_logistic_baseline_output_shape_and_reload(
    xy: tuple[pd.DataFrame, pd.Series], cfg: Config, tmp_path: Path
) -> None:
    features, target = xy
    model = create_model("logreg", cfg).fit(features.iloc[:500], target.iloc[:500])

    proba = model.predict_proba(features.iloc[500:])
    model.save(tmp_path)
    reloaded = model_class("logreg").load(tmp_path)

    assert proba.shape == (100,)
    assert ((proba >= 0) & (proba <= 1)).all()
    np.testing.assert_allclose(reloaded.predict_proba(features.iloc[500:]), proba)
    assert set(model.search.best_params) == set(cfg.baseline.param_grid)
    assert "logreg" in available_models()


def test_threshold_tie_break_keeps_highest_threshold() -> None:
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])

    choice = choose_threshold_f1(y, proba, grid_size=99)

    assert choice.f1 == pytest.approx(1.0)
    assert choice.threshold == pytest.approx(0.8)


def test_f1_vectorized_matches_evaluate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = rng.random(300)

    f1 = f1_by_threshold(y, proba, np.array([0.5]))[0]

    assert f1 == pytest.approx(evaluate("x", y, proba, 0.5).f1)
