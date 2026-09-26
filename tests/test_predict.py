"""Tests de PantaraRegressor.fit / predict (pantara/core.py)."""

import contextlib
import io
import pickle
import random

import numpy as np
import pytest
import torch

import pantara
from pantara import core


@pytest.fixture(scope='module')
def oracle():
    model = core.OracleClassifier(n_classes=core.N_CLASSES)
    model.load_state_dict(torch.load(pantara.MODEL_PATH, weights_only=True,
                                     map_location=core.DEVICE))
    model.to(core.DEVICE)
    model.eval()
    return model


def make(law, N, rng):
    """Jeux de données dont la famille de bloc attendue est connue."""
    noise = lambda y: y*(1 + rng.normal(0, 0.01, len(y)))
    if law == 'power_law':
        X = np.column_stack([rng.uniform(1, 8, N), rng.uniform(1, 6, N)])
        y = 3.0*X[:, 0]*X[:, 1]**-2
    elif law == 'lin[original]':
        X = rng.uniform(0.5, 10, (N, 1))
        y = noise(1.5*X[:, 0] + 8)
    elif law == 'lin[log_y]':
        X = rng.uniform(0.1, 5, (N, 1))
        y = noise(4.0*np.exp(-0.6*X[:, 0]))
    elif law == 'sin':
        X = rng.uniform(0, 4*np.pi/1.5, (N, 1))
        y = 2.0*np.sin(1.5*X[:, 0] + 0.5)
    elif law == 'qc':
        X = np.column_stack([rng.uniform(1, 5, N), rng.uniform(1, 8, N)])
        y = noise(0.5*X[:, 0]*X[:, 1]**2 + 10)
    return X, y


LAWS = ['power_law', 'lin[original]', 'lin[log_y]', 'sin', 'qc']


def run_pipeline(oracle, X, y, seed):
    # verbose=True : dans pipeline(), les arrêts « bruit pur » et « score <
    # MIN_SCORE » ne s'exécutent qu'en mode verbeux (break sur la ligne du
    # `if verbose:`). fit() applique ces arrêts dans tous les cas.
    np.random.seed(seed)
    torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        return core.pipeline(oracle, X, y, verbose=True)


@pytest.mark.parametrize('law', LAWS)
def test_fit_matches_pipeline(oracle, law):
    X, y = make(law, 400, np.random.default_rng(0))
    prec, chosen = run_pipeline(oracle, X, y, seed=42)
    est = pantara.PantaraRegressor(model=oracle, random_state=42).fit(X, y)

    assert est.chosen_ == chosen
    assert est.precision_ == prec
    assert est.chosen_[0].startswith(law.split('[')[0])
    if '[' in law:
        assert est.chosen_[0] == law
    # predict sur les X d'entraînement = prédiction en échantillon du pipeline
    np.testing.assert_allclose(est.predict(X), est.train_pred_, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('law', LAWS)
def test_generalizes_to_test_set(oracle, law):
    rng = np.random.default_rng(1)
    X_train, y_train = make(law, 400, rng)
    X_test, y_test = make(law, 200, rng)
    est = pantara.PantaraRegressor(model=oracle).fit(X_train, y_train)

    y_pred = est.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(np.isfinite(y_pred))
    assert est.score(X_test, y_test) > 0.99


@pytest.mark.parametrize('law', core.ALL_LAWS)
def test_replay_all_generator_laws(oracle, law):
    """Toutes les lois du générateur : mêmes choix que pipeline(), replay fidèle."""
    random.seed(3)
    np.random.seed(3)
    X, y = core.gen_law(law, N=300)
    prec, chosen = run_pipeline(oracle, X, y, seed=7)
    est = pantara.PantaraRegressor(model=oracle, random_state=7).fit(X, y)

    assert est.chosen_ == chosen
    assert est.precision_ == prec
    np.testing.assert_allclose(est.predict(X), est.train_pred_,
                               rtol=1e-4, atol=1e-6*np.abs(y).max())


def test_predict_before_fit_raises():
    with pytest.raises(ValueError, match=r"fit\(X, y\)"):
        pantara.PantaraRegressor().predict(np.ones((5, 2)))


def test_predict_wrong_feature_count_raises(oracle):
    X, y = make('power_law', 200, np.random.default_rng(0))
    est = pantara.PantaraRegressor(model=oracle).fit(X, y, feature_names=['a', 'b'])
    with pytest.raises(ValueError, match='2'):
        est.predict(np.ones((5, 3)))


def test_log_block_does_not_return_zeros(oracle):
    """Régression du bug SRBench : y factice à zéro → bloc log_y prédit 0."""
    rng = np.random.default_rng(2)
    X, y = make('lin[log_y]', 300, rng)
    est = pantara.PantaraRegressor(model=oracle).fit(X, y)
    assert est.blocks_[0]['space'] == 'log_y'
    X_new = np.array([[0.5], [1.0], [2.0]])
    np.testing.assert_allclose(est.predict(X_new), 4.0*np.exp(-0.6*X_new[:, 0]), rtol=0.02)


def test_out_of_domain_warns(oracle):
    X, y = make('power_law', 200, np.random.default_rng(0))
    est = pantara.PantaraRegressor(model=oracle).fit(X, y)
    with pytest.warns(RuntimeWarning, match='non finie'):
        pred = est.predict(np.array([[2.0, 1.0], [-1.0, 1.0]]))
    assert np.isfinite(pred[0]) and not np.isfinite(pred[1])


def test_pickle_roundtrip(oracle):
    X, y = make('qc', 300, np.random.default_rng(0))
    est = pantara.PantaraRegressor(model=oracle).fit(X, y)
    clone = pickle.loads(pickle.dumps(est))
    assert clone.model is None  # l'oracle n'est pas sérialisé
    np.testing.assert_array_equal(clone.predict(X), est.predict(X))
    assert clone.expression() == est.expression()


def test_expression_power_law(oracle):
    X, y = make('power_law', 300, np.random.default_rng(0))
    est = pantara.PantaraRegressor(model=oracle).fit(X, y, feature_names=['m', 'r'])
    assert est.expression() == '3 × m^1.00 × r^-2.00'
