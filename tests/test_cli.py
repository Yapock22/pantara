"""Tests du CLI (pantara/cli.py)."""

import csv

import numpy as np
import pytest

from pantara import cli


def write_csv(path, header, rows, delimiter=',', encoding='utf-8'):
    with open(path, 'w', newline='', encoding=encoding) as f:
        w = csv.writer(f, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)
    return str(path)


def force_rows(n, seed=0, decimal_comma=False):
    rng = np.random.default_rng(seed)
    d = rng.uniform(1, 8, n)
    v = rng.uniform(1, 6, n)
    f = 3.42*d*v**-2
    fmt = (lambda x: f"{x:.12g}".replace('.', ',')) if decimal_comma else (lambda x: f"{x:.12g}")
    return [[fmt(a), fmt(b), fmt(c), 'ok'] for a, b, c in zip(d, v, f)]


# ── detect_task ──
def test_detect_task():
    assert cli.detect_task(np.array(['a', 'b'], dtype=object)) == 'classification'
    assert cli.detect_task(np.array([0., 1.] * 50)) == 'classification'
    assert cli.detect_task(np.repeat(np.arange(20.), 50)) == 'classification'
    assert cli.detect_task(np.linspace(0, 1, 500)) == 'regression'


# ── lecture CSV ──
def test_read_semicolon_latin1_units_decimal_comma(tmp_path):
    path = write_csv(tmp_path / 'fr.csv', ['Déplacement (mm)', 'Vitesse [m/s]', 'Force (N)', 'etat'],
                     force_rows(5, decimal_comma=True), delimiter=';', encoding='latin-1')
    t = cli.read_table(path)
    assert t.delimiter == ';'
    assert t.names == ['Déplacement', 'Vitesse', 'Force', 'etat']
    assert t.is_numeric(0) and t.is_numeric(2) and not t.is_numeric(3)
    assert t.find('Force (N)') == t.find('force') == t.find('Force') == 2
    assert t.find('deplacement') == 0


def test_read_tab_separated(tmp_path):
    path = write_csv(tmp_path / 'tab.csv', ['a', 'b'], [[1, 2], [3, 4]], delimiter='\t')
    t = cli.read_table(path)
    assert t.delimiter == '\t'
    np.testing.assert_array_equal(t.columns[1], [2., 4.])


def test_load_dataset_drops_nan_and_ignores_text(tmp_path, capsys):
    rows = force_rows(40)
    rows[3][1] = ''
    rows[7][2] = 'NaN'
    path = write_csv(tmp_path / 'd.csv', ['displacement', 'velocity', 'force', 'etat'], rows)
    _, X, y, names, target, _ = cli.load_dataset(path, 'force')
    assert names == ['displacement', 'velocity'] and target == 'force'
    assert X.shape == (38, 2) and len(y) == 38
    err = capsys.readouterr().err
    assert '2 ligne(s)' in err and 'etat' in err


def test_unknown_target(tmp_path, capsys):
    path = write_csv(tmp_path / 'd.csv', ['a', 'b'], [[1, 2]] * 40)
    assert cli.main(['run', path, '--target', 'zzz']) == 2
    assert 'introuvable' in capsys.readouterr().err


# ── commandes ──
def test_run_power_law(tmp_path, capsys):
    path = write_csv(tmp_path / 'capteurs.csv',
                     ['displacement', 'velocity', 'force', 'etat'], force_rows(300))
    assert cli.main(['run', path, '--target', 'force']) == 0
    out = capsys.readouterr().out
    assert 'Tâche       : régression (détectée automatiquement)' in out
    assert 'Features    : displacement, velocity (2)' in out
    assert 'Loi détectée    : power_law' in out
    assert 'Expression      : 3.42 × displacement^1.00 × velocity^-2.00' in out
    assert 'Confiance       : haute' in out
    assert 'Blocs           : power_law[log_log_lstsq]' in out


def test_run_classification_exits_cleanly(tmp_path, capsys):
    rng = np.random.default_rng(0)
    rows = [[f"{x:.4f}", int(x > 0.5)] for x in rng.uniform(0, 1, 100)]
    path = write_csv(tmp_path / 'c.csv', ['x', 'label'], rows)
    assert cli.main(['run', path, '--target', 'label']) == 0
    assert 'Classification en cours de développement' in capsys.readouterr().out


def test_run_no_structure(tmp_path, capsys):
    rng = np.random.default_rng(0)
    rows = [[f"{a:.6f}", f"{b:.6f}"] for a, b in rng.normal(size=(200, 2))]
    path = write_csv(tmp_path / 'bruit.csv', ['x', 'y'], rows)
    assert cli.main(['run', path, '--target', 'y']) == 0
    assert 'Aucune structure détectée' in capsys.readouterr().out


def test_fit_then_predict(tmp_path, capsys):
    train = write_csv(tmp_path / 'historique.csv',
                      ['Déplacement (mm)', 'Vitesse (m/s)', 'Force (N)'],
                      [r[:3] for r in force_rows(300, seed=0, decimal_comma=True)],
                      delimiter=';', encoding='latin-1')
    model = str(tmp_path / 'modele.pkl')
    assert cli.main(['fit', train, '--target', 'Force (N)', '--save', model]) == 0
    assert 'Modèle sauvegardé' in capsys.readouterr().out

    # Nouvelles mesures : autre ordre de colonnes, autre séparateur, sans la cible
    rng = np.random.default_rng(5)
    d, v = rng.uniform(1, 8, 20), rng.uniform(1, 6, 20)
    new = write_csv(tmp_path / 'mesures.csv', ['vitesse', 'deplacement'],
                    [[f"{b:.12g}", f"{a:.12g}"] for a, b in zip(d, v)])
    out = str(tmp_path / 'pred.csv')
    assert cli.main(['predict', new, '--model', model, '--output', out]) == 0
    with open(out, encoding='utf-8') as f:
        rows = list(csv.reader(f))
    assert rows[0] == ['vitesse', 'deplacement', 'Force_pred']
    pred = np.array([float(r[2]) for r in rows[1:]])
    np.testing.assert_allclose(pred, 3.42*d*v**-2, rtol=1e-6)


def test_predict_reports_r2_when_target_present(tmp_path, capsys):
    path = write_csv(tmp_path / 'd.csv', ['displacement', 'velocity', 'force'],
                     [r[:3] for r in force_rows(200)])
    model = str(tmp_path / 'm.pkl')
    cli.main(['fit', path, '--target', 'force', '--save', model])
    capsys.readouterr()
    test = write_csv(tmp_path / 't.csv', ['displacement', 'velocity', 'force'],
                     [r[:3] for r in force_rows(50, seed=9)])
    assert cli.main(['predict', test, '--model', model]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == 'displacement,velocity,force,force_pred'
    assert 'R² (cible « force » présente) : 1.0000' in captured.err


def test_predict_rejects_non_model(tmp_path, capsys):
    import pickle
    bad = tmp_path / 'bad.pkl'
    bad.write_bytes(pickle.dumps({'not': 'a model'}))
    data = write_csv(tmp_path / 'd.csv', ['a'], [[1]])
    assert cli.main(['predict', data, '--model', str(bad)]) == 2
    assert "n'est pas un modèle Pantara" in capsys.readouterr().err
