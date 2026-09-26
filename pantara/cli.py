"""
Pantara — interface en ligne de commande.

    pantara run data.csv --target force [--features a,b] [--task regression]
    pantara fit data.csv --target force --save modele.pkl
    pantara predict mesures.csv --model modele.pkl [--output sortie.csv]
"""

import argparse
import csv
import io
import os
import pickle
import re
import sys
import time
import unicodedata
import warnings

import numpy as np

TITLE      = "Pantara v7d — Découverte automatique de lois physiques"
MIN_POINTS   = 30    # en dessous : avertissement
MAX_FEATURES = 8     # au-delà : avertissement performance
NO_LAW_R2    = 0.5   # R² en dessous duquel on considère qu'aucune loi n'est trouvée
NA_TOKENS  = {'', 'na', 'nan', 'null', 'none', 'n/a', '#n/a'}
ENCODINGS  = ('utf-8-sig', 'cp1252', 'latin-1')
DELIMITERS = (',', ';', '\t')

class CliError(Exception):
    """Erreur utilisateur : message affiché sans trace Python."""

def warn(msg):
    print(f"⚠ {msg}", file=sys.stderr)

# ─────────────────────────────────────────────
# LECTURE CSV
# ─────────────────────────────────────────────
class Table:
    """CSV brut : en-têtes, lignes (texte) et colonnes converties."""

    def __init__(self, path, headers, rows, delimiter):
        self.path      = path
        self.headers   = headers
        self.rows      = rows
        self.delimiter = delimiter
        self.names     = _unique_clean_names(headers)
        self.columns   = [_convert_column([r[j] for r in rows], delimiter)
                          for j in range(len(headers))]

    def is_numeric(self, j):
        return self.columns[j].dtype != object

    def find(self, name):
        """Index d'une colonne par nom exact, sans unité, ou sans casse/accents."""
        for candidates in (self.headers, self.names):
            if name in candidates:
                return candidates.index(name)
        key = _fold(clean_name(name))
        for j, n in enumerate(self.names):
            if _fold(n) == key:
                return j
        raise CliError(f"colonne « {name} » introuvable dans {os.path.basename(self.path)}. "
                       f"Colonnes disponibles : {', '.join(self.headers)}")

def read_table(path):
    if not os.path.isfile(path):
        raise CliError(f"fichier introuvable : {path}")
    with open(path, 'rb') as f:
        raw = f.read()
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    delimiter = _sniff_delimiter(text)
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    if not rows:
        raise CliError(f"{path} est vide")
    width = len(rows[0])
    rows = [(r + [''] * width)[:width] for r in rows]
    header = [c.strip() for c in rows[0]]
    if all(_is_number(c, delimiter) for c in header):
        # pas de ligne d'en-tête : colonnes numérotées
        header = [f"col{j+1}" for j in range(width)]
    else:
        rows = rows[1:]
    if not rows:
        raise CliError(f"{path} ne contient aucune ligne de données")
    return Table(path, header, rows, delimiter)

def _sniff_delimiter(text):
    """Séparateur donnant le même nombre (>1) de colonnes sur en-tête et données."""
    lines = [l for l in text.splitlines() if l.strip()][:50]
    best, best_width = ',', 1
    for d in DELIMITERS:
        widths = [len(r) for r in csv.reader(lines, delimiter=d)]
        if not widths or widths[0] < 2:
            continue
        consistent = sum(w == widths[0] for w in widths) >= 0.9*len(widths)
        if consistent and widths[0] > best_width:
            best, best_width = d, widths[0]
    return best

def _to_float(s, delimiter):
    """float, ou None si valeur manquante ; ValueError si non numérique."""
    s = s.strip().replace(' ', '').replace(' ', '')
    if s.lower() in NA_TOKENS:
        return None
    try:
        return float(s)
    except ValueError:
        if delimiter != ',' and s.count(',') == 1:
            return float(s.replace(',', '.'))  # virgule décimale (CSV « ; » français)
        raise

def _is_number(s, delimiter):
    try:
        return _to_float(s, delimiter) is not None
    except ValueError:
        return False

def _convert_column(values, delimiter):
    """ndarray float (NaN = manquant) si la colonne est numérique, sinon object."""
    out = []
    try:
        for v in values:
            f = _to_float(v, delimiter)
            out.append(np.nan if f is None else f)
    except ValueError:
        return np.array([None if v.strip().lower() in NA_TOKENS else v.strip()
                         for v in values], dtype=object)
    return np.array(out, dtype=np.float64)

_UNIT_RE = re.compile(r'\s*[\(\[][^\)\]]*[\)\]]\s*$')

def clean_name(header):
    """'Force (N)' → 'Force' ; 'Déplacement [mm]' → 'Déplacement'."""
    name = _UNIT_RE.sub('', header.strip()).strip()
    return name or header.strip()

def _unique_clean_names(headers):
    names = [clean_name(h) for h in headers]
    return [n if names.count(n) == 1 else h.strip() for n, h in zip(names, headers)]

def _fold(s):
    s = unicodedata.normalize('NFKD', s)
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()

# ─────────────────────────────────────────────
# TÂCHE
# ─────────────────────────────────────────────
def detect_task(y):
    if y.dtype == object:
        return 'classification'
    if len(np.unique(y)) <= 10:
        return 'classification'
    if len(np.unique(y)) / len(y) < 0.05:
        return 'classification'
    return 'regression'

# ─────────────────────────────────────────────
# PRÉPARATION DES DONNÉES
# ─────────────────────────────────────────────
def load_dataset(path, target, features=None):
    """(table, X, y, noms des features, nom de la cible, index des features)."""
    table = read_table(path)
    t = table.find(target)

    if features:
        f_idx = [table.find(f.strip()) for f in features.split(',') if f.strip()]
        if t in f_idx:
            raise CliError(f"la cible « {table.names[t]} » ne peut pas être aussi une feature")
        non_num = [table.headers[j] for j in f_idx if not table.is_numeric(j)]
        if non_num:
            raise CliError(f"feature(s) non numérique(s) : {', '.join(non_num)}")
    else:
        f_idx = [j for j in range(len(table.headers)) if j != t and table.is_numeric(j)]
        ignored = [table.headers[j] for j in range(len(table.headers))
                   if j != t and not table.is_numeric(j)]
        if ignored:
            warn(f"colonne(s) non numérique(s) ignorée(s) : {', '.join(ignored)}")
    if not f_idx:
        raise CliError("aucune feature numérique utilisable")

    y = table.columns[t]
    X = np.column_stack([table.columns[j] for j in f_idx])
    keep = ~np.isnan(X).any(axis=1)
    keep &= (np.array([v is not None for v in y]) if y.dtype == object else ~np.isnan(y))
    n_drop = int((~keep).sum())
    if n_drop:
        warn(f"{n_drop} ligne(s) avec valeurs manquantes supprimée(s)")
    X, y = X[keep], y[keep]
    if len(y) == 0:
        raise CliError("aucune ligne complète après suppression des valeurs manquantes")

    return table, X, y, [table.names[j] for j in f_idx], table.names[t], f_idx

def check_size(n, d):
    if n < MIN_POINTS:
        warn(f"seulement {n} points (< {MIN_POINTS}) : résultats peu fiables")
    if d > MAX_FEATURES:
        warn(f"{d} features (> {MAX_FEATURES}) : performance dégradée "
             f"(recherche combinatoire, oracle limité à 4 variables)")

# ─────────────────────────────────────────────
# RAPPORT
# ─────────────────────────────────────────────
def confidence(r2, n):
    levels = ['faible', 'moyenne', 'haute']
    lvl = 2 if r2 >= 0.99 else 1 if r2 >= 0.90 else 0
    if n < MIN_POINTS:
        lvl = max(lvl - 1, 0)
    return levels[lvl]

def law_name(chosen):
    """['power_law[log_log_lstsq]', 'sin[...]'] → 'power_law + sin'."""
    return ' + '.join(c.split('[')[0] for c in chosen)

def print_header(path, task, task_forced, n, feat_names, target):
    labels = {'regression': 'régression', 'classification': 'classification'}
    how = 'imposée' if task_forced else 'détectée automatiquement'
    print(TITLE)
    print('=' * len(TITLE))
    print(f"Fichier     : {os.path.basename(path)}")
    print(f"Tâche       : {labels[task]} ({how})")
    print(f"Points      : {n}")
    print(f"Features    : {', '.join(feat_names)} ({len(feat_names)})")
    print(f"Cible       : {target}")
    print()

def print_result(est, elapsed, n):
    if not est.blocks_ or not (est.r2_ >= NO_LAW_R2):
        print("Aucune structure détectée dans ces données.")
        print("Vérifier : les variables sont-elles correctement choisies ?")
        print("La relation est-elle algébrique ou puissance ?")
        return False
    print(f"Loi détectée    : {law_name(est.chosen_)}")
    print(f"Expression      : {est.expression()}")
    print(f"R²              : {est.r2_:.4f}")
    print(f"Précision ±5%   : {est.precision_*100:.1f}%")
    print(f"Temps           : {elapsed:.2f}s")
    print(f"Confiance       : {confidence(est.r2_, n)}")
    print()
    print(f"Blocs           : {', '.join(est.chosen_)}")
    return True

# ─────────────────────────────────────────────
# COMMANDES
# ─────────────────────────────────────────────
def _analyse(args):
    """Charge, vérifie la tâche, ajuste. Renvoie (est, trouvé) ou None si classification."""
    from pantara import PantaraRegressor

    table, X, y, feat_names, target, f_idx = load_dataset(args.data, args.target, args.features)
    task = args.task if args.task != 'auto' else detect_task(y)
    print_header(args.data, task, args.task != 'auto', len(y), feat_names, target)

    if task == 'classification':
        print("Classification en cours de développement")
        return None
    if y.dtype == object:
        raise CliError(f"la cible « {target} » n'est pas numérique : régression impossible")
    if np.std(y) == 0:
        raise CliError(f"la cible « {target} » est constante")
    check_size(len(y), len(feat_names))

    print("Analyse en cours...")
    print()
    t0 = time.time()
    est = PantaraRegressor(random_state=args.seed).fit(X, y, feature_names=feat_names)
    elapsed = time.time() - t0
    est.target_name_ = target
    found = print_result(est, elapsed, len(y))
    return est, found

def cmd_run(args):
    _analyse(args)
    return 0

def cmd_fit(args):
    res = _analyse(args)
    if res is None:
        return 0
    est, found = res
    if not found:
        print()
        print("Modèle non sauvegardé.")
        return 1
    with open(args.save, 'wb') as f:
        pickle.dump(est, f)
    print()
    print(f"Modèle sauvegardé : {args.save}")
    return 0

def cmd_predict(args):
    from pantara import PantaraRegressor

    if not os.path.isfile(args.model):
        raise CliError(f"modèle introuvable : {args.model}")
    try:
        with open(args.model, 'rb') as f:
            est = pickle.load(f)
    except Exception as e:
        raise CliError(f"impossible de lire le modèle {args.model} : {e}")
    if not isinstance(est, PantaraRegressor) or not hasattr(est, 'blocks_'):
        raise CliError(f"{args.model} n'est pas un modèle Pantara entraîné "
                       f"(créé par `pantara fit`)")

    table = read_table(args.data)
    f_idx = [table.find(n) for n in est.feature_names_]
    non_num = [table.headers[j] for j in f_idx if not table.is_numeric(j)]
    if non_num:
        raise CliError(f"feature(s) non numérique(s) : {', '.join(non_num)}")
    X = np.column_stack([table.columns[j] for j in f_idx])
    ok = ~np.isnan(X).any(axis=1)
    if not ok.all():
        warn(f"{int((~ok).sum())} ligne(s) avec valeurs manquantes : pas de prédiction")

    pred = np.full(len(X), np.nan)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if ok.any():
            pred[ok] = est.predict(X[ok])
    for w in caught:
        warn(str(w.message))

    target = getattr(est, 'target_name_', 'y')
    out_col = f"{target}_pred"
    out = open(args.output, 'w', newline='', encoding='utf-8') if args.output else sys.stdout
    try:
        writer = csv.writer(out, delimiter=table.delimiter)
        writer.writerow(table.headers + [out_col])
        for row, p in zip(table.rows, pred):
            writer.writerow(row + ['' if not np.isfinite(p) else f"{p:.10g}"])
    finally:
        if args.output:
            out.close()

    print(f"Modèle      : {est.expression()}", file=sys.stderr)
    print(f"Prédictions : {int(np.isfinite(pred).sum())}/{len(pred)} ligne(s)"
          + (f" → {args.output}" if args.output else ''), file=sys.stderr)
    # Si la vraie cible est présente, évaluer la généralisation
    try:
        t = table.find(target)
    except CliError:
        return 0
    y = table.columns[t]
    if table.is_numeric(t):
        m = np.isfinite(pred) & ~np.isnan(y)
        if m.sum() >= 2 and np.std(y[m]) > 0:
            r2 = 1 - np.sum((y[m] - pred[m])**2) / np.sum((y[m] - y[m].mean())**2)
            prec = np.mean(np.abs(pred[m] - y[m]) / (np.abs(y[m]) + 1e-8) < 0.05)
            print(f"R² (cible « {target} » présente) : {r2:.4f}   "
                  f"Précision ±5% : {prec*100:.1f}%", file=sys.stderr)
    return 0

# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
def build_parser():
    p = argparse.ArgumentParser(
        prog='pantara', description=TITLE,
        epilog="Exemple : pantara run capteurs.csv --target force")
    sub = p.add_subparsers(dest='command', required=True)

    def add_data_args(sp):
        sp.add_argument('data', help="fichier CSV (séparateur , ; ou tabulation)")
        sp.add_argument('--target', required=True,
                        help="colonne cible (nom avec ou sans unité : 'Force (N)' ou 'Force')")
        sp.add_argument('--features',
                        help="colonnes explicatives séparées par des virgules "
                             "(défaut : toutes les colonnes numériques sauf la cible)")
        sp.add_argument('--task', choices=['auto', 'regression', 'classification'],
                        default='auto', help="type de tâche (défaut : détection automatique)")
        sp.add_argument('--seed', type=int, default=42, help="graine aléatoire (défaut : 42)")

    sp = sub.add_parser('run', help="découvrir la loi et afficher le rapport")
    add_data_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser('fit', help="découvrir la loi et sauvegarder le modèle")
    add_data_args(sp)
    sp.add_argument('--save', required=True, help="fichier .pkl de sortie")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser('predict', help="prédire avec un modèle sauvegardé",
                        description="Prédit la cible pour chaque ligne. Le modèle est un "
                                    "pickle : ne charger que des fichiers de confiance.")
    sp.add_argument('data', help="fichier CSV contenant les features du modèle")
    sp.add_argument('--model', required=True, help="modèle créé par `pantara fit`")
    sp.add_argument('--output', help="CSV de sortie (défaut : sortie standard)")
    sp.set_defaults(func=cmd_predict)
    return p

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except CliError as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130

if __name__ == '__main__':
    sys.exit(main())
