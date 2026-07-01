"""
PhysiqAI — Agent Compositeur v7 : Sprint 7
===========================================
Mac M5 Pro · PyTorch + MPS

Nouveautés v7 :

  Fix A — sq_y faux positifs
    Exclure sq_y si le résidu est trop symétrique
    (skewness faible = le résidu est déjà du bruit)

  Fix B — dénormalisation log_log_multi
    Après régression dans log_log_multi, la prédiction
    est en espace log → dénormaliser par division dans
    l'espace original, pas soustraction

  Fix C — seuil arrêt matching pursuit plus strict
    min_score passe de 0.20 à 0.30
    évite d'ajouter des blocs parasites sur du bruit

  Oracle sur données brutes
    Au lieu d'un vecteur 56D de statistiques,
    le réseau voit directement (X, y) subsampled.
    Architecture : SetEncoder → vecteur fixe → classificateur
    L'oracle apprend des patterns que les corrélations
    ne capturent pas (forme, courbure, asymétrie locale)
"""

import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ─── DEVICE ───
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ GPU Apple Silicon (MPS) détecté")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ GPU CUDA détecté")
else:
    DEVICE = torch.device("cpu")
    print("⚠ CPU uniquement")
print(f"  Device : {DEVICE}\n")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FAMILIES  = ['lin','sq','pair','qc','trip','qinv','sin','cos']
SPACES    = ['original','log_y','log_x','log_log','sq_y','inv_x']
N_CLASSES = 9   # 8 familles + STOP
N_POINTS  = 64  # points subsampled pour l'oracle
STATE_DIM = 56  # état statistique (conservé pour compatibilité)

# ─────────────────────────────────────────────
# TRANSFORMATIONS D'ESPACE
# ─────────────────────────────────────────────
def transform_space(X_np, y_np, space):
    eps = 1e-10
    try:
        if space == 'original':
            return X_np.copy(), y_np.copy(), True
        elif space == 'log_y':
            if np.any(y_np <= 0): return X_np, y_np, False
            return X_np.copy(), np.log(y_np + eps), True
        elif space == 'log_x':
            if np.any(X_np <= 0): return X_np, y_np, False
            return np.log(X_np + eps), y_np.copy(), True
        elif space == 'log_log':
            if np.any(X_np <= 0) or np.any(y_np <= 0): return X_np, y_np, False
            return np.log(X_np + eps), np.log(y_np + eps), True
        elif space == 'sq_y':
            return X_np.copy(), y_np**2, True
        elif space == 'inv_x':
            if np.any(X_np == 0): return X_np, y_np, False
            return 1.0/(X_np + eps), y_np.copy(), True
    except:
        return X_np, y_np, False

def inverse_transform_y(y_t, space):
    if space == 'original': return y_t
    elif space == 'log_y':  return np.exp(y_t)
    elif space == 'log_x':  return y_t
    elif space == 'log_log':return np.exp(y_t)
    elif space == 'sq_y':   return np.sqrt(np.abs(y_t))
    elif space == 'inv_x':  return y_t
    return y_t

def corr_np(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))

def standardize(col):
    mu = col.mean(); s = col.std()
    if s < 1e-10: return col - mu
    return (col - mu) / s

# ─────────────────────────────────────────────
# FIX A — sq_y faux positifs
# N'utiliser sq_y que si le résidu est asymétrique
# (skewness élevé = y² pourrait linéariser)
# ─────────────────────────────────────────────
SQ_Y_MIN_SKEW = 0.5  # skewness minimum pour utiliser sq_y

def resid_is_skewed(resid_np):
    """Vrai si le résidu est suffisamment asymétrique pour justifier sq_y."""
    mu = resid_np.mean(); s = resid_np.std() + 1e-10
    skew = abs(float(((resid_np - mu)**3).mean() / s**3))
    return skew > SQ_Y_MIN_SKEW

# ─────────────────────────────────────────────
# LOG_LOG MULTI-COLONNES
# ─────────────────────────────────────────────
def best_multivar_loglog(X_np, resid_np):
    if np.any(X_np <= 0) or np.any(resid_np <= 0):
        return 0.0, None, None, 'log_log_multi'
    eps = 1e-10; N, D = X_np.shape
    log_X   = np.log(X_np + eps)
    log_res = np.log(np.abs(resid_np) + eps)
    best_score = 0.0; best_col = None

    for j in range(D):
        lx = log_X[:,j]; cov = np.cov(lx, log_res)
        if cov[0,0] < 1e-10: continue
        n_hat = cov[0,1]/cov[0,0]; col = n_hat * lx
        s = abs(corr_np(standardize(col), standardize(log_res)))
        if s > best_score:
            best_score = s
            best_col = np.sign(X_np[:,j])*np.abs(X_np[:,j])**n_hat
    for j in range(D):
        for k in range(j+1, D):
            col = log_X[:,j]+log_X[:,k]
            s = abs(corr_np(standardize(col), standardize(log_res)))
            if s > best_score:
                best_score = s; best_col = X_np[:,j]*X_np[:,k]
    if D >= 3:
        for j in range(D):
            for k in range(j+1, D):
                for l in range(D):
                    if l in (j,k): continue
                    col = log_X[:,j]+log_X[:,k]-2*log_X[:,l]
                    s = abs(corr_np(standardize(col), standardize(log_res)))
                    if s > best_score:
                        best_score = s
                        best_col = X_np[:,j]*X_np[:,k]/X_np[:,l]**2
        for j in range(D):
            for k in range(j+1,D):
                for l in range(k+1,D):
                    col = log_X[:,j]+log_X[:,k]+log_X[:,l]
                    s = abs(corr_np(standardize(col), standardize(log_res)))
                    if s > best_score:
                        best_score = s
                        best_col = X_np[:,j]*X_np[:,k]*X_np[:,l]
    return best_score, best_col, log_X, 'log_log_multi'

# ─────────────────────────────────────────────
# DÉTECTION TRIG (avec seuils v6b)
# ─────────────────────────────────────────────
TRIG_MIN_CORR   = 0.70
TRIG_MIN_CYCLES = 2.0

def detect_trig(X_np, resid_np):
    N, D = X_np.shape
    best_score = 0.0; best_result = None
    for j in range(D):
        xj = X_np[:,j]
        x_min, x_max = xj.min(), xj.max()
        x_range = x_max - x_min
        if x_range < 1e-10: continue
        omega_max = 2*math.pi*TRIG_MIN_CYCLES/x_range
        sort_idx   = np.argsort(xj)
        xj_sorted  = xj[sort_idx]
        res_sorted = resid_np[sort_idx]
        x_grid     = np.linspace(x_min, x_max, N)
        res_interp = np.interp(x_grid, xj_sorted, res_sorted)
        fft_vals   = np.fft.rfft(res_interp)
        fft_freq   = np.fft.rfftfreq(N, d=x_range/N)
        fft_amp    = np.abs(fft_vals); fft_amp[0] = 0
        top_idx    = np.argsort(fft_amp)[-8:][::-1]
        for idx in top_idx:
            freq = fft_freq[idx]
            if freq < 1e-10: continue
            omega = 2*math.pi*freq
            if omega > omega_max: continue
            col_sin = np.sin(omega*xj)
            s_sin   = abs(corr_np(col_sin, resid_np))
            if s_sin > TRIG_MIN_CORR and s_sin > best_score:
                best_score = s_sin
                best_result = ('sin', col_sin.copy(), f'sin_x{j}_w{omega:.3f}')
            col_cos = np.cos(omega*xj)
            s_cos   = abs(corr_np(col_cos, resid_np))
            if s_cos > TRIG_MIN_CORR and s_cos > best_score:
                best_score = s_cos
                best_result = ('cos', col_cos.copy(), f'cos_x{j}_w{omega:.3f}')
    return best_score, best_result

# ─────────────────────────────────────────────
# DÉTECTION MULTI-ESPACE
# Fix A intégré : sq_y conditionnel
# ─────────────────────────────────────────────
def best_per_family_multispace(X_np, resid_np, y_np):
    N, D = X_np.shape
    scores      = {f:0.0 for f in FAMILIES}
    best_config = {f:(0.0,'original',None,X_np,resid_np) for f in FAMILIES}
    use_sq_y    = resid_is_skewed(resid_np)  # Fix A

    for space in SPACES:
        # Fix A : ignorer sq_y si résidu symétrique
        if space == 'sq_y' and not use_sq_y:
            continue
        X_t, resid_t, valid = transform_space(X_np, resid_np, space)
        if not valid: continue

        for j in range(D):
            col = X_t[:,j]
            s = abs(corr_np(standardize(col), standardize(resid_t)))
            if s > scores['lin']:
                scores['lin']=s
                best_config['lin']=(s,space,col.copy(),X_t,resid_t)
            col2 = col**2
            s2=abs(corr_np(standardize(col2), standardize(resid_t)))
            if s2 > scores['sq']:
                scores['sq']=s2
                best_config['sq']=(s2,space,col2.copy(),X_t,resid_t)

        for j in range(D):
            for k in range(j+1,D):
                col=X_t[:,j]*X_t[:,k]
                s=abs(corr_np(standardize(col), standardize(resid_t)))
                if s>scores['pair']:
                    scores['pair']=s
                    best_config['pair']=(s,space,col.copy(),X_t,resid_t)

        for j in range(D):
            for k in range(D):
                if j!=k:
                    col=X_t[:,j]*X_t[:,k]**2
                    s=abs(corr_np(standardize(col), standardize(resid_t)))
                    if s>scores['qc']:
                        scores['qc']=s
                        best_config['qc']=(s,space,col.copy(),X_t,resid_t)

        if D>=3:
            for j in range(D):
                for k in range(j+1,D):
                    for l in range(k+1,D):
                        col=X_t[:,j]*X_t[:,k]*X_t[:,l]
                        s=abs(corr_np(standardize(col), standardize(resid_t)))
                        if s>scores['trip']:
                            scores['trip']=s
                            best_config['trip']=(s,space,col.copy(),X_t,resid_t)

        for j in range(D):
            for k in range(D):
                if j!=k and np.all(X_t[:,k]!=0):
                    col=X_t[:,j]**2/X_t[:,k]
                    s=abs(corr_np(standardize(col), standardize(resid_t)))
                    if s>scores['qinv']:
                        scores['qinv']=s
                        best_config['qinv']=(s,space,col.copy(),X_t,resid_t)

    # Log_log multi
    if np.all(resid_np > 0):
        s_m, col_m, log_X, _ = best_multivar_loglog(X_np, resid_np)
        if col_m is not None:
            s_orig = abs(corr_np(standardize(col_m), standardize(resid_np)))
            if s_orig > scores['qinv']:
                scores['qinv']=s_orig
                best_config['qinv']=(s_orig,'log_log_multi',
                                      col_m.copy(),log_X,
                                      np.log(np.abs(resid_np)+1e-10))
            if s_orig > scores['trip']:
                scores['trip']=s_orig
                best_config['trip']=(s_orig,'log_log_multi',
                                      col_m.copy(),log_X,
                                      np.log(np.abs(resid_np)+1e-10))

    # Trig
    trig_score, trig_result = detect_trig(X_np, resid_np)
    if trig_result is not None:
        fam_trig, col_trig, space_trig = trig_result
        if trig_score > scores[fam_trig]:
            scores[fam_trig] = trig_score
            best_config[fam_trig] = (trig_score, space_trig,
                                      col_trig, X_np, resid_np)

    return scores, best_config, {}

# ─────────────────────────────────────────────
# FIT BLOCS
# Fix B : dénormalisation log_log_multi
# ─────────────────────────────────────────────
def fit_block_multifeature(X_t, resid_t, space, resid_orig,
                            steps=600, lr=0.01):
    N, D = X_t.shape
    X_std=np.zeros_like(X_t); mu_x=np.zeros(D); s_x=np.ones(D)
    for j in range(D):
        mu_x[j]=X_t[:,j].mean(); s_x[j]=X_t[:,j].std()+1e-10
        X_std[:,j]=(X_t[:,j]-mu_x[j])/s_x[j]
    mu_r=resid_t.mean(); s_r=resid_t.std()+1e-10
    resid_std=(resid_t-mu_r)/s_r
    Xt=torch.tensor(X_std,  dtype=torch.float32,device=DEVICE)
    Rt=torch.tensor(resid_std,dtype=torch.float32,device=DEVICE)
    W=torch.zeros(D,requires_grad=True,device=DEVICE)
    b=torch.zeros(1,requires_grad=True,device=DEVICE)
    opt=optim.Adam([W,b],lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss=((Xt@W+b-Rt)**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred_std=(Xt@W+b).cpu().numpy()
        pred_t=pred_std*s_r+mu_r    # prédiction en espace transformé
        resid_in_space=resid_t-pred_t

    # Fix B : pour log_log_multi, pred_t est log(pred_orig)
    # → pred_orig = exp(pred_t)
    # → new_resid = resid_orig / pred_orig  (division dans l'espace original)
    if space == 'log_log_multi':
        pred_orig = np.exp(pred_t)
        # Éviter division par zéro
        pred_orig = np.where(np.abs(pred_orig) < 1e-30, 1e-30, pred_orig)
        new_resid_orig = resid_orig - pred_orig
    else:
        pred_orig = inverse_transform_y(pred_t, space)
        new_resid_orig = resid_orig - pred_orig

    return new_resid_orig, pred_orig, resid_in_space

def fit_block_1d(col_t, resid_t, space, resid_orig, steps=500, lr=0.02):
    mn_c=np.median(col_t); iqr_c=np.percentile(col_t,75)-np.percentile(col_t,25)+1e-10
    mn_r=np.median(resid_t); iqr_r=np.percentile(resid_t,75)-np.percentile(resid_t,25)+1e-10
    col_n=(col_t-mn_c)/iqr_c; resid_n=(resid_t-mn_r)/iqr_r
    ct=torch.tensor(col_n,  dtype=torch.float32,device=DEVICE)
    rt=torch.tensor(resid_n,dtype=torch.float32,device=DEVICE)
    w=torch.zeros(1,requires_grad=True,device=DEVICE)
    b=torch.zeros(1,requires_grad=True,device=DEVICE)
    opt=optim.Adam([w,b],lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss=((w*ct+b-rt)**2).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        pred_n=(w*ct+b).cpu().numpy()
        pred_t=pred_n*iqr_r+mn_r
        resid_in_space=resid_t-pred_t
    pred_orig=inverse_transform_y(pred_t,space)
    new_resid_orig=resid_orig-pred_orig
    return new_resid_orig, pred_orig, resid_in_space

def fit_block(col_t, resid_t, space, resid_orig,
              X_t=None, steps=500, lr=0.02):
    use_multi = (
        space in ('log_log','log_x','log_y','log_log_multi') and
        X_t is not None and X_t.ndim==2 and X_t.shape[1]>1
    )
    if use_multi:
        return fit_block_multifeature(X_t,resid_t,space,resid_orig,
                                       steps=steps,lr=lr)
    else:
        return fit_block_1d(col_t,resid_t,space,resid_orig,
                             steps=steps,lr=lr)

# ─────────────────────────────────────────────
# ÉTAT STATISTIQUE 56D (conservé)
# ─────────────────────────────────────────────
def moments(arr):
    mu=arr.mean(); s=arr.std()+1e-10
    skew=float(((arr-mu)**3).mean()/s**3)
    kurt=float(((arr-mu)**4).mean()/s**4)-3.0
    return [float(np.tanh(mu/(abs(mu)+1))),float(np.tanh(s)),
            float(np.tanh(skew/3)),float(np.tanh(kurt/10))]

def partial_corr(X_np,y_np,j,k):
    xj=X_np[:,j]; xk=X_np[:,k]
    b=corr_np(xj,xk)*xj.std()/(xk.std()+1e-10)
    res_j=xj-b*xk
    b2=corr_np(y_np,xk)*y_np.std()/(xk.std()+1e-10)
    res_y=y_np-b2*xk
    return abs(corr_np(res_j,res_y))

def log_log_slope(X_np,y_np,j):
    if np.any(X_np[:,j]<=0) or np.any(y_np<=0): return 0.0
    lx=np.log(X_np[:,j]+1e-10); ly=np.log(np.abs(y_np)+1e-10)
    cov=np.cov(lx,ly)
    if cov[0,0]<1e-10: return 0.0
    return float(np.tanh(cov[0,1]/cov[0,0]/5))

def separability_score(X_np,y_np,j,k):
    xk=X_np[:,k]
    bins=np.percentile(xk,[0,20,40,60,80,100])
    total_var=y_np.var()+1e-10; within_var=0.0
    for i in range(len(bins)-1):
        mask=(xk>=bins[i])&(xk<bins[i+1])
        if mask.sum()<5: continue
        y_bin=y_np[mask]; x_bin=X_np[:,j][mask]
        if x_bin.std()<1e-10: within_var+=y_bin.var(); continue
        c=corr_np(x_bin,y_bin); within_var+=y_bin.var()*(1-c**2)
    return float(1.0-within_var/total_var)

def make_state_stats(scores, resid, y_orig, X_np):
    """État statistique 56D (identique v6)."""
    N, D = X_np.shape
    vals=np.array([scores[f] for f in FAMILIES])
    best=vals.max()+1e-10
    raw=vals.tolist(); norm=(vals/best).tolist(); gaps=(best-vals).tolist()
    sv=sorted(vals.tolist(),reverse=True)
    ranks=[sv.index(v)/max(len(sv)-1,1) for v in vals.tolist()]
    ratio=float(resid.std()/(y_orig.std()+1e-10))
    sm=vals.sum()+1e-10; probs=vals/sm
    ent=float(-np.sum(probs*np.log(probs+1e-10))/math.log(len(vals)))
    state_base=raw+norm+gaps+ranks+[ratio,ent]
    mom_y=moments(y_orig); mom_r=moments(resid)
    pcorrs=[]
    for j in range(min(D,3)):
        for k in range(min(D,3)):
            if j!=k: pcorrs.append(partial_corr(X_np,y_orig,j,k))
    while len(pcorrs)<6: pcorrs.append(0.0)
    pcorrs=pcorrs[:6]
    slopes=[log_log_slope(X_np,y_orig,j) for j in range(min(D,4))]
    while len(slopes)<4: slopes.append(0.0)
    slopes=slopes[:4]
    sep_scores=[]
    for j in range(min(D,3)):
        for k in range(min(D,3)):
            if j<k: sep_scores.append(separability_score(X_np,y_orig,j,k))
    while len(sep_scores)<4: sep_scores.append(0.0)
    sep_scores=sep_scores[:4]
    return state_base+mom_y+mom_r+pcorrs+slopes+sep_scores

# ─────────────────────────────────────────────
# ORACLE SUR DONNÉES BRUTES
# Architecture : SetEncoder → classificateur
#
# Idée : subsample N_POINTS points de (X, y),
# normaliser, encoder par un réseau,
# produire un vecteur fixe → classifier.
#
# Le réseau apprend directement depuis les données
# sans passer par des statistiques résumées.
# Il peut détecter : courbure locale, asymétries,
# patterns multi-variables, changements de régime.
# ─────────────────────────────────────────────
def make_oracle_input(X_np, resid_np, y_np, n_points=N_POINTS):
    """
    Prépare l'entrée pour l'oracle sur données brutes.
    Subsample n_points points, normalise tout dans [-1, 1].
    Retourne un tenseur (n_points, D+2) :
      colonnes : [x1_norm, x2_norm, ..., resid_norm, y_norm]
    """
    N, D = X_np.shape
    # Subsample
    idx = np.random.choice(N, min(n_points, N), replace=False)
    X_sub    = X_np[idx]
    resid_sub= resid_np[idx]
    y_sub    = y_np[idx]

    # Normaliser chaque colonne dans [-1, 1]
    def norm_col(col):
        mn, mx = col.min(), col.max()
        r = mx - mn + 1e-10
        return 2*(col - mn)/r - 1

    cols = []
    for j in range(min(D, 4)):  # max 4 features d'entrée
        cols.append(norm_col(X_sub[:,j]))
    # Padding si D < 4
    while len(cols) < 4:
        cols.append(np.zeros(len(idx)))

    cols.append(norm_col(resid_sub))
    cols.append(norm_col(y_sub))

    arr = np.column_stack(cols).astype(np.float32)  # (n_points, 6)
    return arr

class SetEncoder(nn.Module):
    """
    Encode un ensemble de points (X, resid, y) en vecteur fixe.
    Utilise un MLP point-wise + agrégation par moyenne et max.
    Cette architecture est invariante à l'ordre des points.
    """
    def __init__(self, in_dim=6, hidden=64, out_dim=128):
        super().__init__()
        self.point_net = nn.Sequential(
            nn.Linear(in_dim, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim * 2  # mean + max pooling

    def forward(self, x):
        # x : (batch, n_points, in_dim)
        h = self.point_net(x)         # (batch, n_points, out_dim)
        mean_pool = h.mean(dim=1)     # (batch, out_dim)
        max_pool  = h.max(dim=1)[0]  # (batch, out_dim)
        return torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2*out_dim)

class OracleClassifier(nn.Module):
    """
    Classificateur oracle sur données brutes.
    Combine SetEncoder (données brutes) + état statistique 56D.
    """
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.encoder = SetEncoder(in_dim=6, hidden=64, out_dim=128)
        enc_dim = self.encoder.out_dim  # 256

        # Tête de classification combinée
        # Entrée : encodage(256) + stats(56) = 312
        self.head = nn.Sequential(
            nn.Linear(enc_dim + STATE_DIM, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128),                 nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(128, 64),                  nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, points, stats):
        # points : (batch, n_points, 6)
        # stats  : (batch, 56)
        enc  = self.encoder(points)        # (batch, 256)
        feat = torch.cat([enc, stats], 1)  # (batch, 312)
        return self.head(feat)

    def predict_ranked(self, points_np, stats_list):
        pts = torch.tensor(points_np[None], dtype=torch.float32, device=DEVICE)
        st  = torch.tensor([stats_list],    dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self(pts, st)[0], dim=0).cpu().numpy()
        ranked = sorted(range(N_CLASSES), key=lambda i: probs[i], reverse=True)
        return ranked, probs

def oracle_label(scores, resid, y_orig):
    ratio = resid.std()/(y_orig.std()+1e-10)
    if ratio < 0.07: return 8
    best = max(scores, key=scores.get)
    if scores[best] < 0.55: return 8
    return FAMILIES.index(best)

def bloc_score(resid_before, resid_after, y_orig, scores_after):
    std_b=resid_before.std(); std_a=resid_after.std()
    improvement=(std_b-std_a)/(std_b+1e-10)
    residual_level=std_a/(y_orig.std()+1e-10)
    mu=resid_after.mean(); s=resid_after.std()+1e-10
    skew=float(((resid_after-mu)**3).mean()/s**3)
    gaussianity=1.0/(1.0+abs(skew))
    max_resid_corr=max(scores_after.values())
    isolation=1.0-max_resid_corr
    return (0.50*improvement+0.20*gaussianity+
            0.15*(1.0-residual_level)+0.15*isolation)

# ─────────────────────────────────────────────
# GÉNÉRATION DES LOIS (identique v6b)
# ─────────────────────────────────────────────

# ─── ESTIMATEUR TRIGONOMETRIQUE (Sprint 7c) ───
def fft_omega_candidates(x_np, resid_np, n_cand=8):
    """
    Retourne les n_cand fréquences ω les plus probables
    depuis la FFT du résidu.
    """
    N = len(x_np)
    x_min, x_max = x_np.min(), x_np.max()
    x_range = x_max - x_min
    if x_range < 1e-10:
        return []

    # Interpoler sur grille régulière
    sort_idx   = np.argsort(x_np)
    x_sorted   = x_np[sort_idx]
    res_sorted = resid_np[sort_idx]
    x_grid     = np.linspace(x_min, x_max, N)
    res_interp = np.interp(x_grid, x_sorted, res_sorted)

    # FFT
    fft_vals = np.fft.rfft(res_interp)
    fft_freq = np.fft.rfftfreq(N, d=x_range/N)
    fft_amp  = np.abs(fft_vals)
    fft_amp[0] = 0  # ignorer DC

    # Top fréquences
    top_idx  = np.argsort(fft_amp)[-n_cand:][::-1]
    omegas   = []
    for idx in top_idx:
        freq = fft_freq[idx]
        if freq < 1e-10: continue
        omega = 2 * math.pi * freq
        # Exiger au moins 1.5 cycles visibles
        omega_min = 2 * math.pi * 1.5 / x_range
        if omega < omega_min: continue
        omegas.append(float(omega))

    return omegas

# ─────────────────────────────────────────────
# ESTIMATEUR TRIGONOMÉTRIQUE PRINCIPAL
# Optimise A, ω, φ, C par descente de gradient
# ─────────────────────────────────────────────
def fit_trig_single(x_np, resid_np, omega_init,
                     form='sin', steps=800, lr=0.05):
    """
    Optimise y = A·form(ω·x + φ) + C par Adam.

    Retourne (pred, params, loss_final)
    params = (A, omega, phi, C)
    """
    N = len(x_np)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    r_t = torch.tensor(resid_np, dtype=torch.float32, device=DEVICE)

    # Initialisation intelligente
    A_init   = float(resid_np.std() * math.sqrt(2))  # amplitude estimée
    phi_init = 0.0
    C_init   = float(resid_np.mean())

    A   = torch.tensor([A_init],   requires_grad=True, device=DEVICE)
    w   = torch.tensor([omega_init], requires_grad=True, device=DEVICE)
    phi = torch.tensor([phi_init], requires_grad=True, device=DEVICE)
    C   = torch.tensor([C_init],   requires_grad=True, device=DEVICE)

    opt = torch.optim.Adam([A, w, phi, C], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best_loss = float('inf')
    best_params = None

    for step in range(steps):
        opt.zero_grad()
        if form == 'sin':
            pred = A * torch.sin(w * x_t + phi) + C
        else:
            pred = A * torch.cos(w * x_t + phi) + C
        loss = ((pred - r_t)**2).mean()
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_([A, w, phi, C], max_norm=1.0)
        opt.step()
        sched.step()

        # Forcer ω > 0
        with torch.no_grad():
            w.clamp_(min=0.01)

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            best_params = (
                float(A.detach()),
                float(w.detach()),
                float(phi.detach()),
                float(C.detach())
            )

    # Prédiction finale avec meilleurs paramètres
    A_best, w_best, phi_best, C_best = best_params
    if form == 'sin':
        pred_np = A_best * np.sin(w_best * x_np + phi_best) + C_best
    else:
        pred_np = A_best * np.cos(w_best * x_np + phi_best) + C_best

    return pred_np, best_params, best_loss

# ─────────────────────────────────────────────
# ESTIMATEUR AVEC TENDANCE LINÉAIRE
# y = A·sin(ω·x + φ) + B·x + C
# ─────────────────────────────────────────────
def fit_trig_linear(x_np, resid_np, omega_init,
                     form='sin', steps=800, lr=0.03):
    """
    Optimise y = A·form(ω·x + φ) + B·x + C par Adam.
    Utile pour sin+lin, oscillateur amorti, etc.
    """
    N = len(x_np)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)
    r_t = torch.tensor(resid_np, dtype=torch.float32, device=DEVICE)

    A_init = float(resid_np.std() * math.sqrt(2))
    C_init = float(resid_np.mean())

    A   = torch.tensor([A_init],    requires_grad=True, device=DEVICE)
    w   = torch.tensor([omega_init], requires_grad=True, device=DEVICE)
    phi = torch.tensor([0.0],        requires_grad=True, device=DEVICE)
    B   = torch.tensor([0.0],        requires_grad=True, device=DEVICE)
    C   = torch.tensor([C_init],    requires_grad=True, device=DEVICE)

    opt   = torch.optim.Adam([A, w, phi, B, C], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    best_loss = float('inf')
    best_params = None

    for step in range(steps):
        opt.zero_grad()
        if form == 'sin':
            pred = A * torch.sin(w * x_t + phi) + B * x_t + C
        else:
            pred = A * torch.cos(w * x_t + phi) + B * x_t + C
        loss = ((pred - r_t)**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([A, w, phi, B, C], max_norm=1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            w.clamp_(min=0.01)

        lv = loss.item()
        if lv < best_loss:
            best_loss = lv
            best_params = (
                float(A.detach()), float(w.detach()),
                float(phi.detach()), float(B.detach()),
                float(C.detach())
            )

    A_b, w_b, phi_b, B_b, C_b = best_params
    if form == 'sin':
        pred_np = A_b * np.sin(w_b * x_np + phi_b) + B_b * x_np + C_b
    else:
        pred_np = A_b * np.cos(w_b * x_np + phi_b) + B_b * x_np + C_b

    return pred_np, best_params, best_loss

# ─────────────────────────────────────────────
# POINT D'ENTRÉE PRINCIPAL
# Appelé depuis le pipeline quand sin/cos détecté
# ─────────────────────────────────────────────
def fit_trig(x_np, resid_np, y_orig,
              omega_candidates=None,
              try_linear=False,
              steps=600, verbose=False):
    """
    Estimateur trigonométrique complet.

    1. Récupère les candidats ω depuis FFT si non fournis
    2. Pour chaque ω : teste sin et cos
    3. Garde la meilleure combinaison (forme, ω)
    4. Option try_linear : teste aussi sin + tendance linéaire

    Retourne :
      new_resid  : résidu après soustraction
      pred       : prédiction
      info       : dict avec paramètres et forme choisie
    """
    N = len(x_np)

    # Obtenir les candidats ω
    if omega_candidates is None or len(omega_candidates) == 0:
        omega_candidates = fft_omega_candidates(x_np, resid_np, n_cand=6)

    if len(omega_candidates) == 0:
        # Fallback : pas de fréquence détectée
        if verbose: print("    fit_trig : aucune frequence candidate")
        return resid_np.copy(), np.zeros(N), {'form': None, 'loss': float('inf')}

    # Normaliser x pour stabilité numérique
    x_min, x_max = x_np.min(), x_np.max()
    x_range = x_max - x_min + 1e-10

    best_loss  = float('inf')
    best_pred  = None
    best_info  = {}

    # Tester chaque candidat ω × {sin, cos}
    for omega_raw in omega_candidates[:5]:  # max 5 candidats
        # Normaliser ω pour travailler sur x normalisé
        for form in ['sin', 'cos']:
            try:
                pred, params, loss = fit_trig_single(
                    x_np, resid_np, omega_raw,
                    form=form, steps=steps, lr=0.05
                )
                if verbose:
                    print(f"    {form} omega={omega_raw:.3f} "
                          f"→ loss={loss:.4f} "
                          f"A={params[0]:.2f} phi={params[2]:.2f}")
                if loss < best_loss:
                    best_loss = loss
                    best_pred = pred
                    best_info = {
                        'form': form, 'omega': params[1],
                        'A': params[0], 'phi': params[2],
                        'C': params[3], 'loss': loss,
                        'type': 'pure'
                    }
            except Exception as e:
                if verbose: print(f"    Erreur {form}: {e}")
                continue

    # Option : tester sin/cos + tendance linéaire
    if try_linear and len(omega_candidates) > 0:
        for form in ['sin', 'cos']:
            try:
                omega_raw = omega_candidates[0]
                pred, params, loss = fit_trig_linear(
                    x_np, resid_np, omega_raw,
                    form=form, steps=steps, lr=0.03
                )
                # Pénaliser légèrement (plus de paramètres)
                loss_penalized = loss * 1.05
                if verbose:
                    print(f"    {form}+lin omega={omega_raw:.3f} "
                          f"→ loss={loss:.4f}")
                if loss_penalized < best_loss:
                    best_loss = loss_penalized
                    best_pred = pred
                    best_info = {
                        'form': f'{form}+lin',
                        'omega': params[1], 'A': params[0],
                        'phi': params[2], 'B': params[3],
                        'C': params[4], 'loss': loss,
                        'type': 'linear'
                    }
            except Exception as e:
                if verbose: print(f"    Erreur {form}+lin: {e}")

    if best_pred is None:
        return resid_np.copy(), np.zeros(N), {'form': None}

    new_resid = resid_np - best_pred
    return new_resid, best_pred, best_info

# ─────────────────────────────────────────────
# ÉVALUATION DE QUALITÉ (compatible bloc_score)
# ─────────────────────────────────────────────
def evaluate_trig(y_pred, y_true, tol=0.05):
    return float(np.mean(np.abs(y_pred - y_true) /
                          (np.abs(y_true) + 1e-8) < tol))

def trig_quality(resid_before, resid_after, y_orig):
    """Score de qualité du bloc trig (même logique que bloc_score)."""
    std_b = resid_before.std(); std_a = resid_after.std()
    improvement = (std_b - std_a) / (std_b + 1e-10)
    residual_level = std_a / (y_orig.std() + 1e-10)
    mu = resid_after.mean(); s = resid_after.std() + 1e-10
    skew = float(((resid_after - mu)**3).mean() / s**3)
    gaussianity = 1.0 / (1.0 + abs(skew))
    return (0.60 * improvement +
            0.20 * gaussianity +
            0.20 * (1.0 - residual_level))

# ─────────────────────────────────────────────
# TESTS AUTONOMES
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# DETECTEUR LOI PUISSANCE ANALYTIQUE (v7d)
# lstsq dans l'espace log : O(N) exact, pas d'Adam
# Détecte y = C * x1^n1 * x2^n2 * x3^n3
# Exemples : gravitation, Coulomb, lois puissance
# ─────────────────────────────────────────────
POWER_LAW_R2_THRESHOLD = 0.97  # seuil R2 pour accepter

def detect_and_fit_power_law(X_np, y_np):
    """
    Teste si (X, y) suit une loi puissance y = C * prod(xj^nj).

    Methode : regression lineaire dans l espace log (lstsq analytique).
    Retourne (pred, coeffs, r2) si R2 > seuil, sinon None.

    Avantages vs Adam :
    - Exact et instantane (pas d iteration)
    - Stable sur toutes les echelles (1e-11 a 1e12)
    - Donne directement les exposants nj
    """
    N, D = X_np.shape

    # Condition : X et y doivent etre tous strictement positifs
    if np.any(X_np <= 0) or np.any(y_np <= 0):
        return None

    try:
        log_X = np.log(X_np)        # (N, D)
        log_y = np.log(y_np)        # (N,)

        # Matrice augmentee : [1, log(x1), log(x2), ..., log(xD)]
        A = np.column_stack([np.ones(N), log_X])   # (N, D+1)

        # Moindres carres : minimise ||A @ coeffs - log_y||^2
        # coeffs[0]   = log(C)   intercept
        # coeffs[1:] = exposants [n1, n2, ..., nD]
        coeffs, _, _, _ = np.linalg.lstsq(A, log_y, rcond=None)

        pred_log = A @ coeffs
        r2 = 1.0 - float(np.var(log_y - pred_log) / (np.var(log_y) + 1e-10))

        if r2 < POWER_LAW_R2_THRESHOLD:
            return None

        # Fix faux positifs (y=ax+b, y=ax2+b) :
        # Pour une vraie loi puissance, le residu log est symetrique
        # Pour une loi avec constante additive, le residu log est asymetrique
        log_resid = log_y - pred_log
        mu_r = log_resid.mean(); s_r = log_resid.std() + 1e-10
        skew_log_resid = abs(float(((log_resid - mu_r)**3).mean() / s_r**3))

        # Seuil : skew > 0.5 → constante additive → faux positif
        if skew_log_resid > 0.5:
            return None

        pred_orig = np.exp(pred_log)
        return pred_orig, coeffs, r2

    except Exception:
        return None

def power_law_bloc_score(resid_before, resid_after, y_orig):
    """Score de qualite specifique power law."""
    std_b = resid_before.std(); std_a = resid_after.std()
    improvement = (std_b - std_a) / (std_b + 1e-10)
    residual_level = std_a / (y_orig.std() + 1e-10)
    mu = resid_after.mean(); s = resid_after.std() + 1e-10
    skew = float(((resid_after - mu)**3).mean() / s**3)
    gaussianity = 1.0 / (1.0 + abs(skew))
    return (0.60 * improvement +
            0.20 * gaussianity +
            0.20 * (1.0 - residual_level))


def gen_law(law, N=400):
    r=lambda a,b:random.uniform(a,b)
    nz=lambda y:y*(1+np.random.normal(0,0.03,len(y)))
    if law=='lin':
        a,b_c=r(.5,5),r(0,10); X=np.random.uniform(.5,10,(N,1))
        y=nz(a*X[:,0]+b_c)
    elif law=='sq':
        a,b_c=r(.3,2),r(0,2); X=np.random.uniform(.5,8,(N,1))
        y=nz(a*X[:,0]**2+b_c)
    elif law=='pair':
        a=r(.5,3)
        X=np.column_stack([np.random.uniform(1,10,N),np.random.uniform(1,8,N)])
        y=nz(a*X[:,0]*X[:,1])
    elif law=='qc':
        a=r(.3,1)
        X=np.column_stack([np.random.uniform(1,5,N),np.random.uniform(1,8,N)])
        y=nz(a*X[:,0]*X[:,1]**2)
    elif law=='trip':
        a=r(.3,2)
        X=np.column_stack([np.random.uniform(1,5,N),np.random.uniform(1,5,N),
                            np.random.uniform(1,5,N)])
        y=nz(a*X[:,0]*X[:,1]*X[:,2])
    elif law=='qinv':
        a=r(.5,2)
        X=np.column_stack([np.random.uniform(2,12,N),np.random.uniform(.5,4,N),
                            np.random.uniform(2,12,N)])
        y=nz(a*X[:,0]*X[:,1]**2/X[:,2])
    elif law=='power':
        A,n=r(.5,3),r(.5,3); X=np.random.uniform(.5,10,(N,1))
        y=nz(A*X[:,0]**n)
    elif law=='exp':
        A,k=r(1,5),r(.1,1); X=np.random.uniform(0,5,(N,1))
        y=nz(A*np.exp(-k*X[:,0]))
    elif law=='log_law':
        A,B=r(1,5),r(0,3); X=np.random.uniform(.5,20,(N,1))
        y=nz(A*np.log(X[:,0])+B)
    elif law=='inv':
        A=r(1,10); X=np.random.uniform(.5,10,(N,1))
        y=nz(A/X[:,0])
    elif law=='sqrt':
        A=r(.5,3); X=np.random.uniform(.1,10,(N,1))
        y=nz(A*np.sqrt(X[:,0]))
    elif law=='gravity':
        G=6.674e-11
        X=np.column_stack([np.random.uniform(1e10,1e12,N),
                            np.random.uniform(1e10,1e12,N),
                            np.random.uniform(1e6,1e8,N)])
        y=nz(G*X[:,0]*X[:,1]/X[:,2]**2)
    elif law=='qinv_power':
        a=r(.5,2)
        X=np.column_stack([np.random.uniform(1,20,N),np.random.uniform(1,20,N)])
        y=nz(a*X[:,0]**2/X[:,1])
    elif law=='sin':
        A=r(1,5); omega=r(0.5,3); phi=r(0,2*math.pi)
        X=np.random.uniform(0,4*math.pi/omega,(N,1))
        y=nz(A*np.sin(omega*X[:,0]+phi))
    elif law=='cos':
        A=r(1,5); omega=r(0.5,3); phi=r(0,2*math.pi)
        X=np.random.uniform(0,4*math.pi/omega,(N,1))
        y=nz(A*np.cos(omega*X[:,0]+phi))
    elif law=='sin_product':
        q=r(1e-9,1e-6); v=r(1e3,1e6); B2=r(0.01,1)
        X=np.column_stack([np.random.uniform(0,math.pi,N),
                            np.random.uniform(1e3,1e6,N)])
        y=nz(q*B2*X[:,1]*np.sin(X[:,0]))
    elif law=='oscillator':
        A=r(1,5); omega=r(0.5,3)
        X=np.random.uniform(0,4*math.pi/omega,(N,1))
        y=nz(A*np.cos(omega*X[:,0]))
    elif law=='interference':
        I1=r(0.5,2); I2=r(0.5,2)
        X=np.random.uniform(0,2*math.pi,(N,1))
        y=nz(I1+I2+2*math.sqrt(I1*I2)*np.cos(X[:,0]))
    elif law=='pair+lin':
        a,b_c=r(.5,3),r(1,4)
        X=np.column_stack([np.random.uniform(1,8,N),np.random.uniform(1,6,N),
                            np.random.uniform(1,8,N)])
        y=nz(a*X[:,0]*X[:,1]+b_c*X[:,2])
    elif law=='qc+trip':
        a,b_c=r(.3,1),r(.5,2)
        X=np.column_stack([np.random.uniform(1,5,N),np.random.uniform(1,8,N),
                            np.random.uniform(1,8,N)])
        y=nz(a*X[:,0]*X[:,1]**2+b_c*X[:,0]*X[:,2])
    elif law=='qc+qinv':
        a,b_c=r(.5,2),r(.5,2)
        X=np.column_stack([np.random.uniform(2,12,N),np.random.uniform(.5,4,N),
                            np.random.uniform(2,12,N)])
        y=nz(a*X[:,0]*X[:,1]**2+b_c*X[:,2]**2/X[:,0])
    elif law=='qc+qc':
        a,b_c=r(.3,1),r(.3,1)
        X=np.column_stack([np.random.uniform(1,5,N),np.random.uniform(1,8,N),
                            np.random.uniform(1,5,N),np.random.uniform(1,5,N)])
        y=nz(a*X[:,0]*X[:,1]**2+b_c*X[:,2]*X[:,3]**2)
    elif law=='lin+qc':
        a,b_c=r(.5,3),r(.3,1)
        X=np.column_stack([np.random.uniform(1,8,N),np.random.uniform(1,5,N),
                            np.random.uniform(1,6,N)])
        y=nz(a*X[:,0]+b_c*X[:,1]*X[:,2]**2)
    elif law=='exp+lin':
        A,k,B=r(1,5),r(.1,.5),r(.5,2)
        X=np.column_stack([np.random.uniform(.5,5,N),np.random.uniform(1,8,N)])
        y=nz(A*np.exp(-k*X[:,0])+B*X[:,1])
    elif law=='sin+lin':
        A=r(1,5); omega=r(0.5,2); B=r(0.5,3)
        X=np.column_stack([np.random.uniform(0,4*math.pi/omega,N),
                            np.random.uniform(1,8,N)])
        y=nz(A*np.sin(omega*X[:,0])+B*X[:,1])
    return X, y

SINGLE   = ['lin','sq','pair','qc','trip','qinv']
NEW_LAWS = ['power','exp','log_law','inv','sqrt','gravity','qinv_power']
TRIG     = ['sin','cos','sin_product','oscillator','interference','sin+lin']
COMBO    = ['pair+lin','qc+trip','qc+qinv','qc+qc','lin+qc','exp+lin']
ALL_LAWS = SINGLE + NEW_LAWS + TRIG + COMBO

MAX_PER_FAMILY = 3500

def build_dataset(n_episodes=15000, N_per_ep=300):
    """
    Génère (points_oracle, stats, labels).
    points_oracle : tenseur (n, N_POINTS, 6)
    stats         : tenseur (n, 56)
    labels        : tenseur (n,)
    """
    all_points = []; all_stats = []; all_labels = []
    family_counts = {i:0 for i in range(N_CLASSES)}
    t0 = time.time()

    for ep in range(n_episodes):
        if ep % 3000 == 0:
            print(f"  Episode {ep}/{n_episodes}  ({time.time()-t0:.0f}s)")
        law = random.choice(ALL_LAWS)
        X, y = gen_law(law, N=N_per_ep)
        resid = y.copy()

        for step in range(5):
            scores,best_config,_ = best_per_family_multispace(X,resid,y)
            stats  = make_state_stats(scores,resid,y,X)
            points = make_oracle_input(X,resid,y,N_POINTS)
            label  = oracle_label(scores,resid,y)

            if family_counts[label] < MAX_PER_FAMILY:
                all_points.append(points)
                all_stats.append(stats)
                all_labels.append(label)
                family_counts[label] += 1

            if label == 8: break
            if label >= len(FAMILIES): break
            fam = FAMILIES[label]
            sc,sp,col,X_t,resid_t = best_config[fam]
            if col is None: break
            resid,_,_ = fit_block(col,resid_t,sp,resid,X_t=X_t,steps=200)

    print(f"  {len(all_labels)} exemples en {time.time()-t0:.1f}s")
    lnames = FAMILIES+['STOP']
    dist = {(lnames[k] if k<len(lnames) else f'cls{k}'):v
            for k,v in sorted(family_counts.items())}
    print(f"  Distribution : {dist}")

    return (torch.tensor(np.array(all_points), dtype=torch.float32),
            torch.tensor(np.array(all_stats),  dtype=torch.float32),
            torch.tensor(all_labels,            dtype=torch.long))

# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train(model, points_t, stats_t, labels_t,
          epochs=40, batch_size=256, lr=3e-3):
    model.to(DEVICE)
    points_t=points_t.to(DEVICE)
    stats_t =stats_t.to(DEVICE)
    labels_t=labels_t.to(DEVICE)
    N=len(labels_t); split=int(0.9*N)
    Pt_tr,St_tr,Lt_tr = points_t[:split],stats_t[:split],labels_t[:split]
    Pt_va,St_va,Lt_va = points_t[split:],stats_t[split:],labels_t[split:]

    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        model.train()
        idx = torch.randperm(len(Lt_tr), device=DEVICE)
        total_loss = 0.0
        for start in range(0, len(Lt_tr), batch_size):
            bi = idx[start:start+batch_size]
            pb,sb,lb = Pt_tr[bi],St_tr[bi],Lt_tr[bi]
            opt.zero_grad()
            loss = crit(model(pb, sb), lb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc=(model(Pt_va,St_va).argmax(dim=1)==Lt_va).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'physiqai_v7.pt')
        if epoch%8==0:
            print(f"  Époque {epoch:3d} | loss={total_loss:.2f} "
                  f"| val_acc={acc*100:.1f}% | best={best_acc*100:.1f}%")

    model.load_state_dict(torch.load('physiqai_v7.pt', weights_only=True))
    print(f"\n  ✓ Meilleur modèle : {best_acc*100:.1f}%")
    return model

# ─────────────────────────────────────────────
# PIPELINE v7
# Fix C : min_score = 0.30 (plus strict)
# ─────────────────────────────────────────────
MIN_SCORE = 0.30  # Fix C

def pipeline(model, X_np, y_np, max_steps=5, n_candidates=4,
             min_score=MIN_SCORE, verbose=False):
    resid_orig=y_np.copy(); acc=np.zeros_like(y_np); chosen=[]

    for step in range(max_steps):
        ratio=resid_orig.std()/(y_np.std()+1e-10)
        if ratio<0.06:
            if verbose: print(f"  → STOP bruit pur"); break

        scores,best_config,_=best_per_family_multispace(X_np,resid_orig,y_np)
        stats  = make_state_stats(scores,resid_orig,y_np,X_np)
        points = make_oracle_input(X_np,resid_orig,y_np,N_POINTS)
        ranked,probs = model.predict_ranked(points, stats)

        if verbose:
            print(f"\n  Étape {step+1} | résidu={ratio:.3f}")
            top=[(FAMILIES[i] if i<len(FAMILIES) else 'STOP',
                  round(float(probs[i]),3)) for i in ranked[:4]]
            print(f"  Oracle → {top}")
            for fam in FAMILIES:
                s,sp,_,_,_=best_config[fam]
                if s>0.70: print(f"    {fam:6s}[{sp}] score={s:.3f}")

        # ── v7d : tester power law analytique EN PREMIER ──
        # Si X et y sont tous positifs et que lstsq trouve R2>0.97
        # on accepte directement sans passer par le classificateur
        if np.all(X_np > 0) and np.all(resid_orig > 0):
            pl_result = detect_and_fit_power_law(X_np, resid_orig)
            if pl_result is not None:
                pred_pl, coeffs_pl, r2_pl = pl_result
                new_resid_pl = resid_orig - pred_pl
                quality_pl = power_law_bloc_score(
                    resid_orig, new_resid_pl, y_np)
                if quality_pl > 0.30:
                    if verbose:
                        exp_str = ' '.join(
                            [f'x{j}^{coeffs_pl[j+1]:.2f}'
                             for j in range(X_np.shape[1])])
                        print(f"  [POWER LAW] R2={r2_pl:.4f} "
                              f"exposants=[{exp_str}] "
                              f"qualite={quality_pl:.3f}")
                    acc += pred_pl
                    resid_orig = new_resid_pl
                    chosen.append(f'power_law[log_log_lstsq]')
                    if quality_pl > 0.80:
                        if verbose:
                            print(f"  -> STOP haute qualite power law")
                        break
                    continue  # passer a l etape suivante

        candidates=[]
        # Actions a tester : top N du classement oracle
        actions_to_test = list(ranked[:n_candidates])

        # Fix sin+lin : forcer lin[original] si score eleve
        # meme si l'oracle le classe loin
        lin_action = FAMILIES.index('lin')
        sc_lin, sp_lin, _, _, _ = best_config['lin']
        if (sp_lin == 'original' and sc_lin >= 0.85
                and lin_action not in actions_to_test):
            actions_to_test.append(lin_action)

        # Forcer pair[original] si score eleve
        pair_action = FAMILIES.index('pair')
        sc_pair, sp_pair, _, _, _ = best_config['pair']
        if (sp_pair == 'original' and sc_pair >= 0.85
                and pair_action not in actions_to_test):
            actions_to_test.append(pair_action)

        for action in actions_to_test:
            if action>=len(FAMILIES): continue
            fam=FAMILIES[action]
            sc,sp,col,X_t,resid_t=best_config[fam]
            if col is None: continue
            # Sprint 7c : estimateur specialise sin/cos
            if fam in ('sin', 'cos'):
                # Extraire omega depuis le nom de l'espace (ex: sin_x0_w2.146)
                try:
                    omega_init = float(sp.split('_w')[-1])
                except:
                    omega_init = None
                # Identifier la variable x concernee
                try:
                    j_var = int(sp.split('_x')[1].split('_')[0])
                    x_var = X_np[:,j_var]
                except:
                    x_var = X_np[:,0]
                new_resid, pred_orig, trig_info = fit_trig(
                    x_var, resid_orig, y_np,
                    omega_candidates=[omega_init] if omega_init else None,
                    try_linear=False,
                    steps=600
                )
                if verbose and trig_info.get('form'):
                    print(f"    [TRIG] {trig_info['form']} "
                          f"omega={trig_info.get('omega',0):.3f} "
                          f"A={trig_info.get('A',0):.2f} "
                          f"phi={trig_info.get('phi',0):.2f}")
            else:
                new_resid,pred_orig,_=fit_block(
                    col,resid_t,sp,resid_orig,X_t=X_t,steps=500)
            sc_after,_,_=best_per_family_multispace(X_np,new_resid,y_np)
            quality=bloc_score(resid_orig,new_resid,y_np,sc_after)
            clf_w=float(probs[action])/(float(probs[ranked[0]])+1e-10)
            combined=quality*(0.6+0.4*clf_w)
            # Bonus confiance haute en espace original
            # sc_orig = score de la famille en espace original (pas log_log)
            # recalcule independamment du meilleur espace detecte
            sc_orig = 0.0
            if fam in ('lin', 'pair', 'qc', 'qinv'):
                orig_col = best_config[fam][2]  # colonne stockee
                orig_sp  = best_config[fam][1]  # espace stocke
                if orig_sp == 'original':
                    sc_orig = sc  # deja en espace original
                else:
                    # Recalculer en espace original si besoin
                    for j in range(X_np.shape[1]):
                        if fam == 'lin':
                            c_orig = X_np[:,j]
                        elif fam == 'sq':
                            c_orig = X_np[:,j]**2
                        else:
                            c_orig = None
                        if c_orig is not None:
                            s_o = abs(corr_np(standardize(c_orig),
                                              standardize(resid_orig)))
                            if s_o > sc_orig: sc_orig = s_o

            # Fix P=V2/R : qinv original >= 0.97 -> x1.5
            if sp == 'original' and sc >= 0.97:
                combined = combined * 1.5
            # Fix sin+lin : lin tres correlé en original -> x1.3
            elif fam == 'lin' and sc_orig >= 0.85:
                combined = combined * 1.3
            # Fix pair dominant : pair tres correlé en original -> x1.2
            elif fam == 'pair' and sc_orig >= 0.85:
                combined = combined * 1.2

            candidates.append({
                'family':fam,'space':sp,'action':action,
                'quality':quality,'combined':combined,
                'new_resid':new_resid,'pred':pred_orig,
            })
            if verbose:
                marker="← oracle#1" if action==ranked[0] else ""
                bonus="[BONUS]" if sp=='original' and sc>=0.97 else ""
                print(f"    {fam:6s}[{sp:20s}]{bonus} "
                      f"qualité={quality:.3f} combined={combined:.3f} {marker}")

        if not candidates:
            if verbose: print(f"  → Aucun candidat valide → STOP")
            break
        best=max(candidates,key=lambda c:c['combined'])
        if best['combined']<min_score:
            if verbose: print(f"  → Score trop faible ({best['combined']:.3f}) → STOP"); break
        acc+=best['pred']; resid_orig=best['new_resid']
        chosen.append(f"{best['family']}[{best['space']}]")
        if verbose:
            print(f"  → CHOIX : {best['family']}[{best['space']}] "
                  f"combined={best['combined']:.3f}")

        # Fix P=V2/R : apres un bloc haute qualite -> STOP immediat
        # Evite les blocs parasites sur un residu deja bien explique
        if best['quality'] > 0.80:
            if verbose:
                print(f"  → STOP haute qualite ({best['quality']:.3f} > 0.80)")
            break

    prec=float(np.mean(np.abs(acc-y_np)/(np.abs(y_np)+1e-8)<0.05))
    return prec,chosen

def evaluate_all(model, n_runs=20, N=500):
    test_cases=[
        ('lin',        "y = a·x + b           "),
        ('sq',         "y = a·x²              "),
        ('pair',       "U = R·I               "),
        ('qc',         "E = ½mv²              "),
        ('trip',       "E = mgh               "),
        ('qinv',       "P = V²/R (ancien)     "),
        ('power',      "y = A·xⁿ  [log-log]   "),
        ('exp',        "y = Ae^(-kt) [log_y]  "),
        ('log_law',    "y = A·log(x) [log_x]  "),
        ('inv',        "y = A/x               "),
        ('sqrt',       "y = A·√x              "),
        ('gravity',    "F=Gm₁m₂/r² [MULTI]    "),
        ('qinv_power', "P=V²/R [log-log]      "),
        ('sin',        "y=A·sin(ωx+φ) [FFT]   "),
        ('cos',        "y=A·cos(ωx+φ) [FFT]   "),
        ('oscillator', "x=A·cos(ωt) [FFT]     "),
        ('interference',"I=I1+I2+2√cos(δ)     "),
        ('sin_product', "F=qvB·sin(θ)         "),
        ('qc+trip',    "E = ½mv² + mgh        "),
        ('qc+qinv',    "P = R·I² + V²/R       "),
        ('exp+lin',    "y = Ae^(-kt) + Bx     "),
        ('sin+lin',    "y = A·sin(ωx) + Bx    "),
    ]
    print(f"\nÉVALUATION v7 — Oracle données brutes ({n_runs} runs, N={N})")
    print("="*68)
    all_p=[]
    for law,name in test_cases:
        precs=[]; blocs_last=[]
        for _ in range(n_runs):
            X,y=gen_law(law,N=N)
            p,blocs=pipeline(model,X,y)
            precs.append(p); blocs_last=blocs
        avg=np.mean(precs); std_p=np.std(precs); all_p.append(avg)
        print(f"{name} {avg*100:5.1f}% ±{std_p*100:.1f}%  {blocs_last[:2]}")
    print(f"\nMoyenne : {np.mean(all_p)*100:.1f}%")

    for law,title in [
        ('qc',      "E = ½mv² — faux positifs résolus ?"),
        ('gravity', "F = Gm₁m₂/r² — oracle données brutes"),
        ('sin',     "y = A·sin(ωx+φ) — FFT + oracle"),
        ('sin+lin', "y = A·sin(ωx) + Bx — combinée"),
    ]:
        print(f"\nDÉTAIL — {title} :")
        X,y=gen_law(law,N=500)
        p,blocs=pipeline(model,X,y,verbose=True)
        print(f"  Précision={p*100:.1f}%  blocs={blocs}")

if __name__=="__main__":
    print("=" * 68)
    print("  PhysiqAI v7d - Power Law analytique + Trig + Fixes")
    print("=" * 68)
    print("Fix : qinv[original] score >= 0.97 -> combined x 1.5")
    print("Charge physiqai_v7.pt sans reentrainer")
    print()

    import os
    if not os.path.exists("physiqai_v7.pt"):
        print("physiqai_v7.pt introuvable - entrainement complet")
        points_t, stats_t, labels_t = build_dataset(n_episodes=15000, N_per_ep=300)
        model = OracleClassifier(n_classes=N_CLASSES)
        model = train(model, points_t, stats_t, labels_t, epochs=40, batch_size=256)
    else:
        print("Chargement physiqai_v7.pt...")
        model = OracleClassifier(n_classes=N_CLASSES)
        model.load_state_dict(torch.load("physiqai_v7.pt", weights_only=True))
        model.to(DEVICE)
        model.eval()
        print("Modele charge.")
        print()

    evaluate_all(model, n_runs=20, N=500)
