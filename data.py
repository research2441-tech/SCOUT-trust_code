"""Data loading, de-duplication and leakage-safe leave-one-family-out partitioning
for TON_IoT (network release) and CICIoT2023 (subsample of the public test split)."""
import os
import numpy as np, pandas as pd

DATA = os.environ.get('SCOUT_DATA', 'data')
TEXT_DROP = ['src_ip', 'dst_ip', 'src_port', 'dns_query', 'ssl_subject', 'ssl_issuer',
             'http_uri', 'http_user_agent', 'weird_addl']


def slog(X, cols):
    for c in cols:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X


def load_ton():
    d = pd.read_csv(f'{DATA}/TON_IoT_Train_Test_Network.csv', low_memory=False)
    y = d['type'].astype(str).values
    X = d.drop(columns=TEXT_DROP + ['label', 'type'])
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype('category').cat.codes.astype(np.float32)
    X = X.apply(pd.to_numeric, errors='coerce').fillna(-1).astype(np.float64)
    X = slog(X, ['duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 'src_pkts', 'src_ip_bytes',
                 'dst_pkts', 'dst_ip_bytes', 'http_request_body_len', 'http_response_body_len'])
    return finish(X.astype(np.float32), y, 'normal', 'TON_IoT')


def load_cic():
    d = pd.read_parquet(f'{DATA}/ciciot2023_subsample_test.parquet')
    y = d['attack_class'].astype(str).values
    X = d.drop(columns=['Label', 'Label_orig', 'attack_class', 'label']).astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = slog(X, ['flow_duration', 'Header_Length', 'Rate', 'Srate', 'Drate', 'Tot sum', 'Max', 'AVG',
                 'Std', 'Tot size', 'IAT', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Min',
                 'ack_count', 'syn_count', 'fin_count', 'urg_count', 'rst_count', 'Number', 'Weight', 'Duration'])
    return finish(X.astype(np.float32), y, 'Benign', 'CICIoT2023')


def finish(X, y, benign, name):
    """Drop exact duplicate feature vectors and vectors that carry more than one label (leakage and label noise)."""
    key = pd.util.hash_pandas_object(X, index=False).values
    df = pd.DataFrame({'k': key, 'y': y})
    nlab = df.groupby('k')['y'].nunique()
    amb = set(nlab[nlab > 1].index)
    keep = ~df.duplicated('k').values & ~df['k'].isin(amb).values
    info = dict(n_raw=len(y), n_dup=int(df.duplicated('k').sum()), n_ambiguous=len(amb), n_kept=int(keep.sum()),
                n_feat=X.shape[1])
    X = X.loc[keep].reset_index(drop=True); y = y[keep]
    return dict(name=name, X=X.values, cols=list(X.columns), y=y, benign=benign,
                families=sorted(set(y) - {benign}), info=info)


LOADERS = {'TON_IoT': load_ton, 'CICIoT2023': load_cic}

# benign: train / rank-reference ('fit') / calibration / validation / test
B_FRAC = (0.40, 0.15, 0.15, 0.10, 0.20)
# known attack families: train / validation / test ; the held-out family is test-only
K_FRAC = (0.50, 0.20, 0.30)
K_CAP = 8000          # cap on training rows per known family (balance and runtime)


def lofo_partition(D, held, seed):
    rng = np.random.default_rng(seed)
    y = D['y']; b = rng.permutation(np.where(y == D['benign'])[0])
    cuts = np.cumsum([int(f * len(b)) for f in B_FRAC[:-1]])
    btr, bfit, bcal, bval, bte = np.split(b, cuts)
    P = dict(train=[btr], fit=bfit, cal=bcal, val={'benign': bval}, test={'benign': bte}, held=held)
    for f in D['families']:
        r = rng.permutation(np.where(y == f)[0])
        if f == held:
            P['test'][f] = r
            continue
        k1 = int(K_FRAC[0] * len(r)); k2 = k1 + int(K_FRAC[1] * len(r))
        P['train'].append(r[:min(k1, K_CAP)]); P['val'][f] = r[k1:k2]; P['test'][f] = r[k2:]
    P['train'] = np.concatenate(P['train'])
    return P


if __name__ == '__main__':
    for n, f in LOADERS.items():
        D = f(); print(n, D['X'].shape, D['info'])
        print(pd.Series(D['y']).value_counts().to_dict())
