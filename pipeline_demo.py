"""
Dual-view demo: GNN branch + 3D-CNN branch + concatenation fusion.
Predicts protein-ligand binding affinity (pKd).

Run 1 (pipeline test, fake data, minutes):
    python pipeline_demo.py --data synthetic
Run 2 (real data, PDBbind refined set):
    python pipeline_demo.py --data pdbbind --root /path/to/refined-set --index /path/to/INDEX_refined_data.2020

Outputs: results.csv, scatter.png, and a printed table (R and RMSE per model).
"""
import argparse, os, random, csv
import numpy as np

# ---------------------------------------------------------------- DATA
ELEMS = ['C', 'N', 'O', 'S', 'P']
HAL = ['F', 'CL', 'BR', 'I']
NF = 8  # C,N,O,S,P,halogen,other,is_ligand

def feat(sym, is_lig):
    v = np.zeros(NF, np.float32)
    s = sym.upper()
    if s in [e for e in ELEMS]: v[ELEMS.index(s)] = 1
    elif s in HAL: v[5] = 1
    else: v[6] = 1
    v[7] = 1.0 if is_lig else 0.0
    return v

def parse_pocket_pdb(path):
    xyz, el = [], []
    for l in open(path):
        if l.startswith(('ATOM', 'HETATM')):
            e = l[76:78].strip() or l[12:16].strip()[:1]
            if e.upper() in ('H', 'D'): continue
            xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])]); el.append(e)
    return np.array(xyz, np.float32), el

def parse_sdf(path):
    lines = open(path).read().splitlines()
    n = int(lines[3][0:3]); xyz, el = [], []
    for l in lines[4:4 + n]:
        e = l[31:34].strip()
        if e.upper() == 'H': continue
        xyz.append([float(l[0:10]), float(l[10:20]), float(l[20:30])]); el.append(e)
    return np.array(xyz, np.float32), el

def build_item(pxyz, pel, lxyz, lel, y, max_prot=300):
    cen = lxyz.mean(0)
    order = np.argsort(np.linalg.norm(pxyz - cen, axis=1))[:max_prot]
    pxyz = pxyz[order]; pel = [pel[i] for i in order]
    coords = np.concatenate([pxyz, lxyz]).astype(np.float32)
    feats = np.stack([feat(e, False) for e in pel] + [feat(e, True) for e in lel])
    lig = np.array([False] * len(pel) + [True] * len(lel))
    return dict(coords=coords, feats=feats, lig=lig, y=float(y))

def load_pdbbind(root, index, max_n):
    items = []
    for line in open(index):
        if line.startswith('#') or not line.strip(): continue
        p = line.split(); code = p[0]
        try:
            y = float(p[3])
            px, pe = parse_pocket_pdb(os.path.join(root, code, f'{code}_pocket.pdb'))
            lx, le = parse_sdf(os.path.join(root, code, f'{code}_ligand.sdf'))
            if len(px) == 0 or len(lx) == 0: continue
            items.append(build_item(px, pe, lx, le, y))
        except Exception:
            continue
        if len(items) >= max_n: break
    return items

def make_synthetic(n, seed=0):
    """FAKE complexes with a learnable rule. Only for testing that the code runs."""
    rng = np.random.default_rng(seed); items = []
    for _ in range(n):
        nl, npk = rng.integers(15, 35), rng.integers(150, 250)
        lx = rng.normal(0, 2.0, (nl, 3)).astype(np.float32)
        d = rng.normal(size=(npk, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
        px = (d * rng.uniform(4, 12, (npk, 1))).astype(np.float32)
        le = list(rng.choice(['C', 'N', 'O', 'S'], nl, p=[.6, .15, .2, .05]))
        pe = list(rng.choice(['C', 'N', 'O', 'S'], npk, p=[.6, .15, .2, .05]))
        dist = np.linalg.norm(px[:, None] - lx[None], axis=-1)
        contacts = (dist < 4.5).sum()
        hetero = np.mean([e != 'C' for e in le])
        y = 2.0 + 0.008 * contacts + 2.0 * hetero + rng.normal(0, 0.3)
        items.append(build_item(px, pe, lx, le, y))
    return items

def voxelize(item, rotate=False, size=21):
    c = item['coords'] - item['coords'][item['lig']].mean(0)
    if rotate:
        q, _ = np.linalg.qr(np.random.randn(3, 3)); c = c @ q.T
    idx = np.floor(c + size / 2).astype(int)
    ok = np.all((idx >= 0) & (idx < size), axis=1)
    idx, f = idx[ok], item['feats'][ok]
    g = np.zeros((NF, size, size, size), np.float32)
    for ch in range(NF):
        np.add.at(g[ch], (idx[:, 0], idx[:, 1], idx[:, 2]), f[:, ch])
    return g

# ===== MODELS
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DS(Dataset):
    def __init__(s, items, train): s.items, s.train = items, train
    def __len__(s): return len(s.items)
    def __getitem__(s, i):
        it = s.items[i]
        return it, torch.from_numpy(voxelize(it, rotate=s.train))

def collate(batch):
    B = len(batch); N = max(len(b[0]['coords']) for b in batch)
    co = torch.zeros(B, N, 3); fe = torch.zeros(B, N, NF)
    mask = torch.zeros(B, N, dtype=torch.bool); lig = torch.zeros(B, N, dtype=torch.bool)
    for i, (it, _) in enumerate(batch):
        n = len(it['coords'])
        co[i, :n] = torch.from_numpy(it['coords']); fe[i, :n] = torch.from_numpy(it['feats'])
        mask[i, :n] = True; lig[i, :n] = torch.from_numpy(it['lig'])
    vox = torch.stack([b[1] for b in batch])
    y = torch.tensor([b[0]['y'] for b in batch], dtype=torch.float32)
    return co, fe, mask, lig, vox, y

class GNNEnc(nn.Module):
    """Two-stage message passing: stage 1 bonds only, stage 2 bonds + protein-ligand contacts."""
    def __init__(s, h=64):
        super().__init__()
        s.inp = nn.Linear(NF, h)
        s.s1 = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(2)])
        s.s2 = nn.ModuleList([nn.Linear(2 * h, h) for _ in range(2)])
    def forward(s, co, fe, mask, lig):
        N = co.shape[1]
        D = torch.cdist(co, co)
        valid = mask[:, :, None] & mask[:, None, :]
        same = lig[:, :, None] == lig[:, None, :]
        eye = torch.eye(N, dtype=torch.bool, device=co.device)[None]
        A1 = (D < 1.9) & same & valid & ~eye
        A2 = A1 | ((D < 5.0) & ~same & valid)
        A1 = A1.float(); A2 = A2.float()
        A1 = A1 / A1.sum(-1, keepdim=True).clamp(min=1)
        A2 = A2 / A2.sum(-1, keepdim=True).clamp(min=1)
        h = torch.relu(s.inp(fe))
        for l in s.s1: h = torch.relu(l(torch.cat([h, A1 @ h], -1)))
        for l in s.s2: h = torch.relu(l(torch.cat([h, A2 @ h], -1)))
        lm = lig.float()[..., None]
        return (h * lm).sum(1) / lm.sum(1).clamp(min=1)

class CNNEnc(nn.Module):
    def __init__(s, h=64):
        super().__init__()
        s.net = nn.Sequential(
            nn.Conv3d(NF, 16, 5, padding=2), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Flatten(), nn.Linear(64 * 2 * 2 * 2, h), nn.ReLU())
    def forward(s, vox): return s.net(vox)

class Net(nn.Module):
    def __init__(s, mode):
        super().__init__()
        s.mode = mode
        s.g = GNNEnc() if mode in ('gnn', 'fusion') else None
        s.c = CNNEnc() if mode in ('cnn', 'fusion') else None
        d = 64 * (2 if mode == 'fusion' else 1)
        s.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(s, co, fe, mask, lig, vox):
        z = []
        if s.g is not None: z.append(s.g(co, fe, mask, lig))
        if s.c is not None: z.append(s.c(vox))
        return s.head(torch.cat(z, -1)).squeeze(-1)   # concatenation fusion when both

# ===== TRAIN / EVAL
def run(mode, train_items, test_items, epochs, bs, dev):
    ys = np.array([i['y'] for i in train_items]); mu, sd = ys.mean(), ys.std() + 1e-6
    tr = DataLoader(DS(train_items, True), bs, shuffle=True, collate_fn=collate)
    te = DataLoader(DS(test_items, False), bs, shuffle=False, collate_fn=collate)
    m = Net(mode).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        m.train(); tot = 0
        for co, fe, mask, lig, vox, y in tr:
            co, fe, mask, lig, vox, y = [t.to(dev) for t in (co, fe, mask, lig, vox, y)]
            loss = nn.functional.mse_loss(m(co, fe, mask, lig, vox), (y - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(y)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  [{mode}] epoch {ep + 1}/{epochs}  train loss {tot / len(train_items):.3f}', flush=True)
    m.eval(); P, Y = [], []
    with torch.no_grad():
        for co, fe, mask, lig, vox, y in te:
            co, fe, mask, lig, vox = [t.to(dev) for t in (co, fe, mask, lig, vox)]
            P += (m(co, fe, mask, lig, vox).cpu().numpy() * sd + mu).tolist(); Y += y.tolist()
    P, Y = np.array(P), np.array(Y)
    os.makedirs("models", exist_ok=True)

    torch.save({
        "model_state_dict": m.state_dict(),
        "mu": float(mu),
        "sd": float(sd),
        "mode": mode
    }, f"models/{mode}.pth")

    return float(np.corrcoef(P, Y)[0, 1]), float(np.sqrt(np.mean((P - Y) ** 2))), P, Y

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='synthetic', choices=['synthetic', 'pdbbind'])
    ap.add_argument('--root'); ap.add_argument('--index')
    ap.add_argument('--n', type=int, default=300); ap.add_argument('--max_n', type=int, default=400)
    ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--batch', type=int, default=16)
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    items = make_synthetic(a.n) if a.data == 'synthetic' else load_pdbbind(a.root, a.index, a.max_n)
    print(f'Loaded {len(items)} complexes ({a.data})')
    random.shuffle(items); k = int(0.8 * len(items)); trn, tst = items[:k], items[k:]
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Device:', dev)
    rows, res = [], {}
    for mode in ['gnn', 'cnn', 'fusion']:
        r, rmse, P, Y = run(mode, trn, tst, a.epochs, a.batch, dev)
        res[mode] = (P, Y); rows.append((mode, round(r, 3), round(rmse, 3)))
        with open('results.csv', 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['model', 'pearson_R', 'rmse']); w.writerows(rows)
        print(f'==> {mode}: R={r:.3f}  RMSE={rmse:.3f}', flush=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for x, (mode, (P, Y)) in zip(ax, res.items()):
        x.scatter(Y, P, s=12); lo, hi = min(Y.min(), P.min()), max(Y.max(), P.max())
        x.plot([lo, hi], [lo, hi], 'k--', lw=1); x.set_title(mode); x.set_xlabel('True pKd'); x.set_ylabel('Predicted pKd')
    plt.tight_layout(); plt.savefig('scatter.png', dpi=150)
    print('\nModel   R      RMSE'); [print(f'{m:7s} {r:.3f}  {e:.3f}') for m, r, e in rows]
    if a.data == 'synthetic': print('\nNOTE: synthetic data - this only proves the code runs. Do not report as real results.')
