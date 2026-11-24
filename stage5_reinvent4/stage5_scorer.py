#!/usr/bin/env python3
"""
REINVENT4 ExternalProcess multi-endpoint scorer for Galectin-3 Stage 5.
Reads SMILES (one per line) from stdin, computes:
  - vinardo: smina Vinardo score at pH 7.4 (kcal/mol, more negative = better; 0.0 on fail/filter)
  - vina:    smina Vina score at pH 7.4 (kcal/mol)
  - hERG, DILI, AMES, HIA: ADMET-AI probabilities
  - polyF_ok: 1.0 if total F <= 3 and no ring has >=3 F, else 0.0 (anti-reward-hacking!)
  - sa_score: RDKit synthetic accessibility score (lower = easier synthesis)
Outputs REINVENT4 ExternalProcess JSON to stdout:
  {"version": 1, "payload": {"vinardo": [...], ...}}
"""
import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from rdkit import Chem
from rdkit.Chem import AllChem, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
except Exception:
    sascorer = None

SMINA = shutil.which("smina") or str(Path.home() / "aidd_tools/bin/smina")
OBABEL = shutil.which("obabel")
RECEPTOR_PDBQT = "/mnt/d/2026@AI_reinvent4/stage5_aniline_sar/gal3_5e89_A.pdbqt"
BOX_CENTER = (17.95, -9.73, 1.52)
BOX_SIZE = (26.0, 22.0, 20.0)
MAX_WORKERS = min(8, os.cpu_count() or 4)


def check_polyF_ok(mol):
    if mol is None:
        return 0.0
    total_f = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F")
    if total_f > 3:
        return 0.0
    # Reject any single aromatic ring with >= 3 fluorines (blocks 2,3,5-triF / tetraF / pentaF)
    for ring in mol.GetRingInfo().AtomRings():
        f_on_ring = 0
        for idx in ring:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetSymbol() == "F":
                    f_on_ring += 1
        if f_on_ring >= 3:
            return 0.0
    return 1.0


def dock_one(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0, 0.0
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 101
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            if AllChem.EmbedMolecule(mol_h, randomSeed=101, useRandomCoords=True) != 0:
                return 0.0, 0.0
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        sdf = td / "lig.sdf"
        pdbqt = td / "lig.pdbqt"
        w = Chem.SDWriter(str(sdf))
        w.write(mol_h)
        w.close()
        try:
            subprocess.run(
                [OBABEL, str(sdf), "-O", str(pdbqt), "-p", "7.4"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
            )
        except Exception:
            return 0.0, 0.0

        def run_score(scoring):
            cmd = [
                SMINA, "-r", RECEPTOR_PDBQT, "-l", str(pdbqt),
                "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]), "--center_z", str(BOX_CENTER[2]),
                "--size_x", str(BOX_SIZE[0]), "--size_y", str(BOX_SIZE[1]), "--size_z", str(BOX_SIZE[2]),
                "--scoring", scoring, "--exhaustiveness", "4", "--num_modes", "1",
                "--seed", "101", "--cpu", "1", "-o", str(td / f"out_{scoring}.sdf")
            ]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=45)
                m = re.search(r"^\s*1\s+(-?\d+\.\d+)", p.stdout, re.M)
                return float(m.group(1)) if m else 0.0
            except Exception:
                return 0.0

        return run_score("vinardo"), run_score("vina")


def main():
    smiles_list = [line.strip() for line in sys.stdin if line.strip()]
    n = len(smiles_list)
    if n == 0:
        print(json.dumps({"version": 1, "payload": {}}))
        return

    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    polyF_ok = [check_polyF_ok(m) for m in mols]
    sa_scores = [
        float(sascorer.calculateScore(m)) if (m is not None and sascorer is not None) else 10.0
        for m in mols
    ]

    # Batch ADMET-AI prediction (redirect any model stderr logs away from stdout!)
    herg = [1.0] * n
    dili = [1.0] * n
    ames = [1.0] * n
    hia = [0.0] * n
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    if valid_idx:
        try:
            import io, contextlib
            with contextlib.redirect_stdout(sys.stderr):
                from admet_ai import ADMETModel
                model = ADMETModel()
                preds = model.predict(smiles=[smiles_list[i] for i in valid_idx])
            for k, idx in enumerate(valid_idx):
                row = preds.iloc[k]
                herg[idx] = float(row["hERG"])
                dili[idx] = float(row["DILI"])
                ames[idx] = float(row["AMES"])
                hia[idx] = float(row["HIA_Hou"])
        except Exception as exc:
            print(f"[stage5_scorer] ADMET-AI error: {exc}", file=sys.stderr)

    # Parallel smina docking (skip docking if molecule fails polyF or has extreme MW > 600)
    vinardo = [0.0] * n
    vina = [0.0] * n
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(dock_one, smiles_list[i]): i for i in valid_idx}
        for fut, idx in futs.items():
            vn, v = fut.result()
            vinardo[idx] = vn
            vina[idx] = v

    payload = {
        "vinardo": vinardo,
        "vina": vina,
        "hERG": herg,
        "DILI": dili,
        "AMES": ames,
        "HIA": hia,
        "polyF_ok": polyF_ok,
        "sa_score": sa_scores,
    }
    print(json.dumps({"version": 1, "payload": payload}))


if __name__ == "__main__":
    main()
