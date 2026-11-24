#!/usr/bin/env python3
"""
Galectin-3 Stage 5 Pilot Benchmark (Resumable + Parallel + GNINA GPU CNN Scoring):
Evaluates 2xF -> 1xNO2 and commercially available phenylamine (aniline) / EWG bioisostere
substitutions on the reproducible Stage 4 lead c310_15 using:
  1. RDKit physicochemical descriptors, QED, SA Score, PAINS/Brenk alerts
  2. ADMET-AI (exact same ADMET predictor as gal3_explorer.html)
  3. smina (AutoDock Vina & Vinardo, 3 seeds, pH 7.4) — resumes completed SDFs & runs remaining in parallel
  4. gnina-gpu v1.3.3 CNN scoring (CNNscore & CNNaffinity) on the best Vinardo pose
  5. 4.5 A Gal-3 CRD 7-residue contact analysis on the best Vinardo pose
"""
import os
import re
import sys
import csv
import math
import json
import shutil
import subprocess
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors, FilterCatalog, RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
try:
    import sascorer
    HAS_SA = True
except Exception:
    HAS_SA = False

BASE_DIR = Path("/mnt/d/2026@AI_reinvent4")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "stage5_aniline_sar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SMINA = shutil.which("smina") or str(Path.home() / "aidd_tools/bin/smina")
GNINA = str(Path.home() / "aidd_tools/bin/gnina-gpu")
OBABEL = shutil.which("obabel")

RECEPTOR_PDB = DATA_DIR / "gal3_5e89_A.pdb"
RECEPTOR_CLEAN_PDB = OUT_DIR / "gal3_5e89_A_clean.pdb"
RECEPTOR_PDBQT = OUT_DIR / "gal3_5e89_A.pdbqt"

BOX_CENTER = (17.95, -9.73, 1.52)
BOX_SIZE = (26.0, 22.0, 20.0)
SEEDS = [101, 202, 303]
KEY_RES = [144, 158, 162, 174, 181, 184, 186]

CANDIDATES = [
    # --- Controls ---
    ("TD-139", "ref", "TD-139 (clinical inhaled, Kd 14-68 nM)",
     "C1=CC(=CC(=C1)F)C2=CN(N=N2)[C@H]3[C@H]([C@H](O[C@H]([C@@H]3O)S[C@H]4[C@@H]([C@H]([C@H]([C@H](O4)CO)O)N5C=C(N=N5)C6=CC(=CC=C6)F)O)CO)O"),
    ("GB1107", "ref", "GB1107 (oral clinical scaffold, Kd 37 nM)",
     "C1=CC(=C(C=C1S[C@@H]2[C@@H]([C@H]([C@H]([C@H](O2)CO)O)N3C=C(N=N3)C4=CC(=C(C(=C4)F)F)F)O)Cl)Cl"),
    ("GB1211", "ref", "GB1211 (oral clinical lead, Kd 25 nM)",
     "C1=C(C=C(C(=C1F)F)F)C2=CN(N=N2)[C@H]3[C@H]([C@H](O[C@@H]([C@@H]3O)SC4=CC(=CN=C4)Br)CO)O"),
    ("c310_15", "leader", "Stage4 best reproducible lead (2,3,5-F3 right / 2,3-F2-5-NH-BA left)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cc(F)cc(F)c2F)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("c318_12", "leader", "Stage4 high-variance lead (Michael acceptor cinnamamide)",
     "O=C(/C=C/c1c(C(=O)O)cc(F)c(F)c1F)NCCN1CCC(NC(=O)Nc2cc(F)c(F)cc2F)CC1"),

    # --- Series A: Replace right-hand 2,3,5-trifluoroaniline (keep c310_15 left) ---
    ("A01_R_3NO2", "A_nitro", "Right: 3-nitroaniline (2F->1NO2)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc([N+](=O)[O-])c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A02_R_4NO2", "A_nitro", "Right: 4-nitroaniline (2F->1NO2)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc([N+](=O)[O-])cc2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A03_R_3F5NO2", "A_nitro", "Right: 3-fluoro-5-nitroaniline (3F->1F+1NO2)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cc(F)cc([N+](=O)[O-])c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A04_R_4F3NO2", "A_nitro", "Right: 4-fluoro-3-nitroaniline (cheap commercial)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c([N+](=O)[O-])c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A05_R_2F5NO2", "A_nitro", "Right: 2-fluoro-5-nitroaniline (cheap commercial)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cc([N+](=O)[O-])ccc2F)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A06_R_3CN", "A_ewg_safe", "Right: 3-aminobenzonitrile (CN bioisostere)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C#N)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A07_R_4CN", "A_ewg_safe", "Right: 4-aminobenzonitrile (CN bioisostere)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(C#N)cc2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A08_R_3F5CN", "A_ewg_safe", "Right: 3-amino-5-fluorobenzonitrile",
     "O=C(NCCCN1CCC(NC(=O)Nc2cc(F)cc(C#N)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A09_R_3CF3", "A_ewg_safe", "Right: 3-(trifluoromethyl)aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C(F)(F)F)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A10_R_4F3CF3", "A_ewg_safe", "Right: 4-fluoro-3-(trifluoromethyl)aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A11_R_3SO2Me", "A_ewg_safe", "Right: 3-(methylsulfonyl)aniline (sigma~NO2, AMES-safe)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(S(C)(=O)=O)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A12_R_4SO2Me", "A_ewg_safe", "Right: 4-(methylsulfonyl)aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(S(C)(=O)=O)cc2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A13_R_3Cl4F", "A_halogen", "Right: 3-chloro-4-fluoroaniline (bulk commodity)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(Cl)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A14_R_34diCl", "A_halogen", "Right: 3,4-dichloroaniline (GB1107 motif)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(Cl)c(Cl)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A15_R_34diF", "A_halogen", "Right: 3,4-difluoroaniline (cheap 2F)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(F)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A16_R_6ClPy3", "A_heteroaryl", "Right: 6-chloropyridin-3-amine (aza-bioisostere)",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(Cl)nc2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A17_R_6CF3Py3", "A_heteroaryl", "Right: 6-(trifluoromethyl)pyridin-3-amine",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(C(F)(F)F)nc2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A18_R_5CNPy3", "A_heteroaryl", "Right: 5-aminonicotinonitrile (zero F!)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cncc(C#N)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
    ("A19_R_3CONH2", "A_polar", "Right: 3-aminobenzamide",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C(N)=O)c2)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),

    # --- Series B: Replace BOTH rings with cheap commercial building blocks ---
    ("B01_L3NHBA_R3NO2", "B_dual_comm", "L: 3-aminobenzoic acid (0F) + R: 3-NO2-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc([N+](=O)[O-])c2)CC1)Nc1cccc(C(=O)O)c1"),
    ("B02_L3NH5NO2_R3CN", "B_dual_comm", "L: 3-NH-5-NO2-BA (2F->NO2) + R: 3-CN-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C#N)c2)CC1)Nc1cc([N+](=O)[O-])cc(C(=O)O)c1"),
    ("B03_L3NH5NO2_R4F3CF3", "B_dual_comm", "L: 3-NH-5-NO2-BA + R: 4-F-3-CF3-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1cc([N+](=O)[O-])cc(C(=O)O)c1"),
    ("B04_L5NH2FBA_R3CN", "B_dual_comm", "L: 5-NH-2-F-BA (cheap) + R: 3-CN-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C#N)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("B05_L5NH2FBA_R3F5CN", "B_dual_comm", "L: 5-NH-2-F-BA + R: 3-NH-5-F-benzonitrile",
     "O=C(NCCCN1CCC(NC(=O)Nc2cc(F)cc(C#N)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("B06_L5NH2FBA_R4F3CF3", "B_dual_comm", "L: 5-NH-2-F-BA + R: 4-F-3-CF3-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("B07_L5NH2FBA_R3SO2Me", "B_dual_comm", "L: 5-NH-2-F-BA + R: 3-SO2Me-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(S(C)(=O)=O)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("B08_L3NHBA_R4F3CF3", "B_dual_comm", "L: 3-aminobenzoic acid + R: 4-F-3-CF3-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1cccc(C(=O)O)c1"),
    ("B09_L3NHBA_R3Cl4F", "B_dual_comm", "L: 3-aminobenzoic acid + R: 3-Cl-4-F-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(Cl)c2)CC1)Nc1cccc(C(=O)O)c1"),
    ("B10_L3NH5CN_R4F3CF3", "B_dual_comm", "L: 3-NH-5-CN-BA + R: 4-F-3-CF3-aniline",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1cc(C#N)cc(C(=O)O)c1"),
    ("B11_L5NH2ClBA_R3CN", "B_dual_comm", "L: 5-NH-2-Cl-BA + R: 3-CN-aniline (0 F!)",
     "O=C(NCCCN1CCC(NC(=O)Nc2cccc(C#N)c2)CC1)Nc1ccc(Cl)c(C(=O)O)c1"),
    ("B12_L3NHBA_R6CF3Py3", "B_dual_comm", "L: 3-aminobenzoic acid + R: 6-CF3-pyridin-3-amine",
     "O=C(NCCCN1CCC(NC(=O)Nc2ccc(C(F)(F)F)nc2)CC1)Nc1cccc(C(=O)O)c1"),

    # --- Series C: Shorter / rigidified linker (fewer RotB, better QED) ---
    ("C01_C2_L5NH2FBA_R3F5CN", "C_rigid", "C2 linker + 5-NH-2-F-BA + 3-F-5-CN-aniline",
     "O=C(NCCN1CCC(NC(=O)Nc2cc(F)cc(C#N)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("C02_C2_L5NH2FBA_R4F3CF3", "C_rigid", "C2 linker + 5-NH-2-F-BA + 4-F-3-CF3-aniline",
     "O=C(NCCN1CCC(NC(=O)Nc2ccc(F)c(C(F)(F)F)c2)CC1)Nc1ccc(F)c(C(=O)O)c1"),
    ("C03_C2_L3NH5NO2_R3CN", "C_rigid", "C2 linker + 3-NH-5-NO2-BA + 3-CN-aniline",
     "O=C(NCCN1CCC(NC(=O)Nc2cccc(C#N)c2)CC1)Nc1cc([N+](=O)[O-])cc(C(=O)O)c1"),
    ("C04_amide_4CBA_R4F3CF3", "C_rigid", "Terephthalic monoamide + C3 + 4-F-3-CF3-aniline",
     "O=C(O)c1ccc(C(=O)NCCCN2CCC(NC(=O)Nc3ccc(F)c(C(F)(F)F)c3)CC2)cc1"),
]

def load_prot_atoms():
    atoms = []
    with open(RECEPTOR_CLEAN_PDB) as f:
        for line in f:
            if line.startswith("ATOM"):
                resi = int(line[22:26])
                if resi in KEY_RES:
                    atoms.append((resi, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms

def extract_first_sdf_block(sdf_path):
    txt = Path(sdf_path).read_text(encoding="utf-8", errors="ignore")
    first = txt.split("$$$$")[0].rstrip() + "\n$$$$\n"
    return first

def parse_sdf_coords(sdf_text):
    lines = sdf_text.splitlines()
    if len(lines) < 5:
        return []
    try:
        natoms = int(lines[3][:3].strip())
    except Exception:
        return []
    pts = []
    for line in lines[4:4+natoms]:
        parts = line.split()
        if len(parts) >= 4:
            elem = parts[3]
            if elem != "H":
                pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return pts

def compute_crd_contacts(sdf_path, prot_atoms):
    lig_pts = parse_sdf_coords(Path(sdf_path).read_text(encoding="utf-8", errors="ignore"))
    if not lig_pts:
        return []
    hit = set()
    for resi, px, py, pz in prot_atoms:
        if resi in hit:
            continue
        for lx, ly, lz in lig_pts:
            if (px-lx)**2 + (py-ly)**2 + (pz-lz)**2 <= 4.5**2:
                hit.add(resi)
                break
    return sorted(hit)

def parse_first_affinity_from_sdf(sdf_path):
    if not sdf_path.exists() or sdf_path.stat().st_size < 100:
        return None
    txt = sdf_path.read_text(errors="ignore")
    m = re.search(r"<minimizedAffinity>\s*\n\s*(-?\d+\.\d+)", txt)
    return float(m.group(1)) if m else None

def smiles_to_pdbqt_pH74(smi, out_pdbqt):
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 100:
        return
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol_h, params) != 0:
        AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol_h, maxIters=500)
    tmp_sdf = out_pdbqt.with_suffix(".sdf")
    w = Chem.SDWriter(str(tmp_sdf))
    w.write(mol_h)
    w.close()
    subprocess.run([OBABEL, str(tmp_sdf), "-O", str(out_pdbqt), "-p", "7.4"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def dock_molecule(r, dock_dir):
    cid = r["id"]
    pdbqt = dock_dir / f"{cid}.pdbqt"
    smiles_to_pdbqt_pH74(r["smiles"], pdbqt)

    out = {}
    for scoring in ("vina", "vinardo"):
        scores = []
        best_val = 999.0
        best_sdf = None
        for seed in SEEDS:
            out_sdf = dock_dir / f"{cid}_{scoring}_s{seed}.sdf"
            val = parse_first_affinity_from_sdf(out_sdf)
            if val is None:
                cmd = [
                    SMINA, "-r", str(RECEPTOR_PDBQT), "-l", str(pdbqt),
                    "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]), "--center_z", str(BOX_CENTER[2]),
                    "--size_x", str(BOX_SIZE[0]), "--size_y", str(BOX_SIZE[1]), "--size_z", str(BOX_SIZE[2]),
                    "--scoring", scoring, "--exhaustiveness", "8", "--num_modes", "5",
                    "--seed", str(seed), "--cpu", "2", "-o", str(out_sdf),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                val = parse_first_affinity_from_sdf(out_sdf)
            if val is not None:
                scores.append(val)
                if val < best_val:
                    best_val = val
                    best_sdf = out_sdf
        out[scoring] = (
            round(statistics.mean(scores), 3),
            round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
            round(min(scores), 2),
            best_sdf,
        )
    return cid, out

def run_gnina_rescore(best_sdf):
    # Extract rank-1 conformer block directly (avoids RDKit valence sanitization errors on protonated SDF)
    first_block = extract_first_sdf_block(best_sdf)
    tmp_rank1 = best_sdf.with_name(best_sdf.stem + "_rank1.sdf")
    tmp_rank1.write_text(first_block, encoding="utf-8")
    cmd = [GNINA, "-r", str(RECEPTOR_CLEAN_PDB), "-l", str(tmp_rank1), "--score_only"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cnn_score = re.search(r"CNNscore:\s*([0-9.]+)", p.stdout)
    cnn_aff = re.search(r"CNNaffinity:\s*([0-9.\-]+)", p.stdout)
    return (
        round(float(cnn_score.group(1)), 3) if cnn_score else None,
        round(float(cnn_aff.group(1)), 3) if cnn_aff else None,
        str(tmp_rank1),
    )

def main():
    p_pains = FilterCatalog.FilterCatalogParams()
    p_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    cat_pains = FilterCatalog.FilterCatalog(p_pains)
    p_brenk = FilterCatalog.FilterCatalogParams()
    p_brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    cat_brenk = FilterCatalog.FilterCatalog(p_brenk)

    rows = []
    valid_smiles = []
    for cid, series, desc, smi in CANDIDATES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Bad SMILES for {cid}: {smi}"
        can_smi = Chem.MolToSmiles(mol)
        valid_smiles.append(can_smi)
        pains = [e.GetDescription() for e in cat_pains.GetMatches(mol)]
        brenk = [e.GetDescription() for e in cat_brenk.GetMatches(mol)]
        sa = round(sascorer.calculateScore(mol), 2) if HAS_SA else None
        rows.append({
            "id": cid, "series": series, "desc": desc, "smiles": can_smi,
            "MW": round(Descriptors.MolWt(mol), 1),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 1),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "nF": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F"),
            "QED": round(QED.qed(mol), 3),
            "SA": sa,
            "PAINS": ";".join(pains),
            "BRENK": ";".join(brenk),
        })

    print(f"Running ADMET-AI on {len(rows)} molecules...", flush=True)
    from admet_ai import ADMETModel
    model = ADMETModel()
    preds = model.predict(smiles=valid_smiles)
    for i, r in enumerate(rows):
        p = preds.iloc[i]
        r["hERG"] = round(float(p["hERG"]), 3)
        r["DILI"] = round(float(p["DILI"]), 3)
        r["AMES"] = round(float(p["AMES"]), 3)
        r["Pgp"] = round(float(p["Pgp_Broccatelli"]), 3)
        r["BBB"] = round(float(p["BBB_Martins"]), 3)
        r["HIA"] = round(float(p["HIA_Hou"]), 3)
        r["Caco2"] = round(float(p["Caco2_Wang"]), 3)
        r["Bioavail"] = round(float(p["Bioavailability_Ma"]), 3)
        r["Sol"] = round(float(p["Solubility_AqSolDB"]), 2)

    dock_dir = OUT_DIR / "docking"
    dock_dir.mkdir(exist_ok=True)
    row_by_id = {r["id"]: r for r in rows}
    best_sdfs = {}

    print("Running parallel smina (6 workers x 2 CPUs, resuming completed SDFs)...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(dock_molecule, r, dock_dir): r["id"] for r in rows}
        done = 0
        for fut in as_completed(futs):
            cid, res = fut.result()
            done += 1
            r = row_by_id[cid]
            r["Vina_mean"], r["Vina_sd"], r["Vina_best"], _ = res["vina"]
            r["Vinardo_mean"], r["Vinardo_sd"], r["Vinardo_best"], best_vn_sdf = res["vinardo"]
            best_sdfs[cid] = best_vn_sdf
            print(f"[{done:02d}/{len(rows)}] {cid:24s} | Vina={r['Vina_mean']:6.2f}±{r['Vina_sd']:.2f} | Vinardo={r['Vinardo_mean']:6.2f}±{r['Vinardo_sd']:.2f}", flush=True)

    print("Running GNINA GPU CNN rescoring + CRD 7-residue contact mapping...", flush=True)
    prot_atoms = load_prot_atoms()
    for r in rows:
        cid = r["id"]
        best_sdf = best_sdfs[cid]
        cnn_score, cnn_aff, rank1_sdf = run_gnina_rescore(best_sdf)
        contacts = compute_crd_contacts(rank1_sdf, prot_atoms)
        r["GNINA_CNNscore"] = cnn_score
        r["GNINA_CNNaff"] = cnn_aff
        r["CRD_contacts_n"] = len(contacts)
        r["CRD_contacts"] = ",".join(map(str, contacts))
        r["best_pose_sdf"] = rank1_sdf

    out_csv = OUT_DIR / "stage5_aniline_sar_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nDONE! Saved full results to: {out_csv}", flush=True)

if __name__ == "__main__":
    main()
