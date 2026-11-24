#!/usr/bin/env python3
"""
Galectin-3 Stage 5B: High-Throughput Commercial-Aniline, Amide-Linker & Neutral-Acyl Optimization
Tests ~130 curated, synthetically trivial, patent-clean molecules (0 poly-F rings, 0 Michael acceptors)
to break the Stage 5A bottlenecks:
  1. B07 (3-SO2Me) TPSA reduction: left urea -> amide (157 -> 120-145 A^2)
  2. Linker rigidification: C3-urea -> C2-amide / N-benzyl / N-acyl-piperidine (0 Brenk alerts, QED > 0.50!)
  3. Neutral-Acyl hERG-Killer hypothesis: acylating piperidine N1 removes basicity while keeping -COOH
  4. Sterically-shielded nitro (2-Me-5-NO2 / 2-Cl-5-NO2) vs safe EWG (-SO2Me, -CN, -CF3, -OCF3, -Cl/F, pyridines)
Evaluates ALL molecules with:
  - ADMET-AI 1.4.0 (GPU)
  - RDKit QED, SA Score, PAINS, Brenk
  - Parallel smina (Vina & Vinardo, pH 7.4, 2 seeds, in fast Linux /tmp)
  - GNINA v1.3.3 GPU CNNscore & CNNaffinity
  - 4.5 A Gal-3 CRD 7-residue contacts
  - PLIP 3.0.1 3D interaction profiling on Top 25 Pareto winners
"""
import os
import re
import sys
import csv
import json
import time
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
S5A_DIR = BASE_DIR / "stage5_aniline_sar"
OUT_DIR = BASE_DIR / "stage5b_optimization"
POSES_DIR = OUT_DIR / "top_poses"
OUT_DIR.mkdir(parents=True, exist_ok=True)
POSES_DIR.mkdir(parents=True, exist_ok=True)

TMP_WORK = Path("/tmp/gal3_stage5b")
TMP_WORK.mkdir(parents=True, exist_ok=True)

SMINA = shutil.which("smina") or str(Path.home() / "aidd_tools/bin/smina")
GNINA = str(Path.home() / "aidd_tools/bin/gnina-gpu")
OBABEL = shutil.which("obabel")
PLIP_BIN = shutil.which("plip")

RECEPTOR_CLEAN_PDB = TMP_WORK / "gal3_5e89_A_clean.pdb"
RECEPTOR_PDBQT = TMP_WORK / "gal3_5e89_A.pdbqt"

BOX_CENTER = (17.95, -9.73, 1.52)
BOX_SIZE = (26.0, 22.0, 20.0)
SEEDS = [101, 202]
KEY_RES = [144, 158, 162, 174, 181, 184, 186]

# Commercial right-hand anilines (aryl moiety attached to urea NH: Ar_R-NH-C(=O)-NH-pip)
# Format: (code, label, aryl_smiles_with_star_at_NH_attachment)
RIGHT_ANILINES = [
    ("3SO2Me",   "3-(methylsulfonyl)aniline",          "CS(=O)(=O)c1cccc([*:1])c1"),
    ("3CN",      "3-aminobenzonitrile",                "N#Cc1cccc([*:1])c1"),
    ("3F5CN",    "3-amino-5-fluorobenzonitrile",       "N#Cc1cc(F)cc([*:1])c1"),
    ("3CF3",     "3-(trifluoromethyl)aniline",         "FC(F)(F)c1cccc([*:1])c1"),
    ("4F3CF3",   "4-fluoro-3-(trifluoromethyl)aniline","Fc1ccc([*:1])cc1C(F)(F)F"),
    ("3OCF3",    "3-(trifluoromethoxy)aniline",        "FC(F)(F)Oc1cccc([*:1])c1"),
    ("3Cl4F",    "3-chloro-4-fluoroaniline",           "Fc1ccc([*:1])cc1Cl"),
    ("34diCl",   "3,4-dichloroaniline",                "Clc1ccc([*:1])cc1Cl"),
    ("35diCl",   "3,5-dichloroaniline",                "Clc1cc(Cl)cc([*:1])c1"),
    ("6CF3Py3",  "6-(trifluoromethyl)pyridin-3-amine", "FC(F)(F)c1ccc([*:1])cn1"),
    ("6ClPy3",   "6-chloropyridin-3-amine",            "Clc1ccc([*:1])cn1"),
    ("3NO2",     "3-nitroaniline (2F->NO2)",           "[O-][N+](=O)c1cccc([*:1])c1"),
    ("2Me5NO2",  "2-methyl-5-nitroaniline (o-Me NO2)", "Cc1ccc([N+](=O)[O-])cc1[*:1]"),
    ("2Cl5NO2",  "2-chloro-5-nitroaniline (o-Cl NO2)", "Clc1ccc([N+](=O)[O-])cc1[*:1]"),
]

# Left-hand commercial acid building blocks attached to core [*:2]
# Core templates have [*:1] for right-hand aryl and [*:2] for left-hand acid-aryl
LEFT_ACIDS = [
    ("mCOOH",     "isophthalic (3-COOH)",             "O=C(O)c1cccc([*:2])c1"),
    ("pCOOH",     "terephthalic (4-COOH)",            "O=C(O)c1ccc([*:2])cc1"),
    ("4FmCOOH",   "4-fluoroisophthalic (4-F-3-COOH)", "O=C(O)c1cc([*:2])ccc1F"),
]

# 5 Core Linker Architectures (each with [*:1] for Ar_R urea and [*:2] for Ar_L acid)
# Wait: for N-acyl-piperidine (neutral, non-basic!) we test direct benzoyl, phenylacetyl (CH2), hydrocinnamoyl (CH2CH2)
CORES = [
    # (core_code, series_name, core_desc, template_smiles_with_star1_and_star2)
    ("C2amd",   "S5B_C2_amide",      "C2-amide zwitterion (0 Brenk, lower TPSA)",
     "[*:2]C(=O)NCCN1CCC(NC(=O)N[*:1])CC1"),
    ("C3amd",   "S5B_C3_amide",      "C3-amide zwitterion (-12 TPSA vs bis-urea)",
     "[*:2]C(=O)NCCCN1CCC(NC(=O)N[*:1])CC1"),
    ("NBn",     "S5B_N_benzyl",      "Compact N-benzyl zwitterion (MW~440, RotB 5, high QED)",
     "[*:2]CN1CCC(NC(=O)N[*:1])CC1"),
    ("NAc0",    "S5B_neutral_acyl",  "Neutral N-benzoyl-piperidine acid (non-basic hERG-killer!)",
     "[*:2]C(=O)N1CCC(NC(=O)N[*:1])CC1"),
    ("NAc1",    "S5B_neutral_acyl",  "Neutral N-phenylacetyl-piperidine acid (non-basic hERG-killer!)",
     "[*:2]CC(=O)N1CCC(NC(=O)N[*:1])CC1"),
    ("NAc2",    "S5B_neutral_acyl",  "Neutral N-phenylpropionyl-piperidine acid (extended hERG-killer!)",
     "[*:2]CCC(=O)N1CCC(NC(=O)N[*:1])CC1"),
]

def join_at_dummy(mol_a, mol_b, map_num):
    """Join two RDKit Mol objects by fusing the neighbors of dummy atoms with matching atom map number."""
    comb = Chem.CombineMols(mol_a, mol_b)
    rw = Chem.RWMol(comb)
    dummies = [a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == map_num]
    if len(dummies) != 2:
        raise ValueError(f"Expected 2 dummies with map {map_num}, got {len(dummies)}")
    d1, d2 = dummies
    n1 = rw.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
    n2 = rw.GetAtomWithIdx(d2).GetNeighbors()[0].GetIdx()
    rw.AddBond(n1, n2, Chem.BondType.SINGLE)
    for idx in sorted([d1, d2], reverse=True):
        rw.RemoveAtom(idx)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol

def assemble_smiles(core_smi, right_smi, left_smi):
    m_core = Chem.MolFromSmiles(core_smi)
    m_right = Chem.MolFromSmiles(right_smi)
    m_left = Chem.MolFromSmiles(left_smi)
    m1 = join_at_dummy(m_core, m_right, 1)
    m2 = join_at_dummy(m1, m_left, 2)
    return Chem.MolToSmiles(m2)

def build_library():
    lib = [
        # Benchmarks from Stage 5A / clinical refs
        ("TD-139", "ref", "Clinical inhaled ref (Kd 14-68 nM)",
         "C1=CC(=CC(=C1)F)C2=CN(N=N2)[C@H]3[C@H]([C@H](O[C@H]([C@@H]3O)S[C@H]4[C@@H]([C@H]([C@H]([C@H](O4)CO)O)N5C=C(N=N5)C6=CC(=CC=C6)F)O)CO)O"),
        ("GB1107", "ref", "Clinical oral scaffold (Kd 37 nM)",
         "C1=CC(=C(C=C1S[C@@H]2[C@@H]([C@H]([C@H]([C@H](O2)CO)O)N3C=C(N=N3)C4=CC(=C(C(=C4)F)F)F)O)Cl)Cl"),
        ("GB1211", "ref", "Clinical oral lead (Kd 25 nM)",
         "C1=C(C=C(C(=C1F)F)F)C2=CN(N=N2)[C@H]3[C@H]([C@H](O[C@@H]([C@@H]3O)SC4=CC(=CN=C4)Br)CO)O"),
        ("c310_15", "ref_s4", "Stage 4 best reproducible lead (5 F)",
         "O=C(NCCCN1CCC(NC(=O)Nc2cc(F)cc(F)c2F)CC1)Nc1cc(F)c(F)c(C(=O)O)c1"),
        ("S5A_B07_3SO2Me", "ref_s5a", "Stage 5A #1 champion (bis-urea, 5-NH-2-F-BA + 3-SO2Me)",
         "CS(=O)(=O)c1cccc(NC(=O)NC2CCN(CCCNC(=O)Nc3ccc(F)c(C(=O)O)c3)CC2)c1"),
        ("S5A_C03_NO2_CN", "ref_s5a", "Stage 5A 0-F nitro champion (C2 bis-urea, 3-NH-5-NO2-BA + 3-CN)",
         "N#Cc1cccc(NC(=O)NC2CCN(CCNC(=O)Nc3cc(C(=O)O)cc([N+](=O)[O-])c3)CC2)c1"),
    ]

    # Targeted combinatorial matrix (~126 new molecules):
    # 1. C2amd x (mCOOH, pCOOH, 4FmCOOH) x all 14 anilines -> 42? Let's curate cleanly to ~120 high-value molecules!
    # Keep full 14 anilines on the 3 most promising cores (C2amd+4FmCOOH, NBn+mCOOH, NAc1+pCOOH, NAc2+pCOOH, C3amd+4FmCOOH)
    # Plus key left-acid isomer scans (mCOOH vs pCOOH vs 4FmCOOH) on top 8 anilines!
    top8_r = {"3SO2Me", "3CN", "3F5CN", "3CF3", "4F3CF3", "3Cl4F", "6CF3Py3", "3NO2"}
    top6_r = {"3SO2Me", "3CN", "3CF3", "4F3CF3", "3Cl4F", "6CF3Py3"}

    combos = [
        # Series 1: C2-amide zwitterions (0 Brenk Aliphatic_long_chain!)
        ("C2amd", "4FmCOOH", [r for r in RIGHT_ANILINES]),                         # 14
        ("C2amd", "mCOOH",   [r for r in RIGHT_ANILINES if r[0] in top8_r]),       # 8
        ("C2amd", "pCOOH",   [r for r in RIGHT_ANILINES if r[0] in top8_r]),       # 8
        # Series 2: C3-amide zwitterions (direct urea->amide upgrade of B07/B09/B12)
        ("C3amd", "4FmCOOH", [r for r in RIGHT_ANILINES]),                         # 14
        ("C3amd", "pCOOH",   [r for r in RIGHT_ANILINES if r[0] in top6_r]),       # 6
        # Series 3: Compact N-benzyl-COOH zwitterions (MW~440, RotB 5, QED~0.70!)
        ("NBn",   "mCOOH",   [r for r in RIGHT_ANILINES]),                         # 14
        ("NBn",   "pCOOH",   [r for r in RIGHT_ANILINES if r[0] in top6_r]),       # 6
        # Series 4: Neutral N-acyl-piperidine acids (non-basic hERG-killer!)
        ("NAc0",  "mCOOH",   [r for r in RIGHT_ANILINES if r[0] in top8_r]),       # 8
        ("NAc0",  "pCOOH",   [r for r in RIGHT_ANILINES if r[0] in top6_r]),       # 6
        ("NAc1",  "mCOOH",   [r for r in RIGHT_ANILINES]),                         # 14
        ("NAc1",  "pCOOH",   [r for r in RIGHT_ANILINES if r[0] in top8_r]),       # 8
        ("NAc2",  "pCOOH",   [r for r in RIGHT_ANILINES]),                         # 14
    ]

    core_map = {c[0]: c for c in CORES}
    left_map = {l[0]: l for l in LEFT_ACIDS}

    seen_smi = {Chem.MolToSmiles(Chem.MolFromSmiles(x[3])) for x in lib}
    for core_code, left_code, r_list in combos:
        _, series_name, core_desc, core_smi = core_map[core_code]
        _, left_label, left_smi = left_map[left_code]
        for r_code, r_label, r_smi in r_list:
            smi = assemble_smiles(core_smi, r_smi, left_smi)
            if smi in seen_smi:
                continue
            seen_smi.add(smi)
            cid = f"{core_code}_{left_code}_{r_code}"
            # Tag nitro probes clearly in series if nitro
            s_tag = "S5B_nitro_probe" if "NO2" in r_code else series_name
            desc = f"{core_code} | L: {left_label} | R: {r_label}"
            lib.append((cid, s_tag, desc, smi))

    # Also add 6 extra C2-bis-urea 3-SO2Me / 6-CF3-Py3 / 3-Cl-4-F direct B07-rigidified analogs!
    extra = [
        ("C2urea_5NH2FBA_3SO2Me", "S5B_C2_amide", "C2-bis-urea | L: 5-NH-2-F-BA | R: 3-SO2Me (B07 C2 analog)",
         "CS(=O)(=O)c1cccc(NC(=O)NC2CCN(CCNC(=O)Nc3ccc(F)c(C(=O)O)c3)CC2)c1"),
        ("C2urea_3NHBA_3SO2Me", "S5B_C2_amide", "C2-bis-urea | L: 3-NH-BA (0F) | R: 3-SO2Me",
         "CS(=O)(=O)c1cccc(NC(=O)NC2CCN(CCNC(=O)Nc3cccc(C(=O)O)c3)CC2)c1"),
        ("C2urea_3NHBA_3Cl4F", "S5B_C2_amide", "C2-bis-urea | L: 3-NH-BA | R: 3-Cl-4-F (B09 C2 analog)",
         "O=C(NCCN1CCC(NC(=O)Nc2ccc(F)c(Cl)c2)CC1)Nc1cccc(C(=O)O)c1"),
        ("C2urea_3NHBA_6CF3Py3", "S5B_C2_amide", "C2-bis-urea | L: 3-NH-BA | R: 6-CF3-Py3 (B12 C2 analog)",
         "O=C(NCCN1CCC(NC(=O)Nc2ccc(C(F)(F)F)nc2)CC1)Nc1cccc(C(=O)O)c1"),
        ("C2urea_5NH2FBA_3SO2NMe2", "S5B_C2_amide", "C2-bis-urea | L: 5-NH-2-F-BA | R: 3-SO2NMe2-aniline",
         "CN(C)S(=O)(=O)c1cccc(NC(=O)NC2CCN(CCNC(=O)Nc3ccc(F)c(C(=O)O)c3)CC2)c1"),
        ("NAc1_4FmCOOH_3SO2Me", "S5B_neutral_acyl", "Neutral N-acyl(CH2) | L: 4-F-3-COOH | R: 3-SO2Me",
         "CS(=O)(=O)c1cccc(NC(=O)NC2CCN(C(=O)Cc3ccc(F)c(C(=O)O)c3)CC2)c1"),
    ]
    for cid, s_tag, desc, smi in extra:
        can = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        if can not in seen_smi:
            seen_smi.add(can)
            lib.append((cid, s_tag, desc, can))
    return lib

def prepare_receptor():
    shutil.copy2(S5A_DIR / "gal3_5e89_A_clean.pdb", RECEPTOR_CLEAN_PDB)
    shutil.copy2(S5A_DIR / "gal3_5e89_A.pdbqt", RECEPTOR_PDBQT)

def load_prot_atoms():
    atoms = []
    with open(RECEPTOR_CLEAN_PDB) as f:
        for line in f:
            if line.startswith("ATOM"):
                resi = int(line[22:26])
                if resi in KEY_RES:
                    atoms.append((resi, float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return atoms

def parse_first_sdf_and_aff(sdf_path):
    if not sdf_path.exists() or sdf_path.stat().st_size < 100:
        return None, None
    txt = sdf_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<minimizedAffinity>\s*\n\s*(-?\d+\.\d+)", txt)
    first = txt.split("$$$$")[0].rstrip() + "\n$$$$\n"
    return (float(m.group(1)) if m else None), first

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
        if len(parts) >= 4 and parts[3] != "H":
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return pts

def compute_crd_contacts(sdf_text, prot_atoms):
    lig_pts = parse_sdf_coords(sdf_text)
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

def dock_one(r):
    cid = r["id"]
    smi = r["smiles"]
    wdir = TMP_WORK / "dock"
    wdir.mkdir(exist_ok=True)
    pdbqt = wdir / f"{cid}.pdbqt"
    if not pdbqt.exists():
        mol = Chem.MolFromSmiles(smi)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            AllChem.EmbedMolecule(mol_h, randomSeed=42, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=300)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=300)
        sdf_in = wdir / f"{cid}_in.sdf"
        w = Chem.SDWriter(str(sdf_in))
        w.write(mol_h)
        w.close()
        subprocess.run([OBABEL, str(sdf_in), "-O", str(pdbqt), "-p", "7.4"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    res = {}
    for scoring in ("vina", "vinardo"):
        scores = []
        best_val = 999.0
        best_block = ""
        for seed in SEEDS:
            out_sdf = wdir / f"{cid}_{scoring}_s{seed}.sdf"
            val, blk = parse_first_sdf_and_aff(out_sdf)
            if val is None:
                cmd = [
                    SMINA, "-r", str(RECEPTOR_PDBQT), "-l", str(pdbqt),
                    "--center_x", str(BOX_CENTER[0]), "--center_y", str(BOX_CENTER[1]), "--center_z", str(BOX_CENTER[2]),
                    "--size_x", str(BOX_SIZE[0]), "--size_y", str(BOX_SIZE[1]), "--size_z", str(BOX_SIZE[2]),
                    "--scoring", scoring, "--exhaustiveness", "8", "--num_modes", "3",
                    "--seed", str(seed), "--cpu", "2", "-o", str(out_sdf),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                val, blk = parse_first_sdf_and_aff(out_sdf)
            if val is not None:
                scores.append(val)
                if val < best_val:
                    best_val = val
                    best_block = blk
        res[scoring] = (
            round(statistics.mean(scores), 3),
            round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0,
            round(min(scores), 2),
            best_block,
        )
    return cid, res

def run_gnina_score(rank1_sdf):
    cmd = [GNINA, "-r", str(RECEPTOR_CLEAN_PDB), "-l", str(rank1_sdf), "--score_only"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cs = re.search(r"CNNscore:\s*([0-9.]+)", p.stdout)
    ca = re.search(r"CNNaffinity:\s*([0-9.\-]+)", p.stdout)
    return (round(float(cs.group(1)), 3) if cs else None), (round(float(ca.group(1)), 3) if ca else None)

def run_plip_on_pose(cid, pose_sdf):
    """Combine clean receptor PDB + ligand PDB and run PLIP 3.0.1 to extract real H-bonds, hydrophobic, halogen, salt bridges."""
    pdir = TMP_WORK / f"plip_{cid}"
    pdir.mkdir(exist_ok=True)
    lig_pdb = pdir / "lig.pdb"
    cplx_pdb = pdir / "complex.pdb"
    try:
        subprocess.run([OBABEL, str(pose_sdf), "-O", str(lig_pdb)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        prot_lines = [l for l in RECEPTOR_CLEAN_PDB.read_text().splitlines() if l.startswith("ATOM")]
        lig_lines = []
        for l in lig_pdb.read_text().splitlines():
            if l.startswith(("ATOM", "HETATM")):
                # Ensure HETATM, chain Z, resname LIG, resseq 999
                l = "HETATM" + l[6:17] + "LIG Z 999" + l[26:]
                lig_lines.append(l)
        cplx_pdb.write_text("\n".join(prot_lines + ["TER"] + lig_lines + ["END"]) + "\n")
        subprocess.run([PLIP_BIN, "-f", str(cplx_pdb), "-o", str(pdir), "-t", "-q"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        rep = pdir / "report.txt"
        if not rep.exists():
            return []
        txt = rep.read_text(errors="ignore")
        interactions = []
        cur_type = None
        for line in txt.splitlines():
            if line.startswith("**") and line.endswith("**"):
                cur_type = line.strip("*").strip()
            elif cur_type and line.startswith("|") and not line.startswith("| RESNR") and "---" not in line:
                parts = [x.strip() for x in line.strip("|").split("|")]
                if len(parts) >= 3 and parts[0].isdigit():
                    res = f"{parts[1].capitalize()}{parts[0]}"
                    # Find distance column (usually float around 1.5-4.5)
                    dist = ""
                    for p in parts[5:]:
                        if re.match(r"^\d+\.\d+$", p):
                            dist = p
                            break
                    interactions.append({"type": cur_type, "res": res, "dist": dist})
        return interactions
    except Exception as e:
        return []

def compute_mpo(r):
    """Multi-Parameter Optimization (MPO) score (0..1, higher is better) balancing affinity, safety, oral PK & synthesis."""
    def ramp(x, lo, hi, higher_better=True):
        if x is None: return 0.0
        if higher_better:
            return max(0.0, min(1.0, (x - lo) / (hi - lo)))
        return max(0.0, min(1.0, (hi - x) / (hi - lo)))
    # Affinity consensus (Vina + Vinardo + GNINA CNNaff)
    s_vina = ramp(r["Vina_mean"], -9.0, -6.5, higher_better=False)
    s_vn   = ramp(r["Vinardo_mean"], -8.5, -5.5, higher_better=False)
    s_cnn  = ramp(r["GNINA_CNNaff"], 3.5, 5.5, higher_better=True)
    # ADMET safety & drug-likeness
    s_herg = ramp(r["hERG"], 0.10, 0.65, higher_better=False)
    s_dili = ramp(r["DILI"], 0.30, 0.85, higher_better=False)
    s_ames = ramp(r["AMES"], 0.08, 0.45, higher_better=False)
    s_qed  = ramp(r["QED"],  0.20, 0.70, higher_better=True)
    s_sa   = ramp(r["SA"],   1.8,  3.8,  higher_better=False)
    s_crd  = ramp(r["CRD_contacts_n"], 3, 6, higher_better=True)
    # Penalize Brenk nitro/Michael alerts
    alert_pen = 0.70 if ("nitro" in r["BRENK"] or "Michael" in r["BRENK"]) else 1.0
    score = (0.20*s_vina + 0.18*s_vn + 0.10*s_cnn + 0.14*s_herg + 0.10*s_dili + 0.10*s_ames + 0.08*s_qed + 0.05*s_sa + 0.05*s_crd) * alert_pen
    return round(score, 4)

def main():
    t0 = time.time()
    prepare_receptor()
    lib = build_library()
    print(f"Built Stage 5B library: {len(lib)} unique molecules", flush=True)

    p_pains = FilterCatalog.FilterCatalogParams()
    p_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    cat_pains = FilterCatalog.FilterCatalog(p_pains)
    p_brenk = FilterCatalog.FilterCatalogParams()
    p_brenk.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    cat_brenk = FilterCatalog.FilterCatalog(p_brenk)

    rows = []
    smiles_list = []
    for cid, series, desc, smi in lib:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Invalid SMILES {cid}: {smi}"
        can = Chem.MolToSmiles(mol)
        smiles_list.append(can)
        pains = [e.GetDescription() for e in cat_pains.GetMatches(mol)]
        brenk = [e.GetDescription() for e in cat_brenk.GetMatches(mol)]
        rows.append({
            "id": cid, "series": series, "desc": desc, "smiles": can,
            "MW": round(Descriptors.MolWt(mol), 1),
            "cLogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 1),
            "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "nF": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "F"),
            "QED": round(QED.qed(mol), 3),
            "SA": round(sascorer.calculateScore(mol), 2) if HAS_SA else None,
            "PAINS": ";".join(pains),
            "BRENK": ";".join(brenk),
        })

    print(f"Running ADMET-AI 1.4.0 on {len(rows)} molecules (GPU)...", flush=True)
    from admet_ai import ADMETModel
    model = ADMETModel()
    preds = model.predict(smiles=smiles_list)
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

    print("Running parallel smina (6 workers x 2 CPUs in fast /tmp)...", flush=True)
    row_by_id = {r["id"]: r for r in rows}
    best_blocks = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(dock_one, r): r["id"] for r in rows}
        done = 0
        for fut in as_completed(futs):
            cid, res = fut.result()
            done += 1
            r = row_by_id[cid]
            r["Vina_mean"], r["Vina_sd"], r["Vina_best"], vina_blk = res["vina"]
            r["Vinardo_mean"], r["Vinardo_sd"], r["Vinardo_best"], vn_blk = res["vinardo"]
            # Use Vinardo pose if strong, else Vina pose; actually keep Vinardo best for consistency with Stage 5A
            best_blocks[cid] = vn_blk if vn_blk else vina_blk
            if done % 15 == 0 or done == len(rows):
                print(f"  Docked [{done:03d}/{len(rows)}] ({time.time()-t0:.0f}s) last={cid} Vina={r['Vina_mean']} Vinardo={r['Vinardo_mean']}", flush=True)

    print("Running GNINA GPU CNN rescoring & CRD 7-residue contact mapping...", flush=True)
    prot_atoms = load_prot_atoms()
    for r in rows:
        cid = r["id"]
        pose_file = POSES_DIR / f"{cid}_rank1.sdf"
        pose_file.write_text(best_blocks[cid], encoding="utf-8")
        cs, ca = run_gnina_score(pose_file)
        contacts = compute_crd_contacts(best_blocks[cid], prot_atoms)
        r["GNINA_CNNscore"] = cs
        r["GNINA_CNNaff"] = ca
        r["CRD_contacts_n"] = len(contacts)
        r["CRD_contacts"] = ",".join(map(str, contacts))
        r["best_pose_sdf"] = str(pose_file)
        r["MPO"] = compute_mpo(r)

    # Sort by MPO descending and run PLIP on Top 30 + all refs
    rows.sort(key=lambda x: x["MPO"], reverse=True)
    plip_targets = set(r["id"] for r in rows[:30]) | {r["id"] for r in rows if r["series"].startswith("ref")}
    print(f"Running PLIP 3.0.1 on {len(plip_targets)} top/reference poses...", flush=True)
    plip_map = {}
    for r in rows:
        if r["id"] in plip_targets:
            ints = run_plip_on_pose(r["id"], r["best_pose_sdf"])
            plip_map[r["id"]] = ints
            hb_n = sum(1 for x in ints if "Hydrogen" in x["type"])
            r["PLIP_Hbonds"] = hb_n
            r["PLIP_summary"] = "; ".join(f"{x['type'][:4]}:{x['res']}({x['dist']})" for x in ints[:8])
        else:
            r["PLIP_Hbonds"] = ""
            r["PLIP_summary"] = ""

    out_csv = OUT_DIR / "stage5b_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT_DIR / "stage5b_plip.json").write_text(json.dumps(plip_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== STAGE 5B COMPLETE in {time.time()-t0:.1f}s! TOP 15 BY MPO ===", flush=True)
    for i, r in enumerate(rows[:15], 1):
        print(f"#{i:02d} {r['id']:26s} | MPO={r['MPO']:.3f} | Vina={r['Vina_mean']:6.2f} Vn={r['Vinardo_mean']:6.2f} CNN={r['GNINA_CNNaff']:.2f} | hERG={r['hERG']:.2f} DILI={r['DILI']:.2f} AMES={r['AMES']:.2f} QED={r['QED']:.2f} SA={r['SA']:.2f} TPSA={r['TPSA']:5.1f} CRD={r['CRD_contacts_n']}/7 | {r['BRENK'] or 'CLEAN'}", flush=True)

if __name__ == "__main__":
    main()
