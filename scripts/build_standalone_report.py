#!/usr/bin/env python3
"""
Builds a completely self-contained standalone HTML report (`docs/index.html`)
combining ALL stages:
  - Stage 1-4 REINVENT4 + DynamicBind + smina + ADMET-AI evolution (93 molecules)
  - Stage 5A Commercial Phenylamine & 2xF->1xNO2 SAR pilot (40 molecules)
  - Stage 5B High-Throughput Amide-Linker, Compact N-Benzyl & Neutral N-Acyl hERG-Killer Optimization (132 molecules + PLIP 3.0.1 + GNINA v1.3.3 GPU CNN)
  - Embedded 3Dmol-min.js (zero external CDN dependencies; works offline & on GitHub Pages)
"""
import io
import re
import csv
import json
import base64
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw

BASE_DIR = Path("/mnt/d/2026@AI_reinvent4")
EXPLORER_HTML = BASE_DIR / "gal3_explorer/gal3_explorer.html"
SAR_CSV = BASE_DIR / "stage5_aniline_sar/stage5_aniline_sar_results.csv"
S5B_CSV = BASE_DIR / "stage5b_optimization/stage5b_results.csv"
S5B_PLIP = BASE_DIR / "stage5b_optimization/stage5b_plip.json"
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS_DIR / "index.html"


def mol_to_png_b64(smiles, size=(420, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def safe_float(v):
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except Exception:
        return None


def main():
    print("Loading gal3_explorer.html payload and 3Dmol-min.js...")
    html_src = EXPLORER_HTML.read_text(encoding="utf-8")
    m_3dmol = re.search(r"(<script>/\*! For license information please see 3Dmol-min\.js.*?</script>)", html_src, re.S)
    assert m_3dmol, "3Dmol script block not found!"
    threemol_script = m_3dmol.group(1)

    m_payload = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html_src, re.S)
    old_data = json.loads(m_payload.group(1))

    # Stage 1-4 cycles + leaders/refs
    cycles = []
    leaders_refs = []
    for m in old_data["molecules"]:
        a = m.get("admet") or {}
        e = m.get("extra") or {}
        s = m.get("smina") or {}
        mol = Chem.MolFromSmiles(m["smiles"])
        nF = sum(1 for at in mol.GetAtoms() if at.GetSymbol() == "F") if mol else 0
        rec = {
            "id": m["id"], "group": m["group"], "cycle": m.get("cycle"),
            "caption": m.get("caption") or "", "smiles": m["smiles"],
            "aff": m.get("aff"), "aff_sd": m.get("aff_sd"),
            "contacts": m.get("contacts") or [],
            "hERG": a.get("hERG"), "DILI": e.get("DILI"), "AMES": e.get("AMES"),
            "Pgp": e.get("Pgp_Broccatelli"), "BBB": a.get("BBB"), "HIA": a.get("HIA"),
            "Caco2": a.get("Caco2"), "Bioavail": e.get("Bioavailability_Ma"),
            "QED": a.get("QED"), "TPSA": a.get("TPSA"), "Sol": a.get("solubility"),
            "cLogP": e.get("cLogP"), "MW": e.get("MW"), "RotB": e.get("RotB"), "nF": nF,
            "vina_old": (s.get("pH74_vina") or {}).get("mean"),
            "vinardo_old": (s.get("pH74_vinardo") or {}).get("mean"),
            "plip": m.get("plip") or [],
        }
        if m["group"] in ("stage1", "stage2", "stage3", "stage4"):
            cycles.append(rec)
        else:
            rec["mol"] = m.get("mol") or ""
            rec["png"] = m.get("png") or ""
            leaders_refs.append(rec)
    cycles.sort(key=lambda x: x["cycle"] or 0)

    # Stage 5A (40 molecules)
    print("Loading Stage 5A (40 molecules)...")
    stage5a = []
    with open(SAR_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sdf_path = Path(row["best_pose_sdf"])
            mol_block = sdf_path.read_text(encoding="utf-8", errors="ignore") if sdf_path.exists() else ""
            contacts = [int(x) for x in row["CRD_contacts"].split(",") if x.strip()]
            stage5a.append({
                "id": row["id"], "series": row["series"], "desc": row["desc"], "smiles": row["smiles"],
                "MW": safe_float(row["MW"]), "cLogP": safe_float(row["cLogP"]), "TPSA": safe_float(row["TPSA"]),
                "RotB": int(float(row["RotB"])), "nF": int(float(row["nF"])),
                "QED": safe_float(row["QED"]), "SA": safe_float(row["SA"]),
                "PAINS": row["PAINS"], "BRENK": row["BRENK"],
                "hERG": safe_float(row["hERG"]), "DILI": safe_float(row["DILI"]), "AMES": safe_float(row["AMES"]),
                "Pgp": safe_float(row["Pgp"]), "BBB": safe_float(row["BBB"]), "HIA": safe_float(row["HIA"]),
                "Caco2": safe_float(row["Caco2"]), "Bioavail": safe_float(row["Bioavail"]), "Sol": safe_float(row["Sol"]),
                "Vina": safe_float(row["Vina_mean"]), "Vina_sd": safe_float(row["Vina_sd"]),
                "Vinardo": safe_float(row["Vinardo_mean"]), "Vinardo_sd": safe_float(row["Vinardo_sd"]),
                "CNNscore": safe_float(row["GNINA_CNNscore"]), "CNNaff": safe_float(row["GNINA_CNNaff"]),
                "contacts": contacts,
                "png": mol_to_png_b64(row["smiles"]),
                "mol": mol_block,
            })

    # Stage 5B (132 molecules + PLIP 3.0.1 + 3D poses & 2D PNGs for Top 40 + all refs + top 6 nitro probes)
    print("Loading Stage 5B (132 molecules + PLIP 3.0.1)...")
    plip_map = json.loads(S5B_PLIP.read_text(encoding="utf-8")) if S5B_PLIP.exists() else {}
    s5b_raw = list(csv.DictReader(open(S5B_CSV, encoding="utf-8")))
    # Decide which Stage 5B molecules get full 3D pose + 2D PNG embedded (keeps HTML sleek ~2.5 MB)
    embed_ids = set(r["id"] for r in s5b_raw[:35])
    embed_ids |= {r["id"] for r in s5b_raw if r["series"].startswith("ref")}
    embed_ids |= {r["id"] for r in sorted([x for x in s5b_raw if x["series"] == "S5B_nitro_probe"],
                                          key=lambda z: float(z["MPO"]), reverse=True)[:5]}
    embed_ids |= {r["id"] for r in sorted([x for x in s5b_raw if x["series"] == "S5B_N_benzyl"],
                                          key=lambda z: float(z["MPO"]), reverse=True)[:5]}

    stage5b = []
    for row in s5b_raw:
        cid = row["id"]
        contacts = [int(x) for x in row["CRD_contacts"].split(",") if x.strip()]
        has3d = cid in embed_ids
        sdf_path = Path(row["best_pose_sdf"])
        mol_block = (sdf_path.read_text(encoding="utf-8", errors="ignore") if (has3d and sdf_path.exists()) else "")
        stage5b.append({
            "id": cid, "series": row["series"], "desc": row["desc"], "smiles": row["smiles"],
            "MPO": safe_float(row["MPO"]),
            "MW": safe_float(row["MW"]), "cLogP": safe_float(row["cLogP"]), "TPSA": safe_float(row["TPSA"]),
            "HBD": int(float(row["HBD"])), "HBA": int(float(row["HBA"])),
            "RotB": int(float(row["RotB"])), "nF": int(float(row["nF"])),
            "QED": safe_float(row["QED"]), "SA": safe_float(row["SA"]),
            "PAINS": row["PAINS"], "BRENK": row["BRENK"],
            "hERG": safe_float(row["hERG"]), "DILI": safe_float(row["DILI"]), "AMES": safe_float(row["AMES"]),
            "Pgp": safe_float(row["Pgp"]), "BBB": safe_float(row["BBB"]), "HIA": safe_float(row["HIA"]),
            "Caco2": safe_float(row["Caco2"]), "Bioavail": safe_float(row["Bioavail"]), "Sol": safe_float(row["Sol"]),
            "Vina": safe_float(row["Vina_mean"]), "Vina_sd": safe_float(row["Vina_sd"]),
            "Vinardo": safe_float(row["Vinardo_mean"]), "Vinardo_sd": safe_float(row["Vinardo_sd"]),
            "CNNscore": safe_float(row["GNINA_CNNscore"]), "CNNaff": safe_float(row["GNINA_CNNaff"]),
            "contacts": contacts,
            "plip": plip_map.get(cid, []),
            "plip_hb": int(row["PLIP_Hbonds"]) if row.get("PLIP_Hbonds") else None,
            "plip_sum": row.get("PLIP_summary") or "",
            "png": mol_to_png_b64(row["smiles"]) if has3d else "",
            "mol": mol_block,
        })

    payload = {
        "protein_pdb": old_data["protein_pdb"],
        "crd": old_data["crd"],
        "cycles": cycles,
        "leaders_refs": leaders_refs,
        "stage5a": stage5a,
        "stage5b": stage5b,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    print(f"Payload JSON size: {len(payload_json)/1e6:.2f} MB (Stage5B 3D embedded: {len(embed_ids)} poses)")

    html_doc = HTML_TEMPLATE.replace("__THREEMOL_SCRIPT__", threemol_script).replace("__PAYLOAD_JSON__", payload_json)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    (BASE_DIR / "gal3_stage5_report.html").write_text(html_doc, encoding="utf-8")
    print(f"Wrote standalone HTML: {OUT_HTML} ({OUT_HTML.stat().st_size/1e6:.2f} MB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galectin-3 AI 藥物設計綜合報告 — Stage 1–4 演化診斷、Stage 5A 市售苯胺與 Stage 5B 突破性領先優化</title>
<style>
:root{
  --bg:#0b101b; --bg2:#101726; --panel:#141d2e; --panel2:#1a253b; --line:#263552;
  --txt:#eaf0fa; --muted:#95a5c2; --gold:#f0c766; --teal:#3ad1be; --blue:#6ea8fe;
  --green:#6fd08c; --orange:#f2a65a; --red:#f06c7a; --purple:#be95ff;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--bg);color:var(--txt);
  font-family:"PingFang TC","Microsoft JhengHei","Noto Sans TC",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:13.5px;line-height:1.55}
a{color:var(--teal);text-decoration:none}
a:hover{text-decoration:underline}
code,pre,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
button,select,input{font-family:inherit;font-size:12.5px;color:var(--txt);background:var(--panel2);
  border:1px solid var(--line);border-radius:7px;padding:6px 11px}
button{cursor:pointer;transition:.15s}
button:hover{background:#233352;border-color:#3f5885}
button.on,button.active{background:#1d4e4a;border-color:var(--teal);color:#d7fff8}

header{background:linear-gradient(135deg,#121c30 0%,#0d1525 58%,#13273a 100%);
  border-bottom:1px solid var(--line);padding:18px 24px 14px}
.hdr-top{display:flex;flex-wrap:wrap;align-items:flex-start;gap:16px;justify-content:space-between}
header h1{margin:0 0 4px;font-size:20px;font-weight:700;letter-spacing:.3px;color:#fff}
header .sub{color:var(--muted);font-size:12.5px}
.warn-banner{background:rgba(92,45,22,.55);border:1px solid #8a4f2d;color:#ffe0c2;
  border-radius:8px;padding:8px 12px;font-size:12px;max-width:640px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:10px;margin-top:14px}
.kpi{background:rgba(20,29,46,.88);border:1px solid var(--line);border-radius:9px;padding:10px 13px}
.kpi .lbl{font-size:11.5px;color:var(--muted)}
.kpi .val{font-size:18px;font-weight:700;margin:2px 0;color:var(--gold);font-variant-numeric:tabular-nums}
.kpi .desc{font-size:11px;color:var(--muted)}

nav.tabs{display:flex;gap:8px;padding:10px 24px;background:var(--bg2);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:20;flex-wrap:wrap}
nav.tabs button{padding:7px 14px;font-weight:600;border-radius:8px}
nav.tabs button.active{background:linear-gradient(135deg,#1d5c55,#21446b);border-color:var(--teal);color:#fff}

.container{max-width:1480px;margin:0 auto;padding:18px 24px 44px}
.tab-pane{display:none}
.tab-pane.active{display:block}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 17px;margin-bottom:16px}
.card h2{margin:0 0 8px;font-size:15.5px;color:var(--gold);display:flex;align-items:center;gap:8px}
.card h3{margin:0 0 7px;font-size:13.5px;color:var(--teal)}
.note{font-size:12px;color:var(--muted);line-height:1.55}

.callout{border-left:3px solid var(--teal);background:rgba(58,209,190,.08);
  padding:10px 13px;border-radius:0 8px 8px 0;margin:10px 0;font-size:12.5px}
.callout.gold{border-left-color:var(--gold);background:rgba(240,199,102,.09)}
.callout.warn{border-left-color:var(--orange);background:rgba(242,166,90,.09)}
.callout.bad{border-left-color:var(--red);background:rgba(240,108,122,.09)}

table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 7px;border-bottom:1px solid rgba(38,53,82,.7);text-align:left;vertical-align:middle}
th{color:var(--muted);font-weight:600;background:rgba(16,23,38,.85);position:sticky;top:0;white-space:nowrap;z-index:2}
th.sort{cursor:pointer;user-select:none}
th.sort:hover{color:#fff}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Consolas,monospace}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:#1d2b45}

.pill{display:inline-block;padding:1.5px 7px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.p-ref{background:#342652;color:#d6bcff;border:1px solid #5d4596}
.p-lead{background:#473816;color:#ffe08c;border:1px solid #8a6c22}
.p-nitro{background:#4f212b;color:#ffb0b9;border:1px solid #963c4d}
.p-safe{background:#18453d;color:#8cf2df;border:1px solid #2b7a6c}
.p-hal{background:#1f3859;color:#a8d0ff;border:1px solid #3c679c}
.p-het{background:#382c59;color:#cebaff;border:1px solid #654fa3}
.p-dual{background:#24452b;color:#a3f0b5;border:1px solid #3f7d4d}
.p-rig{background:#473620;color:#ffd099;border:1px solid #8a6534}
.p-pol{background:#2a3a4a;color:#c2d7eb;border:1px solid #4c647d}
.p-nac{background:#154a42;color:#7dffea;border:1px solid #2fa393}
.p-nbn{background:#20452e;color:#9df5b6;border:1px solid #3f8a5b}
.p-c2a{background:#1f3859;color:#a8d4ff;border:1px solid #3b6ca8}
.p-c3a{background:#3c3520;color:#ffe29c;border:1px solid #7d6d37}

.good{color:#7de8a8;font-weight:600}
.mid{color:#f5ce78}
.bad{color:#ff8f9b;font-weight:600}

.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.scroll-x{overflow-x:auto}
.scroll-y{max-height:560px;overflow-y:auto}

.v-layout{display:grid;grid-template-columns:285px 1fr 380px;gap:12px;height:780px}
@media(max-width:1180px){.v-layout{grid-template-columns:1fr;height:auto}}
.v-side,.v-right{background:var(--panel);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.v-list{overflow-y:auto;flex:1;padding:6px}
.v-grp{margin:8px 0 3px;padding:3px 6px;font-size:11px;color:var(--gold);border-bottom:1px solid var(--line);font-weight:600}
.v-item{display:flex;align-items:center;gap:7px;padding:5px 7px;border-radius:6px;cursor:pointer;font-size:12px}
.v-item:hover{background:#1d2b45}
.v-item.sel{background:#254266;outline:1px solid #4c82bd}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.v-center{background:var(--panel);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;min-height:480px;overflow:hidden}
.v-tools{display:flex;gap:6px;padding:7px 10px;background:var(--bg2);border-bottom:1px solid var(--line);flex-wrap:wrap;align-items:center}
#viewwrap{position:relative;flex:1;min-height:420px}
#viewer3d{position:absolute;inset:0}
#vcap{position:absolute;left:10px;bottom:8px;background:rgba(11,16,27,.82);padding:5px 9px;border-radius:6px;font-size:11px;color:var(--muted);max-width:78%}
.v-right-in{overflow-y:auto;padding:12px;flex:1}
.png2d{width:100%;background:#fff;border-radius:7px;display:block;margin:6px 0}
.smi-box{font-family:ui-monospace,Consolas,monospace;font-size:10.5px;word-break:break-all;background:#0b101b;border:1px solid var(--line);border-radius:6px;padding:6px;color:#b8d0ef}
.tag{display:inline-block;margin:2px 3px 0 0;padding:1px 6px;border-radius:9px;font-size:10.5px;background:#1e2e4a;border:1px solid #324a73;color:#b8cceb}
.tag.hit{background:#4d3c14;border-color:var(--gold);color:#ffe49e}
pre.codeblock{background:#0b101b;border:1px solid var(--line);border-radius:8px;padding:11px 13px;overflow-x:auto;font-size:11.5px;color:#cde0f8;line-height:1.5}
canvas{display:block;width:100%}
</style>
</head>
<body>
<header>
  <div class="hdr-top">
    <div>
      <h1>Galectin-3 AI 藥物設計綜合報告 — Stage 1–4 診斷 × Stage 5A/5B 市售苯胺與去鹼性 hERG-Killer 突破</h1>
      <div class="sub">REINVENT 4.9.34 × DynamicBind × smina (Vina / Vinardo) × GNINA v1.3.3 GPU CNN × PLIP 3.0.1 × ADMET-AI 1.4.0 × PDB 5E89 CRD</div>
    </div>
    <div class="warn-banner">
      <b>科學誠實聲明：</b>所有結合分數（DynamicBind、smina Vina/Vinardo、GNINA CNNaffinity）皆為<b>計算代理值</b>，非實驗 $K_d$。本頁整合 <b>265 個分子</b>（Stage 1–4 歷史 93 分子 + Stage 5A 40 分子 + Stage 5B 132 分子），所有 Stage 5 分子皆經多引擎交叉驗證與 PLIP 3D 作用解析，最終候選仍須體外 FP/SPR 與 Ames/微粒體實驗確認。
    </div>
  </div>
  <div class="kpis" id="kpiBar"></div>
</header>

<nav class="tabs" id="tabNav">
  <button class="active" data-tab="t5b">★ 1. Stage 5B 突破總覽 (132 分子 + PLIP)</button>
  <button data-tab="t5a">2. Stage 5A：市售苯胺 &amp; 2F→NO₂ 實測 (40 分子)</button>
  <button data-tab="t1">3. Stage 1–4 演化診斷與 Reward Hacking (93 分子)</button>
  <button data-tab="t3">4. 互動式 3D 結合姿勢 &amp; PLIP 檢視器</button>
  <button data-tab="t4">5. 專利 FTO、2 步合成路線與本機 REINVENT 4</button>
</nav>

<div class="container">
  <!-- ================= TAB 5B (NEW HERO TAB) ================= -->
  <section class="tab-pane active" id="t5b">
    <div class="card">
      <h2>Stage 5B 兩大藥物化學突破（132 個本機實測分子：零多氟、零手性、2 步廉價市售合成）</h2>
      <div class="grid3">
        <div class="callout gold" style="margin:0">
          <h3>🏆 突破 A：中性 N-Acyl-哌啶羧酸（hERG-Killer！）</h3>
          <div>將哌啶 N1 由鹼性烷基胺改為<b>中性醯胺（NAc1 / NAc2）</b>並保留 `-COOH`，同時實現：<br>
          • <b>hERG 暴跌至 0.04–0.29</b>（均值 0.21，vs GB1107 0.71 / c310_15 0.51）<br>
          • <b>AMES 0.02–0.13</b>、<b>QED 翻 3 倍至 0.58–0.70</b>、<b>SA 2.1–2.3</b><br>
          • <b>#1 <code>NAc1_4FmCOOH_3SO2Me</code></b>：MPO <b>0.811</b>、Vina <b>−8.54</b>、Vinardo <b>−8.42</b>、CNNaff <b>5.53</b>、PLIP 確認 <b>Trp181 平行 π 堆積 (3.88 Å) + Asn174 氫鍵 (1.87 Å) + His158/Arg162 鹽橋</b>！</div>
        </div>
        <div class="callout" style="margin:0">
          <h3>💎 突破 B：緊緻 N-Benzyl 兩性離子（全場最低 DILI！）</h3>
          <div>用市售 <b>3-/4-甲醯基苯甲酸</b>還原胺化接上 4-氨基哌啶（<code>NBn</code> 系列，MW ~390–430，RotB 5–6）：<br>
          • <b>DILI 均值降至 0.48</b>（全系列最低！遠勝臨床對照 0.94 與 NAc 0.85）<br>
          • <b>QED 均值 0.68（最高）</b>、<b>SA 均值 2.10（最易合成）</b>、<b>口服生體可用率 0.79</b><br>
          • 代表：<b><code>NBn_pCOOH_3SO2Me</code></b>（0 F、hERG 0.15、DILI 0.49、AMES 0.06、SA 2.04）與 <b><code>NBn_mCOOH_6ClPy3</code></b>（PLIP <b>7 條核心氫鍵</b> + Trp181 π 堆積）！</div>
        </div>
        <div class="callout warn" style="margin:0">
          <h3>🔬 2×F → 1×NO₂ 新發現：骨架決定硝基致突變風險</h3>
          <div>在 Stage 5A 雙脲骨架上，<code>-NO₂</code> 使 AMES 飆至 0.41–0.58；但在 <b>Stage 5B 中性 <code>NAc1</code> 骨架</b>上，零氟 <b><code>NAc1_mCOOH_3NO2</code></b> 達到 Vina −7.76 / Vinardo −8.19 / 6/7 CRD / hERG 0.134，且 <b>ADMET-AI AMES 降至 0.159</b>！<br>
          • 但 <code>-NO₂</code> 仍帶 Brenk <code>nitro_group</code> 警示與較高 DILI (0.90)，因此 <b><code>3-SO₂Me</code> / <code>3-CF₃</code> / <code>3-OCF₃</code> / <code>3-Cl-4-F</code> / <code>6-CF₃-吡啶</code> / <code>3-CN</code></b> 在結合力與安全性上依然全面領先。</div>
        </div>
      </div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>Stage 5B 132 分子互動 Pareto 散佈圖（點擊任一點直達 3D + PLIP）</h2>
        <div class="toolbar">
          <label class="note">X 軸：</label>
          <select id="b5ScX">
            <option value="MPO" selected>綜合 MPO 分數 (越高越好)</option>
            <option value="Vina">smina Vina (越負越好)</option>
            <option value="Vinardo">smina Vinardo (越負越好)</option>
            <option value="CNNaff">GNINA GPU CNNaff (越高越好)</option>
            <option value="QED">QED 類藥性 (越高越好)</option>
          </select>
          <label class="note">Y 軸：</label>
          <select id="b5ScY">
            <option value="hERG" selected>hERG 風險 (越低越好)</option>
            <option value="DILI">DILI 肝毒 (越低越好)</option>
            <option value="AMES">AMES 致突變 (越低越好)</option>
            <option value="CNNaff">GNINA CNNaff (越高越好)</option>
            <option value="Bioavail">口服生體可用率 (越高越好)</option>
            <option value="TPSA">TPSA (Å²)</option>
          </select>
        </div>
        <canvas id="b5ParetoCanvas" height="310"></canvas>
        <div class="note" id="b5Hover" style="margin-top:6px">滑鼠移至圓點查看詳情；點擊有 3D 姿勢的候選直接開啟 3Dmol + PLIP 解析。</div>
      </div>

      <div class="card">
        <h2>Stage 5B 各化學架構均值對照（對比臨床與前代基準）</h2>
        <div class="scroll-x"><table id="b5SeriesTable"></table></div>
        <div class="callout" style="margin-top:10px">
          <b>ADMET 取捨總結（雙軸互補策略）：</b><br>
          • <b>追求極致親和力 + 最低 hERG + 高 QED</b> → 主推 <b><code>NAc1</code> / <code>NAc2</code> 中性醯胺系列</b>（<code>NAc1_4FmCOOH_3SO2Me</code>、<code>NAc1_mCOOH_3CF3</code>、<code>NAc2_pCOOH_3OCF3</code>、<code>NAc1_mCOOH_6CF3Py3</code>、<code>NAc1_mCOOH_3Cl4F</code>）。<br>
          • <b>追求極致低 DILI（&lt;0.55）+ 最易合成（SA ~2.0）+ 低 hERG</b> → 主推 <b><code>NBn</code> 緊緻 N-Benzyl 兩性離子系列</b>（<code>NBn_pCOOH_3SO2Me</code>、<code>NBn_mCOOH_6ClPy3</code>、<code>NBn_mCOOH_3Cl4F</code>）。
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Stage 5B 完整 132 分子實測排名總表（含 PLIP 3.0.1 關鍵互動；點欄位排序、點列看 3D）</h2>
      <div class="toolbar">
        <input id="b5Search" type="text" placeholder="搜尋 ID / 苯胺 / PLIP 殘基 (如 Trp181, Asn174)…" style="width:280px">
        <select id="b5SeriesFilter">
          <option value="ALL">全部系列 (132)</option>
          <option value="S5B_neutral_acyl">中性 N-Acyl hERG-Killer (43)</option>
          <option value="S5B_N_benzyl">緊緻 N-Benzyl 低DILI (17)</option>
          <option value="S5B_C2_amide">C2-醯胺兩性離子 (30)</option>
          <option value="S5B_C3_amide">C3-醯胺兩性離子 (17)</option>
          <option value="S5B_nitro_probe">2F→1NO₂ 硝基探針 (19)</option>
          <option value="ref">臨床參考配體 (3)</option>
          <option value="ref_s4">Stage 4 基準 (1)</option>
          <option value="ref_s5a">Stage 5A 冠軍基準 (2)</option>
        </select>
        <label class="note"><input type="checkbox" id="b5CleanOnly"> 僅顯示 0 Brenk 警示</label>
        <label class="note"><input type="checkbox" id="b5LowTox"> 低毒區 (hERG&lt;0.35 &amp; AMES&lt;0.20)</label>
        <label class="note"><input type="checkbox" id="b5LowDili"> 嚴格低 DILI (&lt;0.65)</label>
        <button id="btnExport5b">下載 Stage 5B CSV</button>
      </div>
      <div class="scroll-x scroll-y"><table id="b5Table"></table></div>
    </div>
  </section>

  <!-- ================= TAB 5A ================= -->
  <section class="tab-pane" id="t5a">
    <div class="card">
      <h2>Stage 5A 先導實驗：直接替換 c310_15 苯環（40 分子：2F→1NO₂ vs 市售安全苯胺）</h2>
      <div id="nitroVerdictBox" class="grid3"></div>
    </div>
    <div class="grid2">
      <div class="card">
        <h2>Stage 5A Pareto 散佈圖（Vina / Vinardo vs AMES / DILI / hERG / QED）</h2>
        <div class="toolbar">
          <label class="note">X 軸：</label>
          <select id="scX">
            <option value="Vina" selected>smina Vina pH7.4</option>
            <option value="Vinardo">smina Vinardo pH7.4</option>
            <option value="CNNaff">GNINA CNNaff</option>
          </select>
          <label class="note">Y 軸：</label>
          <select id="scY">
            <option value="AMES" selected>AMES 風險</option>
            <option value="DILI">DILI 風險</option>
            <option value="hERG">hERG 風險</option>
            <option value="QED">QED 類藥性</option>
          </select>
        </div>
        <canvas id="paretoCanvas" height="290"></canvas>
        <div class="note" id="paretoHover" style="margin-top:6px">滑鼠移至圓點查看詳情，點擊直接開啟 3D 姿勢。</div>
      </div>
      <div class="card">
        <h2>Stage 5A 系列均值對照</h2>
        <div class="scroll-x"><table id="seriesSummaryTable"></table></div>
      </div>
    </div>
    <div class="card">
      <h2>Stage 5A 40 分子實測總表</h2>
      <div class="toolbar">
        <input id="sarSearch" type="text" placeholder="搜尋 ID / 取代基…" style="width:230px">
        <select id="seriesFilter">
          <option value="ALL">全部系列 (40)</option>
          <option value="ref">臨床參考 (ref)</option>
          <option value="leader">Stage 4 領先 (leader)</option>
          <option value="A_nitro">A組：2F→1NO₂ (5)</option>
          <option value="A_ewg_safe">A組：安全拉電子 (7)</option>
          <option value="A_halogen">A組：廉價鹵素 (3)</option>
          <option value="A_heteroaryl">A組：缺電子吡啶 (3)</option>
          <option value="B_dual_comm">B組：雙環全市售 (12)</option>
          <option value="C_rigid">C組：縮短連接鏈 (4)</option>
        </select>
        <button id="btnExportCsv">下載 Stage 5A CSV</button>
      </div>
      <div class="scroll-x scroll-y"><table id="sarTable"></table></div>
    </div>
  </section>

  <!-- ================= TAB 1 (Stage 1-4) ================= -->
  <section class="tab-pane" id="t1">
    <div class="grid2">
      <div class="card">
        <h2>多代演化軌跡（80 個 Cycle：Stage 1 → 4）</h2>
        <div class="toolbar">
          <span class="note">指標：</span>
          <button class="on" data-ev="aff">DynamicBind 分數</button>
          <button data-ev="hERG">hERG 風險</button>
          <button data-ev="QED">QED 類藥性</button>
          <button data-ev="nF">氟原子數 (F)</button>
        </div>
        <canvas id="evCanvas" height="270"></canvas>
        <div class="note" style="margin-top:8px">
          <b>歷史軌跡總結：</b>Stage 1（半乳糖-三唑）→ Stage 2（非醣類探索）→ Stage 3（發現 4-氨基哌啶苯脲，但 hERG 0.97）→ Stage 4（導入 `-COOH` 壓低 hERG，卻因單次 DynamicBind 獎勵引發 4F–6F 多氟膨脹與 Michael acceptor）→ <b>Stage 5A/5B 徹底根除多氟與警示基團，並透過 N-Acyl / N-Benzyl 實現 QED 翻 3 倍與 hERG 壓低至 0.05–0.15！</b>
        </div>
      </div>
      <div class="card">
        <h2>Stage 1–4 參考配體與 5 種子領先分子校準</h2>
        <div class="callout warn">
          DynamicBind 對分子量與多氟基團有顯著正偏差（給 MW 649 的 TD-139 6.43，卻只給 25 nM 的 GB1211 5.17）；而 <code>c317_00</code> 單次 9.17 在 5 種子重跑崩跌至 6.49±0.52。因此 Stage 5 起改用 <b>smina 雙打分 + GNINA GPU CNN + PLIP + ADMET-AI</b> 多引擎共識。
        </div>
        <div class="scroll-x"><table id="refCalTable"></table></div>
      </div>
    </div>
    <div class="card">
      <h2>跨世代 ADMET 演進總對比（臨床醣類 vs Stage 3/4 vs Stage 5A vs Stage 5B 突破領先）</h2>
      <div class="scroll-x"><table id="admetMatrixTable"></table></div>
    </div>
  </section>

  <!-- ================= TAB 3 (3D Viewer) ================= -->
  <section class="tab-pane" id="t3">
    <div class="v-layout">
      <div class="v-side">
        <div style="padding:8px;border-bottom:1px solid var(--line)">
          <input id="vSearch" type="text" placeholder="篩選分子 ID / 系列…" style="width:100%">
        </div>
        <div class="v-list" id="vList"></div>
      </div>
      <div class="v-center">
        <div class="v-tools">
          <button id="bReset">重設視角</button>
          <button id="bSurf">蛋白表面</button>
          <button id="bLabel" class="on">CRD 殘基標籤</button>
          <button id="bHb" class="on">3.4 Å 氫鍵</button>
          <button id="bCartoon" class="on">蛋白 Cartoon</button>
          <button id="bSpin">自轉</button>
          <span style="flex:1"></span>
          <span class="note" id="vPoseBadge"></span>
        </div>
        <div id="viewwrap">
          <div id="viewer3d"></div>
          <div id="vcap">受體：Galectin-3 CRD（PDB 5E89 A 鏈）。黃色側鏈＝4.5 Å 內接觸到的 7 個關鍵 CRD 殘基（Arg144、His158、Arg162、Asn174、Trp181、Glu184、Arg186）。</div>
        </div>
      </div>
      <div class="v-right"><div class="v-right-in" id="vRight"></div></div>
    </div>
  </section>

  <!-- ================= TAB 4 (Synthesis / FTO / Local REINVENT4) ================= -->
  <section class="tab-pane" id="t4">
    <div class="grid2">
      <div class="card">
        <h2>Stage 5B 兩大新骨架：極簡 2 步大宗試劑合成路線（0 手性中心！）</h2>
        <div class="callout gold">
          相較臨床醣類抑制劑 TD-139 / GB1107 / GB1211 需 6–10 步立體化學合成（SA 4.2–5.1），Stage 5B 的 <b><code>NAc1</code> 中性醯胺</b>與 <b><code>NBn</code> 緊緻兩性離子</b> 全部 <b>SA = 2.0–2.3</b>、零手性中心、2–3 步一鍋化組裝：
        </div>
        <pre class="codeblock">
【路線 1：NAc1 中性 N-Acyl-哌啶（#1 NAc1_4FmCOOH_3SO2Me / #3 NAc1_mCOOH_3CF3）】
  Step 1（脲鍵）：Boc-4-aminopiperidine + 市售異氰酸酯 Ar_R-N=C=O
                 （如 3-SO₂Me / 3-CF₃ / 3-OCF₃ / 3-Cl-4-F / 6-CF₃-Py3 / 3-CN）
                 → TFA/DCM 脫 Boc（幾乎定量）
  Step 2（醯胺偶聯 + 水解）：+ 市售單甲酯二酸（如 2-(3-(methoxycarbonyl)phenyl)acetic acid
                 或 2-(3-(methoxycarbonyl)-4-fluorophenyl)acetic acid）+ HATU/DIPEA
                 → LiOH 水解 → 目標分子（0 手性、2 步完成、純化極易！）

【路線 2：NBn 緊緻低 DILI 兩性離子（#15 NBn_pCOOH_3SO2Me / #13 NBn_mCOOH_6ClPy3）】
  Step 1：Boc-4-aminopiperidine + 市售 3- 或 4-formylbenzoic acid methyl ester
          + NaBH(OAc)₃ 還原胺化 → TFA 脫 Boc
  Step 2：+ 市售異氰酸酯 Ar_R-N=C=O → LiOH 水解 → 目標分子（SA 僅 2.01–2.10！）
        </pre>
      </div>

      <div class="card">
        <h2>專利迴避（FTO）與下一輪濕實驗／REINVENT 4 建議</h2>
        <table>
          <thead><tr><th>比較維度</th><th>Galecto 臨床專利群 (TD-139/GB1107/1211)</th><th>Stage 5B 主推雙系列 (NAc1 / NBn)</th></tr></thead>
          <tbody>
            <tr><td>核心架構</td><td>硫代半乳糖苷 + 1,2,3-三唑（多手性中心）</td><td><b>非醣類 N-醯基/N-苄基 4-氨基哌啶單脲（0 手性）</b></td></tr>
            <tr><td>苯環取代</td><td>3-F、3,4,5-triF 苯基（專利高度重疊）</td><td><b>3-SO₂Me、3-CF₃、3-OCF₃、3-Cl-4-F、6-CF₃-Py3、3-CN、3-NO₂（0–1 個環氟）</b></td></tr>
            <tr><td>結合機制 (PLIP)</td><td>半乳糖 C-H/π 堆疊 Trp181 + 羥基接 His158/Asn174</td><td><b>芳環平行 π 堆疊 Trp181 (3.6–3.9 Å) + 羧酸/脲鍵接 Asn174/His158/Arg162/Arg144</b></td></tr>
            <tr><td>ADMET 突破</td><td>hERG 0.60–0.71 · DILI 0.93–0.94 · QED 0.15–0.44</td><td><b>NAc1：hERG 0.05–0.29 / AMES 0.02–0.13 / QED 0.58–0.70</b><br><b>NBn：DILI 0.48–0.54 / hERG 0.11–0.34 / QED 0.65–0.70</b></td></tr>
          </tbody>
        </table>
        <div class="callout" style="margin-top:12px">
          <b>建議優先合成驗證（6 支首選 Panel）：</b><br>
          1. <code>NAc1_4FmCOOH_3SO2Me</code>（綜合 #1，MPO 0.811，hERG 0.08，1 F）<br>
          2. <code>NAc1_mCOOH_3CF3</code>（7/7 滿額 CRD，QED 0.65，TPSA 98.7）<br>
          3. <code>NAc2_pCOOH_3OCF3</code>（最高對接 Vina −8.60 / Vinardo −8.74 / CNNaff 5.78）<br>
          4. <code>NAc1_mCOOH_6CF3Py3</code>（雜芳環，Vinardo −8.74，hERG 0.15）<br>
          5. <code>NBn_pCOOH_3SO2Me</code>（0 F，低 DILI 0.49，hERG 0.15，SA 2.04）<br>
          6. <code>NAc1_mCOOH_3NO2</code>（0 F 硝基對照探針，Vinardo −8.19，AMES 0.159）
        </div>
      </div>
    </div>

    <div class="card">
      <h2>本機 WSL2 + RTX 4060 環境清單與 REINVENT 4 續跑指令</h2>
      <pre class="codeblock"># 已於本機 WSL2 (Ubuntu-22.04 + NVIDIA GeForce RTX 4060) 完整安裝並驗證通過：
#   1) conda env: reinvent4   -> REINVENT 4.9.34 + PyTorch 2.13.0+cu126 (CUDA True)
#   2) conda env: gal3_eval   -> ADMET-AI 1.4.0 + OpenBabel 3.2.1 + PLIP 3.0.1 + Meeko 0.8 + RDKit 2026.03.6
#   3) ~/aidd_tools/bin/      -> smina (Vina/Vinardo) &amp; gnina-gpu v1.3.3 (GPU CNN)
#   4) conda env: dynamicbind + relax -> PyTorch 2.0.1+cu117 + OpenMM 8.5.2

# 若要啟動 REINVENT 4 Mol2Mol 強化學習（stage5_seeds.smi 已更新為 Stage 5B Top 10）：
#   請從瀏覽器下載官方 prior (https://doi.org/10.5281/zenodo.15641296) 至 ~/aidd_tools/REINVENT4/priors/
conda activate reinvent4
reinvent /mnt/d/2026@AI_reinvent4/stage5_reinvent4/stage5_staged_learning.toml

# 重新執行或擴充 Stage 5B 大規模市售組合庫：
conda activate gal3_eval
python /mnt/d/2026@AI_reinvent4/scripts/run_stage5b_library.py
python /mnt/d/2026@AI_reinvent4/scripts/build_standalone_report.py</pre>
    </div>
  </section>
</div>

<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
__THREEMOL_SCRIPT__
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const CYCLES = DATA.cycles;
const LR = DATA.leaders_refs;
const S5A = DATA.stage5a;
const S5B = DATA.stage5b;
const CRD = DATA.crd;
const KEYRES = Object.keys(CRD).map(Number);

const SERIES_META = {
  ref:              {name:'臨床參考', cls:'p-ref',   col:'#be95ff'},
  ref_s4:           {name:'Stage4基準', cls:'p-lead', col:'#f0c766'},
  ref_s5a:          {name:'Stage5A基準', cls:'p-pol', col:'#94a3b8'},
  leader:           {name:'Stage4領先', cls:'p-lead', col:'#f0c766'},
  S5B_neutral_acyl: {name:'中性 N-Acyl', cls:'p-nac', col:'#3ad1be'},
  S5B_N_benzyl:     {name:'緊緻 N-Benzyl', cls:'p-nbn', col:'#6fd08c'},
  S5B_C2_amide:     {name:'C2-醯胺', cls:'p-c2a', col:'#6ea8fe'},
  S5B_C3_amide:     {name:'C3-醯胺', cls:'p-c3a', col:'#f2a65a'},
  S5B_nitro_probe:  {name:'2F→NO₂ 探針', cls:'p-nitro', col:'#f06c7a'},
  A_nitro:          {name:'A: 2F→NO₂', cls:'p-nitro', col:'#f06c7a'},
  A_ewg_safe:       {name:'A: 安全EWG', cls:'p-safe', col:'#3ad1be'},
  A_halogen:        {name:'A: 廉價鹵素', cls:'p-hal', col:'#6ea8fe'},
  A_heteroaryl:     {name:'A: 缺電子吡啶', cls:'p-het', col:'#a78bfa'},
  A_polar:          {name:'A: 極性醯胺', cls:'p-pol', col:'#94a3b8'},
  B_dual_comm:      {name:'B: 雙環市售', cls:'p-dual', col:'#6fd08c'},
  C_rigid:          {name:'C: 剛性連接鏈', cls:'p-rig', col:'#f2a65a'}
};
const fmt = (v,d=2) => (v===null||v===undefined||Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
const mean = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : null;
const riskCls = (v, hiBad) => {
  if(v==null) return '';
  if(hiBad) return v>=0.6?'bad':(v<=0.35?'good':'mid');
  return v>=0.7?'good':(v<=0.35?'bad':'mid');
};

/* ---------------- Tabs ---------------- */
let viewerInit = false;
document.querySelectorAll('#tabNav button').forEach(btn=>{
  btn.onclick = ()=>{
    document.querySelectorAll('#tabNav button').forEach(b=>b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
    if(btn.dataset.tab==='t5b') drawB5Pareto();
    if(btn.dataset.tab==='t5a') drawPareto();
    if(btn.dataset.tab==='t1') drawEvolution();
    if(btn.dataset.tab==='t3'){
      if(!viewerInit){ initViewer(); viewerInit=true; }
      else if(viewer3d){ viewer3d.resize(); viewer3d.render(); }
    }
  };
});

/* ---------------- KPIs ---------------- */
(function buildKPIs(){
  const top1 = S5B[0];
  const lowDili = S5B.find(m=>m.id==='NBn_pCOOH_3SO2Me') || S5B.find(m=>m.series==='S5B_N_benzyl');
  const nacArr = S5B.filter(m=>m.series==='S5B_neutral_acyl');
  const nbnArr = S5B.filter(m=>m.series==='S5B_N_benzyl');
  document.getElementById('kpiBar').innerHTML = `
    <div class="kpi"><div class="lbl">全專案累計分析分子</div><div class="val">${CYCLES.length+LR.length+S5A.length+S5B.length}</div><div class="desc">Stage 1–4 (93) + Stage 5A (40) + Stage 5B (132)</div></div>
    <div class="kpi"><div class="lbl">Stage 5B 綜合冠軍 (#1 MPO)</div><div class="val" style="color:var(--teal);font-size:15px">${top1.id}</div><div class="desc">MPO ${fmt(top1.MPO,3)} · Vina ${fmt(top1.Vina)} / Vn ${fmt(top1.Vinardo)} · CNN ${fmt(top1.CNNaff)} · hERG ${fmt(top1.hERG)}</div></div>
    <div class="kpi"><div class="lbl">hERG-Killer (NAc 均值)</div><div class="val" style="color:var(--green)">hERG ${fmt(mean(nacArr.map(x=>x.hERG)),2)} · QED ${fmt(mean(nacArr.map(x=>x.QED)),2)}</div><div class="desc">AMES 均值 ${fmt(mean(nacArr.map(x=>x.AMES)),2)} · SA ${fmt(mean(nacArr.map(x=>x.SA)),2)} (vs 臨床 hERG 0.67)</div></div>
    <div class="kpi"><div class="lbl">低 DILI 冠軍 (NBn 均值)</div><div class="val" style="color:#9df5b6">DILI ${fmt(mean(nbnArr.map(x=>x.DILI)),2)} · SA ${fmt(mean(nbnArr.map(x=>x.SA)),2)}</div><div class="desc">${lowDili?.id}: DILI ${fmt(lowDili?.DILI)} · hERG ${fmt(lowDili?.hERG)} · 0 F</div></div>
    <div class="kpi"><div class="lbl">PLIP 3.0.1 關鍵驗證</div><div class="val" style="font-size:15px">Trp181 π + Asn174 HB</div><div class="desc">Top 15 幾乎全數精準形成 Trp181 平行 π 堆積 (3.6–3.9 Å)</div></div>
  `;
})();

/* ---------------- Tab 5B: Series Summary, Pareto, Table ---------------- */
S5B.forEach(m=>{ m.contacts_n = m.contacts.length; });
(function buildB5Summary(){
  const byS = {};
  S5B.forEach(m=>{ (byS[m.series]=byS[m.series]||[]).push(m); });
  const order = ['S5B_neutral_acyl','S5B_N_benzyl','S5B_C2_amide','S5B_C3_amide','S5B_nitro_probe','ref_s5a','ref_s4','ref'];
  document.getElementById('b5SeriesTable').innerHTML = `
    <thead><tr><th>化學系列</th><th class="n">n</th><th class="n">MPO ↑</th><th class="n">Vina ↓</th><th class="n">Vn ↓</th><th class="n">CNN ↑</th><th class="n">hERG ↓</th><th class="n">DILI ↓</th><th class="n">AMES ↓</th><th class="n">QED ↑</th><th class="n">SA ↓</th><th class="n">TPSA</th></tr></thead>
    <tbody>${order.filter(k=>byS[k]).map(k=>{
      const a = byS[k], meta = SERIES_META[k]||{name:k,cls:'p-pol'};
      return `<tr>
        <td><span class="pill ${meta.cls}">${meta.name}</span></td>
        <td class="n">${a.length}</td>
        <td class="n"><b>${fmt(mean(a.map(x=>x.MPO)),3)}</b></td>
        <td class="n">${fmt(mean(a.map(x=>x.Vina)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.Vinardo)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.CNNaff)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.hERG)),true)}">${fmt(mean(a.map(x=>x.hERG)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.DILI)),true)}">${fmt(mean(a.map(x=>x.DILI)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.AMES)),true)}">${fmt(mean(a.map(x=>x.AMES)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.QED)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.SA)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.TPSA)),0)}</td>
      </tr>`;
    }).join('')}</tbody>`;
})();

let b5Pts = [];
function drawB5Pareto(){
  const c = document.getElementById('b5ParetoCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 560, H = 310;
  c.width = W*dpr; c.height = H*dpr; c.style.height = H+'px';
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const kx = document.getElementById('b5ScX').value, ky = document.getElementById('b5ScY').value;
  const L=48, R=18, T=16, B=34, pw=W-L-R, ph=H-T-B;
  const xs = S5B.map(m=>m[kx]).filter(v=>v!=null), ys = S5B.map(m=>m[ky]).filter(v=>v!=null);
  const xpad = (Math.max(...xs)-Math.min(...xs))*0.06 || 0.2;
  const ypad = (Math.max(...ys)-Math.min(...ys))*0.06 || 0.05;
  const xlo = Math.min(...xs)-xpad, xhi = Math.max(...xs)+xpad;
  const ylo = Math.min(...ys)-ypad, yhi = Math.max(...ys)+ypad;
  const X = v => L + pw*(v-xlo)/(xhi-xlo||1);
  const Y = v => T + ph*(1-(v-ylo)/(yhi-ylo||1));

  g.strokeStyle='#24334f'; g.lineWidth=1; g.fillStyle='#8c9cb8'; g.font='10.5px monospace';
  for(let k=0;k<=4;k++){
    const xv = xlo+(xhi-xlo)*k/4, yv = ylo+(yhi-ylo)*k/4;
    g.beginPath(); g.moveTo(X(xv),T); g.lineTo(X(xv),T+ph); g.stroke();
    g.beginPath(); g.moveTo(L,Y(yv)); g.lineTo(W-R,Y(yv)); g.stroke();
    g.textAlign='center'; g.fillText(xv.toFixed(2), X(xv), H-12);
    g.textAlign='right'; g.fillText(yv.toFixed(2), L-6, Y(yv)+3);
  }
  b5Pts = [];
  S5B.forEach((m,idx)=>{
    if(m[kx]==null||m[ky]==null) return;
    const px = X(m[kx]), py = Y(m[ky]), col = (SERIES_META[m.series]||{}).col||'#ccc';
    const isKey = m.series.startsWith('ref') || idx < 4;
    const r = isKey ? 5.8 : 4.2;
    g.fillStyle = col; g.globalAlpha = isKey ? 1 : 0.82;
    g.beginPath(); g.arc(px,py,r,0,Math.PI*2); g.fill();
    g.globalAlpha = 1;
    if(isKey){
      g.strokeStyle='#fff'; g.lineWidth=1.4; g.stroke();
      g.fillStyle='#eaf0fa'; g.font='10px sans-serif'; g.textAlign='left';
      g.fillText(m.id, Math.min(px+7, W-110), py+3);
    }
    b5Pts.push({m, px, py});
  });
}
document.getElementById('b5ScX').onchange = drawB5Pareto;
document.getElementById('b5ScY').onchange = drawB5Pareto;
document.getElementById('b5ParetoCanvas').onmousemove = ev=>{
  const c = ev.currentTarget, rect = c.getBoundingClientRect();
  const mx = ev.clientX-rect.left, my = ev.clientY-rect.top;
  let hit = null, bestD = 14;
  b5Pts.forEach(p=>{ const d=Math.hypot(mx-p.px,my-p.py); if(d<bestD){bestD=d;hit=p.m;} });
  document.getElementById('b5Hover').innerHTML = hit
    ? `<b>${hit.id}</b> · MPO <b>${fmt(hit.MPO,3)}</b> · Vina ${fmt(hit.Vina)} · Vn ${fmt(hit.Vinardo)} · CNN ${fmt(hit.CNNaff)} · hERG ${fmt(hit.hERG)} · DILI ${fmt(hit.DILI)} · AMES ${fmt(hit.AMES)} · QED ${fmt(hit.QED)} · SA ${fmt(hit.SA)}${hit.plip_sum?' · PLIP: '+hit.plip_sum:''}`
    : '滑鼠移至圓點查看詳情；點擊有 3D 姿勢的候選直接開啟 3Dmol + PLIP 解析。';
};
document.getElementById('b5ParetoCanvas').onclick = ev=>{
  const c = ev.currentTarget, rect = c.getBoundingClientRect();
  const mx = ev.clientX-rect.left, my = ev.clientY-rect.top;
  let hit = null, bestD = 14;
  b5Pts.forEach(p=>{ const d=Math.hypot(mx-p.px,my-p.py); if(d<bestD){bestD=d;hit=p.m;} });
  if(hit) jumpTo3D('5B:' + hit.id);
};

let b5SortKey = 'MPO', b5SortAsc = false;
function renderB5Table(){
  const q = (document.getElementById('b5Search').value||'').trim().toLowerCase();
  const sf = document.getElementById('b5SeriesFilter').value;
  const cleanOnly = document.getElementById('b5CleanOnly').checked;
  const lowTox = document.getElementById('b5LowTox').checked;
  const lowDili = document.getElementById('b5LowDili').checked;
  let list = S5B.filter(m=>{
    if(sf!=='ALL' && m.series!==sf) return false;
    if(cleanOnly && m.BRENK) return false;
    if(lowTox && (m.hERG>=0.35 || m.AMES>=0.20)) return false;
    if(lowDili && m.DILI>=0.65) return false;
    if(q && !(m.id.toLowerCase().includes(q) || m.desc.toLowerCase().includes(q) || (m.plip_sum||'').toLowerCase().includes(q) || m.smiles.toLowerCase().includes(q))) return false;
    return true;
  });
  list.sort((a,b)=>{
    const va = a[b5SortKey], vb = b[b5SortKey];
    if(va==null) return 1; if(vb==null) return -1;
    if(typeof va==='string') return b5SortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return b5SortAsc ? va-vb : vb-va;
  });
  const cols = [
    ['id','ID (3D)'],['series','系列'],['MPO','MPO ↑'],['Vina','Vina ↓'],['Vinardo','Vn ↓'],
    ['CNNaff','CNN ↑'],['hERG','hERG ↓'],['DILI','DILI ↓'],['AMES','AMES ↓'],
    ['Bioavail','Bio ↑'],['QED','QED ↑'],['SA','SA ↓'],['TPSA','TPSA'],['MW','MW'],['nF','F'],['contacts_n','CRD'],['plip_sum','PLIP 3.0.1 關鍵互動']
  ];
  const thead = '<thead><tr>'+cols.map(([k,lbl])=>{
    const arrow = b5SortKey===k ? (b5SortAsc?' ▲':' ▼') : '';
    const numCls = ['id','series','plip_sum'].includes(k)?'':'n';
    return `<th class="sort ${numCls}" data-k="${k}">${lbl}${arrow}</th>`;
  }).join('')+'</tr></thead>';
  const tbody = '<tbody>'+list.map(m=>{
    const meta = SERIES_META[m.series]||{name:m.series,cls:'p-pol'};
    const badge3d = m.mol ? '<span style="color:var(--teal);font-size:10px"> 🧊</span>' : '';
    return `<tr class="${m.mol?'clickable':''}" data-id="${m.mol?'5B:'+m.id:''}">
      <td><b>${m.id}</b>${badge3d}</td>
      <td><span class="pill ${meta.cls}">${meta.name}</span></td>
      <td class="n"><b>${fmt(m.MPO,3)}</b></td>
      <td class="n">${fmt(m.Vina,2)}</td>
      <td class="n">${fmt(m.Vinardo,2)}</td>
      <td class="n">${fmt(m.CNNaff,2)}</td>
      <td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG,2)}</td>
      <td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI,2)}</td>
      <td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES,2)}</td>
      <td class="n ${riskCls(m.Bioavail,false)}">${fmt(m.Bioavail,2)}</td>
      <td class="n">${fmt(m.QED,2)}</td>
      <td class="n">${fmt(m.SA,2)}</td>
      <td class="n">${fmt(m.TPSA,0)}</td>
      <td class="n">${fmt(m.MW,0)}</td>
      <td class="n">${m.nF}</td>
      <td class="n">${m.contacts_n}/7</td>
      <td class="note" style="min-width:210px;font-size:11px">${m.plip_sum || (m.BRENK?'⚠ '+m.BRENK:'—')}</td>
    </tr>`;
  }).join('')+'</tbody>';
  const tbl = document.getElementById('b5Table');
  tbl.innerHTML = thead + tbody;
  tbl.querySelectorAll('th.sort').forEach(th=>{
    th.onclick = ()=>{
      const k = th.dataset.k;
      if(b5SortKey===k) b5SortAsc=!b5SortAsc;
      else { b5SortKey=k; b5SortAsc = !['MPO','CNNaff','Bioavail','QED','contacts_n'].includes(k); }
      renderB5Table();
    };
  });
  tbl.querySelectorAll('tr.clickable').forEach(tr=>{
    tr.onclick = ()=>{ if(tr.dataset.id) jumpTo3D(tr.dataset.id); };
  });
}
['b5Search','b5SeriesFilter','b5CleanOnly','b5LowTox','b5LowDili'].forEach(id=>{
  document.getElementById(id).oninput = renderB5Table;
  document.getElementById(id).onchange = renderB5Table;
});
document.getElementById('btnExport5b').onclick = ()=>{
  const keys = ['id','series','desc','smiles','MPO','Vina','Vinardo','CNNaff','CNNscore','hERG','DILI','AMES','Pgp','HIA','Bioavail','Sol','QED','SA','TPSA','cLogP','MW','RotB','nF','BRENK','plip_sum'];
  const csv = [keys.join(','), ...S5B.map(m=>keys.map(k=>`"${String(m[k]??'').replace(/"/g,'""')}"`).join(','))].join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  a.download = 'gal3_stage5b_132_molecules.csv';
  a.click();
};

/* ---------------- Tab 5A ---------------- */
S5A.forEach(m=>{ m.contacts_n = m.contacts.length; });
(function build5ASummary(){
  const byS = {}; S5A.forEach(m=>{ (byS[m.series]=byS[m.series]||[]).push(m); });
  const order = ['ref','leader','A_nitro','A_ewg_safe','A_halogen','A_heteroaryl','A_polar','B_dual_comm','C_rigid'];
  document.getElementById('seriesSummaryTable').innerHTML = `
    <thead><tr><th>系列</th><th class="n">n</th><th class="n">Vina ↓</th><th class="n">Vn ↓</th><th class="n">CNN ↑</th><th class="n">hERG</th><th class="n">DILI</th><th class="n">AMES</th><th class="n">QED</th><th class="n">SA ↓</th></tr></thead>
    <tbody>${order.filter(k=>byS[k]).map(k=>{
      const a = byS[k], meta = SERIES_META[k];
      return `<tr>
        <td><span class="pill ${meta.cls}">${meta.name}</span></td><td class="n">${a.length}</td>
        <td class="n">${fmt(mean(a.map(x=>x.Vina)),2)}</td><td class="n">${fmt(mean(a.map(x=>x.Vinardo)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.CNNaff)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.hERG)),true)}">${fmt(mean(a.map(x=>x.hERG)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.DILI)),true)}">${fmt(mean(a.map(x=>x.DILI)),2)}</td>
        <td class="n ${riskCls(mean(a.map(x=>x.AMES)),true)}">${fmt(mean(a.map(x=>x.AMES)),2)}</td>
        <td class="n">${fmt(mean(a.map(x=>x.QED)),2)}</td><td class="n">${fmt(mean(a.map(x=>x.SA)),2)}</td>
      </tr>`;
    }).join('')}</tbody>`;

  const nitro = byS['A_nitro']||[], c310 = S5A.find(x=>x.id==='c310_15'), b07 = S5A.find(x=>x.id==='B07_L5NH2FBA_R3SO2Me'), c03 = S5A.find(x=>x.id==='C03_C2_L3NH5NO2_R3CN');
  document.getElementById('nitroVerdictBox').innerHTML = `
    <div class="callout warn" style="margin:0"><h3>① 2F → 1NO₂ 結合力達標</h3><div>零氟 <b>${c03?.id}</b> 達 Vina ${fmt(c03?.Vina)} / Vinardo ${fmt(c03?.Vinardo)}（勝過 c310_15 ${fmt(c310?.Vina)} / ${fmt(c310?.Vinardo)}），驗證電子等效假說。</div></div>
    <div class="callout bad" style="margin:0"><h3>② 雙脲硝基 AMES 放大 3.4 倍</h3><div>硝基組平均 AMES = <b>${fmt(mean(nitro.map(x=>x.AMES)),3)}</b>（對比 c310_15 ${fmt(c310?.AMES,3)}）、TPSA 升至 166–190 Å²。</div></div>
    <div class="callout" style="margin:0"><h3>③ Stage 5A 冠軍催生 Stage 5B</h3><div><b>${b07?.id}</b>（3-SO₂Me，僅 1 F）跑出 Vina ${fmt(b07?.Vina)} / hERG ${fmt(b07?.hERG)} / AMES ${fmt(b07?.AMES)}，直接引導出 Stage 5B 的 <b>NAc1_4FmCOOH_3SO2Me (MPO 0.811)</b>！</div></div>`;
})();

let p5aPts = [];
function drawPareto(){
  const c = document.getElementById('paretoCanvas'), dpr = window.devicePixelRatio||1;
  const W = c.clientWidth||540, H = 290;
  c.width = W*dpr; c.height = H*dpr; c.style.height = H+'px';
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const kx = document.getElementById('scX').value, ky = document.getElementById('scY').value;
  const L=48, R=18, T=16, B=34, pw=W-L-R, ph=H-T-B;
  const xs = S5A.map(m=>m[kx]).filter(v=>v!=null), ys = S5A.map(m=>m[ky]).filter(v=>v!=null);
  const xlo = Math.min(...xs)-0.2, xhi = Math.max(...xs)+0.2, ylo = Math.min(...ys)*0.9, yhi = Math.max(...ys)*1.06;
  const X = v => L + pw*(v-xlo)/(xhi-xlo||1), Y = v => T + ph*(1-(v-ylo)/(yhi-ylo||1));
  g.strokeStyle='#24334f'; g.lineWidth=1; g.fillStyle='#8c9cb8'; g.font='10.5px monospace';
  for(let k=0;k<=4;k++){
    const xv=xlo+(xhi-xlo)*k/4, yv=ylo+(yhi-ylo)*k/4;
    g.beginPath(); g.moveTo(X(xv),T); g.lineTo(X(xv),T+ph); g.stroke();
    g.beginPath(); g.moveTo(L,Y(yv)); g.lineTo(W-R,Y(yv)); g.stroke();
    g.textAlign='center'; g.fillText(xv.toFixed(2), X(xv), H-12);
    g.textAlign='right'; g.fillText(yv.toFixed(2), L-6, Y(yv)+3);
  }
  p5aPts = [];
  S5A.forEach(m=>{
    if(m[kx]==null||m[ky]==null) return;
    const px=X(m[kx]), py=Y(m[ky]), col=(SERIES_META[m.series]||{}).col||'#ccc';
    g.fillStyle=col; g.beginPath(); g.arc(px,py,5,0,Math.PI*2); g.fill();
    if(['ref','leader'].includes(m.series)){ g.strokeStyle='#fff'; g.lineWidth=1.4; g.stroke(); g.fillStyle='#eaf0fa'; g.font='10px sans-serif'; g.textAlign='left'; g.fillText(m.id, px+6, py+3); }
    p5aPts.push({m,px,py});
  });
}
document.getElementById('scX').onchange = drawPareto;
document.getElementById('scY').onchange = drawPareto;
document.getElementById('paretoCanvas').onmousemove = ev=>{
  const r = ev.currentTarget.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  let hit=null, bestD=14; p5aPts.forEach(p=>{const d=Math.hypot(mx-p.px,my-p.py); if(d<bestD){bestD=d;hit=p.m;}});
  document.getElementById('paretoHover').innerHTML = hit ? `<b>${hit.id}</b> · Vina ${fmt(hit.Vina)} · Vn ${fmt(hit.Vinardo)} · hERG ${fmt(hit.hERG)} · DILI ${fmt(hit.DILI)} · AMES ${fmt(hit.AMES)}` : '滑鼠移至圓點查看詳情，點擊直接開啟 3D 姿勢。';
};
document.getElementById('paretoCanvas').onclick = ev=>{
  const r = ev.currentTarget.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
  let hit=null, bestD=14; p5aPts.forEach(p=>{const d=Math.hypot(mx-p.px,my-p.py); if(d<bestD){bestD=d;hit=p.m;}});
  if(hit) jumpTo3D('5A:' + hit.id);
};

let s5aSort='Vina', s5aAsc=true;
function renderSarTable(){
  const q=(document.getElementById('sarSearch').value||'').trim().toLowerCase(), sf=document.getElementById('seriesFilter').value;
  let list = S5A.filter(m=>(sf==='ALL'||m.series===sf) && (!q || m.id.toLowerCase().includes(q) || m.desc.toLowerCase().includes(q)));
  list.sort((a,b)=>{ const va=a[s5aSort], vb=b[s5aSort]; if(va==null)return 1; if(vb==null)return -1; return typeof va==='string'?(s5aAsc?va.localeCompare(vb):vb.localeCompare(va)):(s5aAsc?va-vb:vb-va); });
  const cols=[['id','ID'],['series','系列'],['desc','設計'],['Vina','Vina ↓'],['Vinardo','Vn ↓'],['CNNaff','CNN ↑'],['hERG','hERG'],['DILI','DILI'],['AMES','AMES'],['QED','QED'],['SA','SA'],['MW','MW'],['nF','F'],['contacts_n','CRD']];
  const tbl=document.getElementById('sarTable');
  tbl.innerHTML='<thead><tr>'+cols.map(([k,l])=>`<th class="sort ${['id','series','desc'].includes(k)?'':'n'}" data-k="${k}">${l}${s5aSort===k?(s5aAsc?' ▲':' ▼'):''}</th>`).join('')+'</tr></thead><tbody>'+
    list.map(m=>{const meta=SERIES_META[m.series]||{name:m.series,cls:'p-pol'}; return `<tr class="clickable" data-id="5A:${m.id}"><td><b>${m.id}</b></td><td><span class="pill ${meta.cls}">${meta.name}</span></td><td class="note">${m.desc}</td><td class="n">${fmt(m.Vina)}</td><td class="n">${fmt(m.Vinardo)}</td><td class="n">${fmt(m.CNNaff)}</td><td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG)}</td><td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI)}</td><td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES)}</td><td class="n">${fmt(m.QED)}</td><td class="n">${fmt(m.SA)}</td><td class="n">${fmt(m.MW,0)}</td><td class="n">${m.nF}</td><td class="n">${m.contacts_n}/7</td></tr>`;}).join('')+'</tbody>';
  tbl.querySelectorAll('th.sort').forEach(th=>th.onclick=()=>{const k=th.dataset.k; if(s5aSort===k)s5aAsc=!s5aAsc; else{s5aSort=k;s5aAsc=!['CNNaff','QED','contacts_n'].includes(k);} renderSarTable();});
  tbl.querySelectorAll('tr.clickable').forEach(tr=>tr.onclick=()=>jumpTo3D(tr.dataset.id));
}
document.getElementById('sarSearch').oninput = renderSarTable;
document.getElementById('seriesFilter').onchange = renderSarTable;
document.getElementById('btnExportCsv').onclick = ()=>{
  const keys=['id','series','desc','smiles','Vina','Vinardo','CNNaff','hERG','DILI','AMES','QED','SA','TPSA','MW','nF'];
  const csv=[keys.join(','),...S5A.map(m=>keys.map(k=>`"${String(m[k]??'').replace(/"/g,'""')}"`).join(','))].join('\n');
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'})); a.download='gal3_stage5a_40_molecules.csv'; a.click();
};

/* ---------------- Tab 1 (Stage 1-4 Evolution & Cross-Generation Matrix) ---------------- */
let evMetric = 'aff';
document.querySelectorAll('[data-ev]').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('[data-ev]').forEach(x=>x.classList.remove('on')); b.classList.add('on'); evMetric=b.dataset.ev; drawEvolution();
});
function drawEvolution(){
  const c=document.getElementById('evCanvas'), dpr=window.devicePixelRatio||1, W=c.clientWidth||540, H=270;
  c.width=W*dpr; c.height=H*dpr; c.style.height=H+'px';
  const g=c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const L=46, R=16, T=22, B=32, pw=W-L-R, ph=H-T-B;
  const vals=CYCLES.map(m=>m[evMetric]).filter(v=>v!=null);
  const lo=Math.min(...vals, evMetric==='aff'?3.5:0), hi=Math.max(...vals, evMetric==='QED'||evMetric==='hERG'?1:0)*1.05;
  const X=i=>L+pw*i/(CYCLES.length-1), Y=v=>T+ph*(1-(v-lo)/(hi-lo||1));
  const sc={stage1:'#6ea8fe',stage2:'#6fd08c',stage3:'#f2a65a',stage4:'#f06c7a'};
  [0,20,40,60].forEach((s,si)=>{ const x1=X(s),x2=X(Math.min(s+19,CYCLES.length-1)); g.fillStyle=['rgba(110,168,254,.07)','rgba(111,208,140,.07)','rgba(242,166,90,.07)','rgba(240,108,122,.08)'][si]; g.fillRect(x1,T,x2-x1,ph); g.fillStyle='#95a5c2'; g.font='11px sans-serif'; g.textAlign='center'; g.fillText('Stage '+(si+1),(x1+x2)/2,T-6); });
  g.strokeStyle='#24334f'; g.lineWidth=1; g.fillStyle='#8c9cb8'; g.font='10.5px monospace'; g.textAlign='right';
  for(let k=0;k<=4;k++){ const v=lo+(hi-lo)*k/4, y=Y(v); g.beginPath(); g.moveTo(L,y); g.lineTo(W-R,y); g.stroke(); g.fillText(v.toFixed(evMetric==='nF'?0:2),L-6,y+3); }
  if(evMetric==='aff'){ const y=Y(6.434); g.strokeStyle='#be95ff'; g.setLineDash([5,4]); g.beginPath(); g.moveTo(L,y); g.lineTo(W-R,y); g.stroke(); g.setLineDash([]); g.fillStyle='#be95ff'; g.fillText('TD-139 (6.43)',W-R-4,y-5); }
  g.strokeStyle='#4a638f'; g.lineWidth=1.6; g.beginPath(); let st=false;
  CYCLES.forEach((m,i)=>{ const v=m[evMetric]; if(v==null)return; st?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)); st=true; }); g.stroke();
  CYCLES.forEach((m,i)=>{ const v=m[evMetric]; if(v==null)return; g.fillStyle=sc[m.group]||'#ccc'; g.beginPath(); g.arc(X(i),Y(v),3.2,0,Math.PI*2); g.fill(); });
}

(function buildTab1Tables(){
  const pkd={'TD-139':'14–68 nM','GB1211':'25 nM','GB1107':'37 nM','TAZTDG':'840 nM','thiodigalactoside':'75.4 µM','LacNAc':'~50 µM','lactose':'~200 µM'};
  document.getElementById('refCalTable').innerHTML = '<thead><tr><th>分子</th><th>類型</th><th class="n">實驗 Kd</th><th class="n">DynamicBind</th><th class="n">舊 Vina</th><th class="n">舊 Vn</th><th class="n">CRD</th><th class="n">F</th></tr></thead><tbody>'+
    LR.map(m=>`<tr><td><b>${m.id}</b></td><td><span class="pill ${m.group==='ref'?'p-ref':'p-lead'}">${m.group==='ref'?'參考':'領先'}</span></td><td class="n">${pkd[m.id]||'—'}</td><td class="n">${fmt(m.aff,3)}${m.aff_sd!=null?'±'+fmt(m.aff_sd,2):''}</td><td class="n">${fmt(m.vina_old,2)}</td><td class="n">${fmt(m.vinardo_old,2)}</td><td class="n">${m.contacts.length}/7</td><td class="n">${m.nF}</td></tr>`).join('')+'</tbody>';

  const pickIds = ['TD-139','GB1107','GB1211','c310_15','S5A_B07_3SO2Me','NAc1_4FmCOOH_3SO2Me','NAc1_mCOOH_3CF3','NAc2_pCOOH_3OCF3','NAc1_mCOOH_6CF3Py3','NBn_pCOOH_3SO2Me','NBn_mCOOH_6ClPy3','NAc1_mCOOH_3NO2'];
  const pick = pickIds.map(id=>S5B.find(x=>x.id===id)).filter(Boolean);
  document.getElementById('admetMatrixTable').innerHTML = '<thead><tr><th>分子</th><th>世代/系列</th><th class="n">Vina ↓</th><th class="n">Vn ↓</th><th class="n">CNN ↑</th><th class="n">hERG ↓</th><th class="n">DILI ↓</th><th class="n">AMES ↓</th><th class="n">Bio ↑</th><th class="n">QED ↑</th><th class="n">SA ↓</th><th class="n">TPSA</th><th class="n">F</th></tr></thead><tbody>'+
    pick.map(m=>{ const meta=SERIES_META[m.series]||{name:m.series,cls:'p-pol'}; return `<tr><td><b>${m.id}</b></td><td><span class="pill ${meta.cls}">${meta.name}</span></td><td class="n">${fmt(m.Vina)}</td><td class="n">${fmt(m.Vinardo)}</td><td class="n">${fmt(m.CNNaff)}</td><td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG)}</td><td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI)}</td><td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES)}</td><td class="n ${riskCls(m.Bioavail,false)}">${fmt(m.Bioavail)}</td><td class="n">${fmt(m.QED)}</td><td class="n">${fmt(m.SA)}</td><td class="n">${fmt(m.TPSA,0)}</td><td class="n">${m.nF}</td></tr>`; }).join('')+'</tbody>';
})();

/* ---------------- Tab 3: 3D Viewer (Stage 5B Top 45 + Stage 5A 40 + Stage 1-4 DynamicBind 13) ---------------- */
const ALL3D = [
  ...S5B.filter(m=>m.mol).map(m=>({...m, uid:'5B:'+m.id, grp:'★ Stage 5B 優化領先 (smina+PLIP)', src:'Stage 5B smina Vinardo 最佳姿勢 (pH 7.4)'})),
  ...S5A.map(m=>({...m, uid:'5A:'+m.id, grp:'Stage 5A 市售苯胺 SAR', src:'Stage 5A smina Vinardo 最佳姿勢 (pH 7.4)', plip:[]})),
  ...LR.map(m=>({
    uid:'DB:'+m.id, id:m.id, series:m.group==='ref'?'ref':'leader', grp:'Stage 1–4 DynamicBind 歷史姿勢',
    desc:(m.caption||'DynamicBind') + ` (DB=${fmt(m.aff,2)})`, smiles:m.smiles,
    MW:m.MW, cLogP:m.cLogP, TPSA:m.TPSA, RotB:m.RotB, nF:m.nF, QED:m.QED, SA:null, MPO:null,
    hERG:m.hERG, DILI:m.DILI, AMES:m.AMES, Pgp:m.Pgp, HIA:m.HIA, Bioavail:m.Bioavail, Sol:m.Sol,
    Vina:m.vina_old, Vinardo:m.vinardo_old, CNNaff:null, CNNscore:null, BRENK:'', contacts:m.contacts,
    plip:m.plip||[], png:m.png, mol:m.mol, src:'Stage 1–4 DynamicBind 疊合姿勢 (5E89 Cα Kabsch)'
  }))
];
let cur3D=0, viewer3d=null, protM=null, ligM=null, surf3d=null, labels3d=[], hb3d=[];
let showSurf=false, showLabel=true, showHb=true, showCartoon=true, spinOn=false;

function initViewer(){
  try{
    viewer3d = $3Dmol.createViewer(document.getElementById('viewer3d'), {backgroundColor:'#0b101b', antialias:true});
    protM = viewer3d.addModel(DATA.protein_pdb, 'pdb');
  } catch(e){
    document.getElementById('viewer3d').innerHTML = '<div style="padding:24px;color:#ffd0a0">WebGL 無法初始化：'+e+'</div>';
  }
  renderVList(); load3D(cur3D);
}
function renderVList(){
  const q=(document.getElementById('vSearch').value||'').trim().toLowerCase();
  const host=document.getElementById('vList'); host.innerHTML='';
  const grps = ['★ Stage 5B 優化領先 (smina+PLIP)','Stage 5A 市售苯胺 SAR','Stage 1–4 DynamicBind 歷史姿勢'];
  grps.forEach(gn=>{
    const items = ALL3D.map((m,i)=>({m,i})).filter(({m})=>m.grp===gn && (!q || m.id.toLowerCase().includes(q) || (m.desc||'').toLowerCase().includes(q)));
    if(!items.length) return;
    const h=document.createElement('div'); h.className='v-grp'; h.textContent=`${gn} (${items.length})`; host.appendChild(h);
    items.forEach(({m,i})=>{
      const d=document.createElement('div'); d.className='v-item'+(i===cur3D?' sel':'');
      const col=(SERIES_META[m.series]||{}).col||'#ccc';
      const rightTxt = m.MPO!=null ? 'MPO '+fmt(m.MPO,2) : (m.Vinardo!=null ? fmt(m.Vinardo,2) : '—');
      d.innerHTML=`<span class="dot" style="background:${col}"></span><span class="mono" style="font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${m.id}</span><span class="mono note" style="margin-left:auto">${rightTxt}</span>`;
      d.onclick=()=>load3D(i);
      host.appendChild(d);
    });
  });
}
document.getElementById('vSearch').oninput = renderVList;

function styleProt3D(contacts){
  if(!viewer3d) return;
  protM.setStyle({},{});
  if(showCartoon) protM.setStyle({},{cartoon:{color:'#5476a8',opacity:0.84,thickness:0.3}});
  protM.setStyle({resi:KEYRES},{stick:{radius:0.14,colorscheme:'whiteCarbon'},cartoon:showCartoon?{color:'#5476a8',opacity:0.84}:undefined});
  if(contacts.length) protM.setStyle({resi:contacts},{stick:{radius:0.21,colorscheme:'yellowCarbon'},cartoon:showCartoon?{color:'#5476a8',opacity:0.84}:undefined});
}
function drawLabels3D(contacts){
  if(!viewer3d) return;
  labels3d.forEach(l=>viewer3d.removeLabel(l)); labels3d=[];
  if(!showLabel) return;
  const set=new Set(contacts);
  KEYRES.forEach(r=>{
    const ats=protM.selectedAtoms({resi:r,atom:'CA'}); if(!ats.length) return;
    const a=ats[0], hit=set.has(r);
    labels3d.push(viewer3d.addLabel(CRD[String(r)], {position:{x:a.x,y:a.y,z:a.z}, fontSize:11, fontColor:hit?'#ffe49e':'#b8c8e0', backgroundColor:hit?'rgba(95,66,18,.88)':'rgba(16,24,38,.75)', backgroundOpacity:0.9, borderThickness:hit?1:0, borderColor:'#f0c766', inFront:true}));
  });
}
function drawHb3D(){
  if(!viewer3d) return [];
  hb3d.forEach(s=>viewer3d.removeShape(s)); hb3d=[];
  if(!showHb || !ligM) return [];
  const lig=ligM.selectedAtoms({}).filter(a=>a.elem==='N'||a.elem==='O');
  const pro=protM.selectedAtoms({}).filter(a=>a.elem==='N'||a.elem==='O');
  const found=[];
  lig.forEach(l=>pro.forEach(p=>{ const d=Math.hypot(l.x-p.x,l.y-p.y,l.z-p.z); if(d<=3.4){ found.push({d,res:(p.resn||'')+p.resi}); hb3d.push(viewer3d.addCylinder({start:{x:l.x,y:l.y,z:l.z},end:{x:p.x,y:p.y,z:p.z},radius:0.045,dashed:true,dashLength:0.22,gapLength:0.18,fromCap:1,toCap:1,color:'#3ad1be'})); }}));
  return found;
}
function load3D(idx){
  cur3D = idx; const m = ALL3D[idx]; let hb=[];
  if(viewer3d){
    if(ligM){ viewer3d.removeModel(ligM); ligM=null; }
    if(m.mol){ ligM=viewer3d.addModel(m.mol,'sdf'); ligM.setStyle({},{stick:{radius:0.17,colorscheme:'cyanCarbon'},sphere:{radius:0.25,colorscheme:'cyanCarbon'}}); }
    styleProt3D(m.contacts||[]); drawLabels3D(m.contacts||[]); hb=drawHb3D();
    if(ligM){ viewer3d.zoomTo({model:ligM}); viewer3d.zoom(0.55); }
    viewer3d.render();
  }
  document.getElementById('vPoseBadge').textContent = m.src;
  const set=new Set(m.contacts||[]), tags=KEYRES.map(r=>`<span class="tag ${set.has(r)?'hit':''}">${CRD[String(r)]}</span>`).join('');
  const hbTxt=hb.length ? hb.sort((a,b)=>a.d-b.d).slice(0,10).map(x=>`${x.res} (${x.d.toFixed(2)}Å)`).join('、') : '3.4 Å 內無幾何氫鍵';
  const meta=SERIES_META[m.series]||{name:m.series,cls:'p-pol'};
  const byP={}; (m.plip||[]).forEach(p=>{(byP[p.type]=byP[p.type]||[]).push(p.res+(p.dist?` (${p.dist}Å)`:''));});
  const plipRows = Object.keys(byP).length ? Object.entries(byP).map(([t,v])=>`<tr><td class="note">${t}</td><td>${v.join('、')}</td></tr>`).join('') : '<tr><td class="note" colspan="2">此姿勢未執行 PLIP（請參考上方 4.5 Å CRD 接觸與即時氫鍵）</td></tr>';
  document.getElementById('vRight').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap"><h2 class="mono" style="margin:0;font-size:15.5px;color:#fff">${m.id}</h2><span class="pill ${meta.cls}">${meta.name}</span></div>
    <div class="note" style="margin:3px 0 6px">${m.desc||''}</div>
    ${m.png?`<img class="png2d" src="data:image/png;base64,${m.png}" alt="2D">`:''}
    <div class="smi-box">${m.smiles}</div>
    <div style="margin:6px 0 10px"><button id="bCopy3D">複製 SMILES</button> <span id="cpMsg" class="note" style="color:var(--teal)"></span></div>
    <div class="card" style="padding:10px;margin-bottom:9px"><h3>多引擎對接與 MPO</h3><table>
      ${m.MPO!=null?`<tr><td class="note">綜合 MPO 分數</td><td class="n" style="color:var(--gold);font-weight:700">${fmt(m.MPO,3)}</td></tr>`:''}
      <tr><td class="note">smina Vina (pH 7.4)</td><td class="n">${fmt(m.Vina,3)}</td></tr>
      <tr><td class="note">smina Vinardo (pH 7.4)</td><td class="n">${fmt(m.Vinardo,3)}</td></tr>
      <tr><td class="note">GNINA GPU CNNaffinity</td><td class="n">${fmt(m.CNNaff,3)}</td></tr>
    </table></div>
    <div class="card" style="padding:10px;margin-bottom:9px"><h3>PLIP 3.0.1 非共價互動</h3><table>${plipRows}</table></div>
    <div class="card" style="padding:10px;margin-bottom:9px"><h3>CRD 關鍵殘基接觸 (${(m.contacts||[]).length}/7)</h3><div>${tags}</div><div class="note" style="margin-top:5px"><b>即時 3.4 Å 氫鍵：</b>${hbTxt}</div></div>
    <div class="card" style="padding:10px"><h3>ADMET-AI 1.4.0 與成藥性</h3><table>
      <tr><td class="note">hERG ↓</td><td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG,3)}</td><td class="note">DILI ↓</td><td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI,3)}</td></tr>
      <tr><td class="note">AMES ↓</td><td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES,3)}</td><td class="note">Pgp ↓</td><td class="n ${riskCls(m.Pgp,true)}">${fmt(m.Pgp,3)}</td></tr>
      <tr><td class="note">HIA ↑</td><td class="n ${riskCls(m.HIA,false)}">${fmt(m.HIA,3)}</td><td class="note">Bioavail ↑</td><td class="n ${riskCls(m.Bioavail,false)}">${fmt(m.Bioavail,3)}</td></tr>
      <tr><td class="note">QED ↑</td><td class="n">${fmt(m.QED,3)}</td><td class="note">SA Score ↓</td><td class="n">${fmt(m.SA,2)}</td></tr>
      <tr><td class="note">TPSA</td><td class="n">${fmt(m.TPSA,1)}</td><td class="note">Sol (logS)</td><td class="n">${fmt(m.Sol,2)}</td></tr>
      <tr><td class="note">MW</td><td class="n">${fmt(m.MW,1)}</td><td class="note">RotB / F</td><td class="n">${m.RotB??'—'} / ${m.nF??'—'}</td></tr>
    </table><div class="note" style="margin-top:5px"><b>Brenk 警示：</b>${m.BRENK||'無 ✅'}</div></div>`;
  document.getElementById('bCopy3D').onclick=()=>{ navigator.clipboard?.writeText(m.smiles); document.getElementById('cpMsg').textContent='已複製！'; setTimeout(()=>document.getElementById('cpMsg').textContent='',1500); };
  renderVList();
}
function jumpTo3D(uid){
  const idx = ALL3D.findIndex(x=>x.uid===uid);
  if(idx>=0) cur3D = idx;
  document.querySelector('[data-tab="t3"]').click();
  if(viewerInit) load3D(cur3D);
}
document.getElementById('bReset').onclick=()=>{if(viewer3d&&ligM){viewer3d.zoomTo({model:ligM});viewer3d.zoom(0.55);viewer3d.render();}};
document.getElementById('bSurf').onclick=function(){showSurf=!showSurf;this.classList.toggle('on',showSurf);if(!viewer3d)return;if(surf3d){viewer3d.removeSurface(surf3d);surf3d=null;}if(showSurf){surf3d=viewer3d.addSurface($3Dmol.SurfaceType.MS,{opacity:0.58,color:'#3b5985'},{model:protM});if(surf3d?.then)surf3d.then(s=>{surf3d=s;viewer3d.render();});}viewer3d.render();};
document.getElementById('bLabel').onclick=function(){showLabel=!showLabel;this.classList.toggle('on',showLabel);drawLabels3D(ALL3D[cur3D].contacts||[]);viewer3d?.render();};
document.getElementById('bHb').onclick=function(){showHb=!showHb;this.classList.toggle('on',showHb);drawHb3D();viewer3d?.render();};
document.getElementById('bCartoon').onclick=function(){showCartoon=!showCartoon;this.classList.toggle('on',showCartoon);styleProt3D(ALL3D[cur3D].contacts||[]);viewer3d?.render();};
document.getElementById('bSpin').onclick=function(){spinOn=!spinOn;this.classList.toggle('on',spinOn);viewer3d?.spin(spinOn?'y':false);};

/* ---------------- Boot ---------------- */
drawB5Pareto();
renderB5Table();
renderSarTable();
window.addEventListener('resize', ()=>{
  if(document.getElementById('t5b').classList.contains('active')) drawB5Pareto();
  if(document.getElementById('t5a').classList.contains('active')) drawPareto();
  if(document.getElementById('t1').classList.contains('active')) drawEvolution();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
