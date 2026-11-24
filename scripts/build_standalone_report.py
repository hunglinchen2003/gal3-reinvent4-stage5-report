#!/usr/bin/env python3
"""
Builds a completely self-contained standalone HTML report (`docs/index.html`)
combining:
  - Stage 1-4 REINVENT4 + DynamicBind + smina + ADMET-AI evolution (93 molecules from gal3_explorer.html)
  - Stage 5 Commercial Phenylamine & 2xF->1xNO2 local benchmark (40 molecules: smina Vina/Vinardo + GNINA GPU CNN + ADMET-AI + RDKit SA/QED/Brenk + 3D poses)
  - Embedded 3Dmol-min.js (zero external CDN dependencies; works offline & on GitHub Pages)
"""
import io
import os
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
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUT_HTML = DOCS_DIR / "index.html"


def mol_to_png_b64(smiles, size=(420, 210)):
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
    assert m_payload, "Payload JSON not found!"
    old_data = json.loads(m_payload.group(1))

    # Extract Stage 1-4 summary (80 cycles) + 6 leaders + 7 refs
    cycles = []
    leaders_refs = []
    for m in old_data["molecules"]:
        a = m.get("admet") or {}
        e = m.get("extra") or {}
        s = m.get("smina") or {}
        mol = Chem.MolFromSmiles(m["smiles"])
        nF = sum(1 for at in mol.GetAtoms() if at.GetSymbol() == "F") if mol else 0
        rec = {
            "id": m["id"],
            "group": m["group"],
            "cycle": m.get("cycle"),
            "caption": m.get("caption") or "",
            "smiles": m["smiles"],
            "aff": m.get("aff"),
            "aff_sd": m.get("aff_sd"),
            "contacts": m.get("contacts") or [],
            "hERG": a.get("hERG"),
            "DILI": e.get("DILI"),
            "AMES": e.get("AMES"),
            "Pgp": e.get("Pgp_Broccatelli"),
            "BBB": a.get("BBB"),
            "HIA": a.get("HIA"),
            "Caco2": a.get("Caco2"),
            "Bioavail": e.get("Bioavailability_Ma"),
            "QED": a.get("QED"),
            "TPSA": a.get("TPSA"),
            "Sol": a.get("solubility"),
            "cLogP": e.get("cLogP"),
            "MW": e.get("MW"),
            "RotB": e.get("RotB"),
            "nF": nF,
            "vina_old": (s.get("pH74_vina") or {}).get("mean"),
            "vina_old_sd": (s.get("pH74_vina") or {}).get("sd"),
            "vinardo_old": (s.get("pH74_vinardo") or {}).get("mean"),
            "vinardo_old_sd": (s.get("pH74_vinardo") or {}).get("sd"),
            "plip": m.get("plip") or [],
        }
        if m["group"] in ("stage1", "stage2", "stage3", "stage4"):
            cycles.append(rec)
        else:
            # Keep 3D SDF & 2D PNG for leaders and refs
            rec["mol"] = m.get("mol") or ""
            rec["png"] = m.get("png") or ""
            leaders_refs.append(rec)

    cycles.sort(key=lambda x: x["cycle"] or 0)

    # Load Stage 5 SAR results
    print("Loading Stage 5 SAR benchmark results...")
    stage5 = []
    with open(SAR_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sdf_path = Path(row["best_pose_sdf"])
            mol_block = sdf_path.read_text(encoding="utf-8", errors="ignore") if sdf_path.exists() else ""
            contacts = [int(x) for x in row["CRD_contacts"].split(",") if x.strip()]
            stage5.append({
                "id": row["id"],
                "series": row["series"],
                "desc": row["desc"],
                "smiles": row["smiles"],
                "MW": safe_float(row["MW"]),
                "cLogP": safe_float(row["cLogP"]),
                "TPSA": safe_float(row["TPSA"]),
                "RotB": int(float(row["RotB"])),
                "nF": int(float(row["nF"])),
                "QED": safe_float(row["QED"]),
                "SA": safe_float(row["SA"]),
                "PAINS": row["PAINS"],
                "BRENK": row["BRENK"],
                "hERG": safe_float(row["hERG"]),
                "DILI": safe_float(row["DILI"]),
                "AMES": safe_float(row["AMES"]),
                "Pgp": safe_float(row["Pgp"]),
                "BBB": safe_float(row["BBB"]),
                "HIA": safe_float(row["HIA"]),
                "Caco2": safe_float(row["Caco2"]),
                "Bioavail": safe_float(row["Bioavail"]),
                "Sol": safe_float(row["Sol"]),
                "Vina": safe_float(row["Vina_mean"]),
                "Vina_sd": safe_float(row["Vina_sd"]),
                "Vina_best": safe_float(row["Vina_best"]),
                "Vinardo": safe_float(row["Vinardo_mean"]),
                "Vinardo_sd": safe_float(row["Vinardo_sd"]),
                "Vinardo_best": safe_float(row["Vinardo_best"]),
                "CNNscore": safe_float(row["GNINA_CNNscore"]),
                "CNNaff": safe_float(row["GNINA_CNNaff"]),
                "contacts": contacts,
                "png": mol_to_png_b64(row["smiles"]),
                "mol": mol_block,
            })

    payload = {
        "protein_pdb": old_data["protein_pdb"],
        "crd": old_data["crd"],
        "cycles": cycles,
        "leaders_refs": leaders_refs,
        "stage5": stage5,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    print(f"Payload JSON size: {len(payload_json)/1e6:.2f} MB")

    html_doc = HTML_TEMPLATE.replace("__THREEMOL_SCRIPT__", threemol_script).replace("__PAYLOAD_JSON__", payload_json)
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    # Also save a convenience copy at workspace root
    (BASE_DIR / "gal3_stage5_report.html").write_text(html_doc, encoding="utf-8")
    print(f"Wrote standalone HTML: {OUT_HTML} ({OUT_HTML.stat().st_size/1e6:.2f} MB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-Hant-TW">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Galectin-3 AI 藥物設計報告 — Stage 1–4 演化診斷與 Stage 5 市售苯胺 / 2F→NO₂ 實測</title>
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

header{background:linear-gradient(135deg,#121c30 0%,#0d1525 60%,#132436 100%);
  border-bottom:1px solid var(--line);padding:18px 24px 14px}
.hdr-top{display:flex;flex-wrap:wrap;align-items:flex-start;gap:16px;justify-content:space-between}
header h1{margin:0 0 4px;font-size:20px;font-weight:700;letter-spacing:.3px;color:#fff}
header .sub{color:var(--muted);font-size:12.5px}
.warn-banner{background:rgba(92,45,22,.55);border:1px solid #8a4f2d;color:#ffe0c2;
  border-radius:8px;padding:8px 12px;font-size:12px;max-width:640px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin-top:14px}
.kpi{background:rgba(20,29,46,.85);border:1px solid var(--line);border-radius:9px;padding:10px 13px}
.kpi .lbl{font-size:11.5px;color:var(--muted)}
.kpi .val{font-size:19px;font-weight:700;margin:2px 0;color:var(--gold);font-variant-numeric:tabular-nums}
.kpi .desc{font-size:11px;color:var(--muted)}

nav.tabs{display:flex;gap:8px;padding:10px 24px;background:var(--bg2);
  border-bottom:1px solid var(--line);position:sticky;top:0;z-index:20;flex-wrap:wrap}
nav.tabs button{padding:7px 15px;font-weight:600;border-radius:8px}
nav.tabs button.active{background:linear-gradient(135deg,#1d5c55,#21446b);border-color:var(--teal);color:#fff}

.container{max-width:1460px;margin:0 auto;padding:18px 24px 40px}
.tab-pane{display:none}
.tab-pane.active{display:block}

.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:15px 17px;margin-bottom:16px}
.card h2{margin:0 0 8px;font-size:15.5px;color:var(--gold);display:flex;align-items:center;gap:8px}
.card h3{margin:0 0 7px;font-size:13.5px;color:var(--teal)}
.note{font-size:12px;color:var(--muted);line-height:1.55}

.callout{border-left:3px solid var(--teal);background:rgba(58,209,190,.08);
  padding:10px 13px;border-radius:0 8px 8px 0;margin:10px 0;font-size:12.5px}
.callout.warn{border-left-color:var(--orange);background:rgba(242,166,90,.09)}
.callout.bad{border-left-color:var(--red);background:rgba(240,108,122,.09)}

table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px 8px;border-bottom:1px solid rgba(38,53,82,.7);text-align:left;vertical-align:middle}
th{color:var(--muted);font-weight:600;background:rgba(16,23,38,.7);position:sticky;top:0;white-space:nowrap}
th.sort{cursor:pointer;user-select:none}
th.sort:hover{color:#fff}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,Consolas,monospace}
tr.clickable{cursor:pointer}
tr.clickable:hover{background:#1d2b45}
tr.sel-row{background:#223a5c !important}

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

.good{color:#7de8a8;font-weight:600}
.mid{color:#f5ce78}
.bad{color:#ff8f9b;font-weight:600}

.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.scroll-x{overflow-x:auto}
.scroll-y{max-height:540px;overflow-y:auto}

/* 3D Viewer tab layout */
.v-layout{display:grid;grid-template-columns:270px 1fr 370px;gap:12px;height:760px}
@media(max-width:1180px){.v-layout{grid-template-columns:1fr;height:auto}}
.v-side,.v-right{background:var(--panel);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.v-list{overflow-y:auto;flex:1;padding:6px}
.v-item{display:flex;align-items:center;gap:7px;padding:5px 7px;border-radius:6px;cursor:pointer;font-size:12px}
.v-item:hover{background:#1d2b45}
.v-item.sel{background:#254266;outline:1px solid #4c82bd}
.dot{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
.v-center{background:var(--panel);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;min-height:480px;overflow:hidden}
.v-tools{display:flex;gap:6px;padding:7px 10px;background:var(--bg2);border-bottom:1px solid var(--line);flex-wrap:wrap;align-items:center}
#viewwrap{position:relative;flex:1;min-height:420px}
#viewer3d{position:absolute;inset:0}
#vcap{position:absolute;left:10px;bottom:8px;background:rgba(11,16,27,.8);padding:5px 9px;border-radius:6px;font-size:11px;color:var(--muted);max-width:78%}
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
      <h1>Galectin-3 AI 藥物設計報告 — Stage 1–4 演化診斷與 Stage 5 市售苯胺 / 2F→NO₂ 實測</h1>
      <div class="sub">REINVENT 4.9.34 × DynamicBind × smina (Vina / Vinardo) × GNINA v1.3.3 GPU CNN × ADMET-AI × PDB 5E89 CRD</div>
    </div>
    <div class="warn-banner">
      <b>科學誠實聲明：</b>所有親和力分數（DynamicBind、smina Vina/Vinardo、GNINA CNNaffinity）皆為<b>計算對接代理值</b>，非實驗 $K_d$。Stage 4 出現嚴重的 DynamicBind 多氟膨脹（reward hacking），故 Stage 5 全面改採 <b>smina 雙打分 + GNINA GPU CNN + ADMET-AI + 市售合成/專利篩選</b> 多引擎共識評估。
    </div>
  </div>
  <div class="kpis" id="kpiBar"></div>
</header>

<nav class="tabs" id="tabNav">
  <button class="active" data-tab="t1">1. 多代演化與 ADMET 取捨 (Stage 1–4)</button>
  <button data-tab="t2">2. Stage 5 實測：市售苯胺 &amp; 2F→NO₂ SAR</button>
  <button data-tab="t3">3. 互動式 3D 結合姿勢 &amp; 2D 結構檢視器</button>
  <button data-tab="t4">4. 專利 FTO、合成路線與本機 REINVENT 4 迭代指南</button>
</nav>

<div class="container">
  <!-- ================= TAB 1 ================= -->
  <section class="tab-pane active" id="t1">
    <div class="grid2">
      <div class="card">
        <h2>多代演化軌跡（80 個 Cycle：Stage 1 → 4）</h2>
        <div class="toolbar">
          <span class="note">顯示指標：</span>
          <button class="on" data-ev="aff">DynamicBind 分數</button>
          <button data-ev="hERG">hERG 風險</button>
          <button data-ev="QED">QED 類藥性</button>
          <button data-ev="nF">氟原子數 (F)</button>
        </div>
        <canvas id="evCanvas" height="270"></canvas>
        <div class="note" style="margin-top:8px">
          <b>演化關鍵轉折：</b>Stage 1（藍）為半乳糖-三唑骨架 → Stage 2（綠）非醣類探索 → Stage 3（橘）發現 4-氨基哌啶苯脲（但 hERG 飆至 0.97）→ Stage 4（粉紅）導入 <b>-COOH 兩性離子化</b>成功壓低 hERG 至 ~0.31，卻因單次 DynamicBind 獎勵引發 <b>4F–5F 多氟苯胺爆發</b>與 QED 崩跌（0.14–0.20）。
        </div>
      </div>

      <div class="card">
        <h2>參考配體校準與 Reward Hacking 診斷</h2>
        <div class="callout warn">
          <b>為什麼不能只看 DynamicBind 單次分數？</b><br>
          1. <b>尺寸/多氟偏差：</b>實驗上口服 <b>GB1211 (25 nM)</b> 與 <b>GB1107 (37 nM)</b> 親和力等同 TD-139 (14–68 nM)，但 DynamicBind 僅給 GB1107/GB1211 <b>5.13–5.17</b>，卻給 MW 649 的 TD-139 <b>6.43</b>、給 4F–6F 產生分子高達 <b>8.0–9.17</b>。<br>
          2. <b>Winner's Curse：</b><code>c317_00</code> 單次 9.17 → 5 種子重跑跌至 6.49±0.52（Vinardo 僅 −6.90，偏位結合 Asn119/Lys233）。<br>
          3. <b>正向訊號：</b>唯一低變異領先分子 <b><code>c310_15 ★</code></b> 在 Vina (−7.32) 與 Vinardo (−8.36) <b>均優於口服臨床骨架 GB1107 (−7.00 / −7.95)</b>！
        </div>
        <div class="scroll-x">
          <table id="refCalTable"></table>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>跨代 ADMET 取捨矩陣（吸入雙醣 vs 口服單醣 vs 兩性離子非醣類）</h2>
      <div class="scroll-x"><table id="admetMatrixTable"></table></div>
    </div>
  </section>

  <!-- ================= TAB 2 ================= -->
  <section class="tab-pane" id="t2">
    <div class="card">
      <h2>核心問題實測結論：「兩個 F 用一個 NO₂ 取代」vs「市售安全拉電子苯胺」</h2>
      <div id="nitroVerdictBox" class="grid3"></div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>Stage 5 多引擎 Pareto 散佈圖（點選任一點可跳至 3D 檢視器）</h2>
        <div class="toolbar">
          <label class="note">X 軸：</label>
          <select id="scX">
            <option value="Vina" selected>smina Vina pH7.4 (越負越好)</option>
            <option value="Vinardo">smina Vinardo pH7.4 (越負越好)</option>
            <option value="CNNaff">GNINA GPU CNNaffinity (越高越好)</option>
            <option value="CNNscore">GNINA GPU CNNscore (越高越好)</option>
          </select>
          <label class="note">Y 軸：</label>
          <select id="scY">
            <option value="AMES" selected>AMES 致突變風險 (越低越好)</option>
            <option value="DILI">DILI 肝毒風險 (越低越好)</option>
            <option value="hERG">hERG 心臟毒性 (越低越好)</option>
            <option value="QED">QED 類藥性 (越高越好)</option>
            <option value="SA">SA 合成難度 (越低越好)</option>
            <option value="CNNaff">GNINA CNNaffinity (越高越好)</option>
          </select>
        </div>
        <canvas id="paretoCanvas" height="310"></canvas>
        <div class="note" id="paretoHover" style="margin-top:6px">滑鼠移至圓點查看詳情，點擊直接開啟 3D 結合姿勢。</div>
      </div>

      <div class="card">
        <h2>系列群組實測均值對照（40 個本機實測分子）</h2>
        <div class="scroll-x"><table id="seriesSummaryTable"></table></div>
        <div class="callout" style="margin-top:12px">
          <b>實測關鍵發現（本機 RTX 4060 + 3 種子 smina + GNINA CNN + ADMET-AI）：</b>
          <div id="autoFindings" style="margin-top:4px"></div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Stage 5 完整 40 分子實測總表（點擊欄位排序、點擊列直接檢視 3D 姿勢）</h2>
      <div class="toolbar">
        <input id="sarSearch" type="text" placeholder="搜尋 ID / 取代基 / SMILES…" style="width:240px">
        <select id="seriesFilter">
          <option value="ALL">全部系列 (40)</option>
          <option value="ref">臨床參考配體 (ref)</option>
          <option value="leader">Stage 4 領先對照 (leader)</option>
          <option value="A_nitro">A組：2F→1NO₂ 硝基苯胺 (5)</option>
          <option value="A_ewg_safe">A組：安全拉電子 -CN/-CF₃/-SO₂Me (7)</option>
          <option value="A_halogen">A組：廉價氯/氟苯胺 (3)</option>
          <option value="A_heteroaryl">A組：缺電子吡啶胺 (3)</option>
          <option value="A_polar">A組：極性醯胺 (1)</option>
          <option value="B_dual_comm">B組：雙環全改市售廉價原料 (12)</option>
          <option value="C_rigid">C組：縮短/剛性化連接鏈 (4)</option>
        </select>
        <label class="note"><input type="checkbox" id="onlyCleanAmes"> 僅顯示 AMES &lt; 0.45</label>
        <label class="note"><input type="checkbox" id="onlyZeroAlert"> 僅顯示無硝基/無 Michael 警示</label>
        <button id="btnExportCsv">下載本表 CSV</button>
      </div>
      <div class="scroll-x scroll-y"><table id="sarTable"></table></div>
    </div>
  </section>

  <!-- ================= TAB 3 ================= -->
  <section class="tab-pane" id="t3">
    <div class="v-layout">
      <div class="v-side">
        <div style="padding:8px;border-bottom:1px solid var(--line)">
          <input id="vSearch" type="text" placeholder="篩選分子 ID / 描述…" style="width:100%">
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
          <div id="vcap">受體：Galectin-3 CRD（PDB 5E89 A 鏈）。黃色側鏈＝4.5 Å 內接觸到的 CRD 關鍵殘基（Arg144、His158、Arg162、Asn174、Trp181、Glu184、Arg186）。左鍵旋轉、滾輪縮放、右鍵平移。</div>
        </div>
      </div>
      <div class="v-right">
        <div class="v-right-in" id="vRight"></div>
      </div>
    </div>
  </section>

  <!-- ================= TAB 4 ================= -->
  <section class="tab-pane" id="t4">
    <div class="grid2">
      <div class="card">
        <h2>合成可行性：4-Aminopiperidine 雙脲模組化 2–3 步合成</h2>
        <div class="callout">
          相較於 TD-139 / GB1107 / GB1211 需多步立體選擇性硫代醣苷化與疊氮-炔 Click 反應（6–10 步），Stage 5 的非醣類雙脲骨架可由 <b>3 個廉價市售建構單元</b>直接組裝，完全沒有手性中心（0 stereocenters）：
        </div>
        <pre class="codeblock">
[建構單元 1] 市售保護核：tert-butyl (1-(3-aminopropyl)piperidin-4-yl)carbamate
             （或 Boc-4-aminopiperidine + 3-bromopropylphthalimide）

Step 1（左端脲鍵）：一級胺 + 市售異氰酸酯（或 CDI / triphosgene + 廉價氨基苯甲酸酯）
       例如：methyl 3-aminobenzoate / methyl 5-amino-2-fluorobenzoate
             / methyl 3-amino-5-nitrobenzoate（皆 &lt; $30 / 100g）

Step 2（去保護）：TFA / DCM 脫除 Boc → 游離 4-氨基哌啶

Step 3（右端脲鍵 + 水解）：
       + 市售異氰酸酯 Ar_R-N=C=O（如 3-CF₃ / 4-F-3-CF₃ / 3-CN / 3-Cl-4-F / 3-NO₂）
       → LiOH/THF-H₂O 水解甲酯 → 目標兩性離子抑制劑（直接析晶或逆相純化）
        </pre>
        <h3>為什麼淘汰四氟/五氟苯胺？</h3>
        <ul class="note">
          <li><b>親核性太差：</b>4–5 個氟強力拉電子導致苯胺氨基幾乎喪失親核性，CDI 反應產率極低且需使用劇毒光氣/三光氣。</li>
          <li><b>原料價格差距 10–30 倍：</b><code>2,3,5-trifluoroaniline</code> 與 <code>5-amino-2,3-difluorobenzoic acid</code> 屬特化試劑；改用 <code>3-CF₃</code>、<code>4-F-3-CF₃</code>、<code>3-CN</code>、<code>3-Cl-4-F</code>、<code>3-NO₂</code> 苯胺與 <code>3-aminobenzoic acid</code> / <code>5-amino-2-fluorobenzoic acid</code> 後，全路線皆為大宗試劑且 <b>SA Score 顯著下降</b>。</li>
        </ul>
      </div>

      <div class="card">
        <h2>專利迴避（FTO）與硝基雙軌策略</h2>
        <table>
          <thead><tr><th>化學特徵</th><th>Galecto 臨床專利群 (TD-139/GB1107/GB1211)</th><th>本案 Stage 5 候選系列</th></tr></thead>
          <tbody>
            <tr><td>核心骨架</td><td>硫代雙半乳糖 / α-D-硫代半乳糖苷 + 1,2,3-三唑</td><td><b>非醣類 4-氨基哌啶雙脲（無手性中心）</b></td></tr>
            <tr><td>芳環取代基</td><td>3-氟苯基、3,4,5-三氟苯基（專利密集區）</td><td><b>3-CF₃、4-F-3-CF₃、3-CN、3-Cl-4-F、6-CF₃-吡啶、3-NO₂ 探針</b></td></tr>
            <tr><td>給藥與 ADMET</td><td>TD-139 僅吸入；GB1107/1211 hERG 0.60–0.71、DILI 0.94</td><td><b>兩性離子口服導向：hERG 0.39–0.50、Pgp &lt;0.10、DILI 降至 0.41–0.54（同版 ADMET-AI 1.4.0）</b></td></tr>
          </tbody>
        </table>
        <div class="callout warn" style="margin-top:12px">
          <b>關於「2×F → 1×NO₂」的專利與開發雙軌建議（基於本機實測）：</b><br>
          • <b>實測驗證：</b>如 Tab 2 本機 ADMET-AI 1.4.0 實測所示，導入 <code>-NO₂</code> 確實能維持或超越多氟對照的結合力（例如零氟的 <b><code>C03_C2_L3NH5NO2_R3CN</code></b> 達 Vina −8.30 / Vinardo −8.38），但代價是 <b>AMES 風險放大 ~3.4 倍（從 c310_15 的 0.138 升至 0.407–0.545）、DILI 升至 0.677–0.778、TPSA 飆高至 166–190 Å² 壓低口服吸收</b>，並觸發 <code>nitro_group</code> 結構警示。<br>
          • <b>建議策略：</b>將 <code>-NO₂</code> 系列（如 <code>C03</code>、<code>A04</code>、<code>A05</code>）作為<b>低成本快速體外實驗探針（SPR/FP）與專利說明書實施例</b>；主攻推進口服候選藥則優先鎖定實測表現極佳且 <b>AMES 乾淨（0.10–0.28）</b> 的市售安全拉電子苯胺：<b><code>B07_L5NH2FBA_R3SO2Me</code></b>（僅 1 個 F、hERG 0.391、AMES 0.131、Vina −8.13 / Vinardo −7.62、6/7 CRD）、<b><code>B09_L3NHBA_R3Cl4F</code></b>（MW 492、SA 2.23 最易合成）、<b><code>B12_L3NHBA_R6CF3Py3</code></b>（全場最低 DILI 0.405）、<b><code>C01_C2_L5NH2FBA_R3F5CN</code></b>（0 Brenk 警示、QED 0.407）與 <b><code>A09_R_3CF3</code></b>（Vina −8.66）！
        </div>
      </div>
    </div>

    <div class="card">
      <h2>本機軟體架構（WSL2 Ubuntu-22.04 + NVIDIA GeForce RTX 4060）與 Stage 5 啟動指令</h2>
      <div class="grid2">
        <div>
          <h3>已建置完成的本機環境與工具</h3>
          <table>
            <thead><tr><th>環境 / 路徑</th><th>核心軟體與版本</th><th>用途</th></tr></thead>
            <tbody>
              <tr><td class="mono">conda: reinvent4</td><td>Python 3.11, <b>REINVENT 4.9.34</b>, PyTorch 2.13+cu126 (RTX 4060)</td><td>Mol2Mol / LibInvent 強化學習分子生成</td></tr>
              <tr><td class="mono">conda: gal3_eval</td><td>Python 3.10, <b>ADMET-AI 1.4.0</b>, OpenBabel 3.2.1, PLIP 3.0.1, Meeko 0.8, RDKit</td><td>批次 ADMET、質子化、3D 互動與打分服務</td></tr>
              <tr><td class="mono">~/aidd_tools/bin/</td><td><b>smina</b> (Vina/Vinardo) &amp; <b>gnina-gpu v1.3.3</b> (GPU CNN)</td><td>快速內迴圈對接 + GPU CNN 交叉驗證</td></tr>
              <tr><td class="mono">conda: dynamicbind + relax</td><td>PyTorch 2.0.1+cu117 (RTX 4060), OpenMM 8.5.2, DynamicBind repo</td><td>外迴圈 Top 15 候選 5 種子柔性對接驗證</td></tr>
            </tbody>
          </table>
        </div>
        <div>
          <h3>階段 5 REINVENT 4 啟動步驟（WSL2）</h3>
          <pre class="codeblock"># 1. 放入官方先驗模型（因 Zenodo 擋 CLI 403，請由瀏覽器下載或從 Mac 複製）：
#    REINVENT4 Priors (Zenodo 15641296):
#      -> ~/aidd_tools/REINVENT4/priors/mol2mol_medium_similarity.prior
#    DynamicBind v2 Weights (Zenodo 10183369):
#      -> ~/aidd_tools/DynamicBind/workdir/

# 2. 啟動 Stage 5 多目標強化學習（已內建反多氟/反Michael警示 + smina + ADMET-AI）：
conda activate reinvent4
reinvent /mnt/d/2026@AI_reinvent4/stage5_reinvent4/stage5_staged_learning.toml

# 3. 隨時重跑或擴充市售苯胺 SAR 批次基準測試：
conda activate gal3_eval
python /mnt/d/2026@AI_reinvent4/scripts/benchmark_aniline_sar.py</pre>
        </div>
      </div>
    </div>
  </section>
</div>

<script id="payload" type="application/json">__PAYLOAD_JSON__</script>
__THREEMOL_SCRIPT__
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const CYCLES = DATA.cycles;
const LR = DATA.leaders_refs;
const S5 = DATA.stage5;
const CRD = DATA.crd;
const KEYRES = Object.keys(CRD).map(Number);

const SERIES_META = {
  ref:          {name:'臨床參考', cls:'p-ref',   col:'#be95ff'},
  leader:       {name:'Stage4領先', cls:'p-lead', col:'#f0c766'},
  A_nitro:      {name:'A: 2F→NO₂', cls:'p-nitro', col:'#f06c7a'},
  A_ewg_safe:   {name:'A: 安全EWG', cls:'p-safe', col:'#3ad1be'},
  A_halogen:    {name:'A: 廉價鹵素', cls:'p-hal', col:'#6ea8fe'},
  A_heteroaryl: {name:'A: 缺電子吡啶', cls:'p-het', col:'#a78bfa'},
  A_polar:      {name:'A: 極性醯胺', cls:'p-pol', col:'#94a3b8'},
  B_dual_comm:  {name:'B: 雙環市售', cls:'p-dual', col:'#6fd08c'},
  C_rigid:      {name:'C: 剛性連接鏈', cls:'p-rig', col:'#f2a65a'}
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
    if(btn.dataset.tab==='t1') drawEvolution();
    if(btn.dataset.tab==='t2') drawPareto();
    if(btn.dataset.tab==='t3'){
      if(!viewerInit){ initViewer(); viewerInit=true; }
      else if(viewer3d){ viewer3d.resize(); viewer3d.render(); }
    }
  };
});

/* ---------------- KPIs ---------------- */
(function buildKPIs(){
  const nonRefS5 = S5.filter(m=>!['ref','leader'].includes(m.series));
  const safeS5 = nonRefS5.filter(m=>m.AMES<0.45 && m.hERG<0.45 && !m.BRENK.includes('nitro') && !m.BRENK.includes('Michael'));
  const bestSafe = [...safeS5].sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo))[0];
  const nitroMeanAmes = mean(S5.filter(m=>m.series==='A_nitro').map(m=>m.AMES));
  const safeMeanAmes = mean(S5.filter(m=>m.series==='A_ewg_safe').map(m=>m.AMES));
  const c310 = S5.find(m=>m.id==='c310_15');
  const gb1107 = S5.find(m=>m.id==='GB1107');
  document.getElementById('kpiBar').innerHTML = `
    <div class="kpi"><div class="lbl">總分析分子數</div><div class="val">${CYCLES.length + LR.length} + ${S5.length}</div><div class="desc">Stage 1–4 歷史 93 分子 + Stage 5 本機實測 40 分子</div></div>
    <div class="kpi"><div class="lbl">最佳安全市售苯胺先導</div><div class="val" style="color:var(--teal);font-size:15px">${bestSafe?bestSafe.id:'—'}</div><div class="desc">Vina ${fmt(bestSafe?.Vina)} / Vinardo ${fmt(bestSafe?.Vinardo)} · AMES ${fmt(bestSafe?.AMES)} · hERG ${fmt(bestSafe?.hERG)}</div></div>
    <div class="kpi"><div class="lbl">2F→1NO₂ vs 安全EWG (AMES)</div><div class="val" style="color:var(--red)">${fmt(nitroMeanAmes)} <span style="color:var(--muted);font-size:13px">vs</span> <span style="color:var(--green)">${fmt(safeMeanAmes)}</span></div><div class="desc">硝基組 AMES 顯著躍升；-CN/-CF₃/-SO₂Me 維持低風險</div></div>
    <div class="kpi"><div class="lbl">對照基準 (Vina / Vinardo)</div><div class="val" style="font-size:15px">c310_15: ${fmt(c310?.Vina)}/${fmt(c310?.Vinardo)}</div><div class="desc">GB1107 (37 nM 口服): ${fmt(gb1107?.Vina)} / ${fmt(gb1107?.Vinardo)}</div></div>
    <div class="kpi"><div class="lbl">本機 GPU 運算管線</div><div class="val" style="color:var(--green);font-size:16px">RTX 4060 就緒</div><div class="desc">REINVENT 4.9.34 + smina + GNINA CNN + ADMET-AI</div></div>
  `;
})();

/* ---------------- Tab 1: Evolution chart ---------------- */
let evMetric = 'aff';
document.querySelectorAll('[data-ev]').forEach(b=>{
  b.onclick = ()=>{
    document.querySelectorAll('[data-ev]').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');
    evMetric = b.dataset.ev;
    drawEvolution();
  };
});

function drawEvolution(){
  const c = document.getElementById('evCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 560, H = 270;
  c.width = W*dpr; c.height = H*dpr; c.style.height = H+'px';
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const L=46, R=16, T=22, B=32, pw=W-L-R, ph=H-T-B;
  const vals = CYCLES.map(m=>m[evMetric]).filter(v=>v!=null);
  const lo = Math.min(...vals, evMetric==='aff'?3.5:0), hi = Math.max(...vals, evMetric==='QED'||evMetric==='hERG'?1:0) * 1.05;
  const X = i => L + pw*i/(CYCLES.length-1);
  const Y = v => T + ph*(1-(v-lo)/(hi-lo||1));
  const stageCols = {stage1:'#6ea8fe', stage2:'#6fd08c', stage3:'#f2a65a', stage4:'#f06c7a'};

  // stage shading
  [0,20,40,60].forEach((start,si)=>{
    const x1 = X(start), x2 = X(Math.min(start+19, CYCLES.length-1));
    g.fillStyle = ['rgba(110,168,254,.07)','rgba(111,208,140,.07)','rgba(242,166,90,.07)','rgba(240,108,122,.08)'][si];
    g.fillRect(x1, T, x2-x1, ph);
    g.fillStyle = '#95a5c2'; g.font='11px sans-serif'; g.textAlign='center';
    g.fillText('Stage '+(si+1), (x1+x2)/2, T-6);
  });
  // grid
  g.strokeStyle='#24334f'; g.lineWidth=1; g.fillStyle='#8c9cb8'; g.font='10.5px monospace'; g.textAlign='right';
  for(let k=0;k<=4;k++){
    const v = lo+(hi-lo)*k/4, y = Y(v);
    g.beginPath(); g.moveTo(L,y); g.lineTo(W-R,y); g.stroke();
    g.fillText(v.toFixed(evMetric==='nF'?0:2), L-6, y+3);
  }
  // TD-139 reference line for DynamicBind aff
  if(evMetric==='aff'){
    const ytd = Y(6.434);
    g.strokeStyle='#be95ff'; g.setLineDash([5,4]); g.beginPath(); g.moveTo(L,ytd); g.lineTo(W-R,ytd); g.stroke(); g.setLineDash([]);
    g.fillStyle='#be95ff'; g.textAlign='right'; g.fillText('TD-139 (6.43)', W-R-4, ytd-5);
  }
  // line + points
  g.strokeStyle='#4a638f'; g.lineWidth=1.6; g.beginPath();
  let started=false;
  CYCLES.forEach((m,i)=>{
    const v = m[evMetric]; if(v==null) return;
    started ? g.lineTo(X(i),Y(v)) : g.moveTo(X(i),Y(v));
    started=true;
  });
  g.stroke();
  CYCLES.forEach((m,i)=>{
    const v = m[evMetric]; if(v==null) return;
    g.fillStyle = stageCols[m.group]||'#ccc';
    g.beginPath(); g.arc(X(i),Y(v),3.2,0,Math.PI*2); g.fill();
  });
  g.fillStyle='#8c9cb8'; g.textAlign='center';
  [0,19,20,39,40,59,60,79].forEach(i=>{
    if(i<CYCLES.length) g.fillText('c'+String(CYCLES[i].cycle).padStart(3,'0'), X(i), H-10);
  });
}

/* ---------------- Tab 1: Tables ---------------- */
(function buildTab1Tables(){
  const pkdMap = {'TD-139':'14–68 nM','GB1211':'25 nM','GB1107':'37 nM','TAZTDG':'840 nM','thiodigalactoside':'75.4 µM','LacNAc':'~50 µM','lactose':'~200 µM'};
  const rows = LR.map(m=>`<tr>
    <td><b>${m.id}</b></td>
    <td><span class="pill ${m.group==='ref'?'p-ref':'p-lead'}">${m.group==='ref'?'參考':'領先'}</span></td>
    <td class="n">${pkdMap[m.id]||'—'}</td>
    <td class="n">${fmt(m.aff,3)}${m.aff_sd!=null?' ± '+fmt(m.aff_sd,2):''}</td>
    <td class="n">${m.vina_old!=null?fmt(m.vina_old,2):'—'}</td>
    <td class="n">${m.vinardo_old!=null?fmt(m.vinardo_old,2):'—'}</td>
    <td class="n">${m.contacts.length}/7</td>
    <td class="n">${m.nF}</td>
  </tr>`).join('');
  document.getElementById('refCalTable').innerHTML = `
    <thead><tr><th>分子</th><th>類型</th><th class="n">實驗 Kd</th><th class="n">DynamicBind</th><th class="n">舊 Vina</th><th class="n">舊 Vinardo</th><th class="n">CRD</th><th class="n">F 數</th></tr></thead>
    <tbody>${rows}</tbody>`;

  // ADMET matrix comparing TD-139, GB1107, GB1211, c203_16, c310_15, and Top 3 safe Stage 5 hits
  const topSafe3 = [...S5.filter(m=>!['ref','leader'].includes(m.series) && m.AMES<0.45 && !m.BRENK.includes('nitro'))]
    .sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo)).slice(0,3);
  const pick = [
    ...['TD-139','GB1107','GB1211','c310_15'].map(id=>S5.find(x=>x.id===id)),
    LR.find(x=>x.id.startsWith('c203_16')),
    ...topSafe3
  ].filter(Boolean);
  document.getElementById('admetMatrixTable').innerHTML = `
    <thead><tr><th>分子</th><th>定位</th><th class="n">hERG ↓</th><th class="n">DILI ↓</th><th class="n">AMES ↓</th><th class="n">Pgp ↓</th><th class="n">HIA ↑</th><th class="n">Bioavail ↑</th><th class="n">QED ↑</th><th class="n">TPSA</th><th class="n">cLogP</th><th class="n">MW</th><th class="n">F</th></tr></thead>
    <tbody>${pick.map(m=>`<tr>
      <td><b>${m.id}</b></td>
      <td class="note">${m.desc||m.caption||''}</td>
      <td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG,3)}</td>
      <td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI,3)}</td>
      <td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES,3)}</td>
      <td class="n ${riskCls(m.Pgp,true)}">${fmt(m.Pgp,3)}</td>
      <td class="n ${riskCls(m.HIA,false)}">${fmt(m.HIA,3)}</td>
      <td class="n ${riskCls(m.Bioavail,false)}">${fmt(m.Bioavail,3)}</td>
      <td class="n">${fmt(m.QED,3)}</td>
      <td class="n">${fmt(m.TPSA,1)}</td>
      <td class="n">${fmt(m.cLogP,2)}</td>
      <td class="n">${fmt(m.MW,1)}</td>
      <td class="n">${m.nF}</td>
    </tr>`).join('')}</tbody>`;
})();

/* ---------------- Tab 2: Stage 5 SAR Dashboard ---------------- */
(function buildTab2Summary(){
  const byS = {};
  S5.forEach(m=>{ (byS[m.series]=byS[m.series]||[]).push(m); });
  const order = ['ref','leader','A_nitro','A_ewg_safe','A_halogen','A_heteroaryl','A_polar','B_dual_comm','C_rigid'];
  document.getElementById('seriesSummaryTable').innerHTML = `
    <thead><tr><th>系列</th><th class="n">數量</th><th class="n">Vina 均值</th><th class="n">Vinardo 均值</th><th class="n">CNNaff 均值</th><th class="n">hERG</th><th class="n">DILI</th><th class="n">AMES</th><th class="n">QED</th><th class="n">SA ↓</th></tr></thead>
    <tbody>${order.filter(k=>byS[k]).map(k=>{
      const arr = byS[k], meta = SERIES_META[k];
      return `<tr>
        <td><span class="pill ${meta.cls}">${meta.name}</span></td>
        <td class="n">${arr.length}</td>
        <td class="n">${fmt(mean(arr.map(x=>x.Vina)),2)}</td>
        <td class="n">${fmt(mean(arr.map(x=>x.Vinardo)),2)}</td>
        <td class="n">${fmt(mean(arr.map(x=>x.CNNaff)),2)}</td>
        <td class="n ${riskCls(mean(arr.map(x=>x.hERG)),true)}">${fmt(mean(arr.map(x=>x.hERG)),3)}</td>
        <td class="n ${riskCls(mean(arr.map(x=>x.DILI)),true)}">${fmt(mean(arr.map(x=>x.DILI)),3)}</td>
        <td class="n ${riskCls(mean(arr.map(x=>x.AMES)),true)}">${fmt(mean(arr.map(x=>x.AMES)),3)}</td>
        <td class="n">${fmt(mean(arr.map(x=>x.QED)),3)}</td>
        <td class="n">${fmt(mean(arr.map(x=>x.SA)),2)}</td>
      </tr>`;
    }).join('')}</tbody>`;

  // Auto-generate findings & Nitro verdict cards from real data
  const nitro = byS['A_nitro'] || [];
  const safe = byS['A_ewg_safe'] || [];
  const dual = byS['B_dual_comm'] || [];
  const c310 = S5.find(x=>x.id==='c310_15');
  const bestNitro = [...nitro].sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo))[0];
  const bestSafe = [...safe].sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo))[0];
  const bestDualSafe = [...dual.filter(x=>x.AMES<0.45 && !x.BRENK.includes('nitro'))].sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo))[0];

  document.getElementById('nitroVerdictBox').innerHTML = `
    <div class="callout warn" style="margin:0">
      <h3>① 電子與對接：2F → 1NO₂ 確實可行</h3>
      <div>硝基組最佳 <b>${bestNitro?.id}</b>：Vina <b>${fmt(bestNitro?.Vina)}</b>、Vinardo <b>${fmt(bestNitro?.Vinardo)}</b>、GNINA CNNaff <b>${fmt(bestNitro?.CNNaff)}</b>（對比原 c310_15：${fmt(c310?.Vina)} / ${fmt(c310?.Vinardo)} / ${fmt(c310?.CNNaff)}）。證實 1 個 NO₂ 在結合力上完全能取代 2–3 個 F。</div>
    </div>
    <div class="callout bad" style="margin:0">
      <h3>② ADMET 代價：AMES 致突變顯著飆升</h3>
      <div>硝基組 5 個分子平均 AMES = <b>${fmt(mean(nitro.map(x=>x.AMES)),3)}</b>、DILI = <b>${fmt(mean(nitro.map(x=>x.DILI)),3)}</b>，且觸發 RDKit <code>nitro_group</code> 警示。適合做體外 SAR 探針與專利對照例，但不宜單押臨床候選藥。</div>
    </div>
    <div class="callout" style="margin:0">
      <h3>③ 更佳解答：市售安全 EWG 苯胺勝出！</h3>
      <div>右端替換最佳 <b>${bestSafe?.id}</b>（Vina ${fmt(bestSafe?.Vina)} / Vinardo ${fmt(bestSafe?.Vinardo)} / AMES ${fmt(bestSafe?.AMES)}）；雙環全市售化最佳 <b>${bestDualSafe?.id}</b>（Vina ${fmt(bestDualSafe?.Vina)} / Vinardo ${fmt(bestDualSafe?.Vinardo)} / AMES ${fmt(bestDualSafe?.AMES)} / QED ${fmt(bestDualSafe?.QED)} / SA ${fmt(bestDualSafe?.SA)}）！</div>
    </div>`;

  const topAll = [...S5.filter(m=>!['ref','leader'].includes(m.series) && m.AMES<0.45 && !m.BRENK.includes('nitro'))]
    .sort((a,b)=>(a.Vina+a.Vinardo)-(b.Vina+b.Vinardo)).slice(0,5);
  document.getElementById('autoFindings').innerHTML = `
    無硝基/低致突變（AMES &lt; 0.45）綜合對接前 5 名市售苯胺候選：<br>
    ${topAll.map((m,i)=>`<b>#${i+1} ${m.id}</b>（${m.desc}）：Vina <b>${fmt(m.Vina)}</b>、Vinardo <b>${fmt(m.Vinardo)}</b>、CNNaff <b>${fmt(m.CNNaff)}</b>、hERG ${fmt(m.hERG)}、DILI ${fmt(m.DILI)}、AMES ${fmt(m.AMES)}、QED ${fmt(m.QED)}、SA ${fmt(m.SA)}`).join('<br>')}
  `;
})();

/* ---------------- Tab 2: Pareto Scatter ---------------- */
let paretoPts = [];
function drawPareto(){
  const c = document.getElementById('paretoCanvas');
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth || 560, H = 310;
  c.width = W*dpr; c.height = H*dpr; c.style.height = H+'px';
  const g = c.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
  const kx = document.getElementById('scX').value, ky = document.getElementById('scY').value;
  const L=48, R=18, T=16, B=34, pw=W-L-R, ph=H-T-B;
  const xs = S5.map(m=>m[kx]).filter(v=>v!=null), ys = S5.map(m=>m[ky]).filter(v=>v!=null);
  const xlo = Math.min(...xs)-0.25, xhi = Math.max(...xs)+0.25;
  const ylo = Math.min(...ys)*0.92, yhi = Math.max(...ys)*1.06;
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
  // Safe threshold line if Y is AMES/hERG/DILI
  if(['AMES','hERG','DILI'].includes(ky) && 0.45>=ylo && 0.45<=yhi){
    const y45 = Y(0.45);
    g.strokeStyle='rgba(240,108,122,.6)'; g.setLineDash([4,4]);
    g.beginPath(); g.moveTo(L,y45); g.lineTo(W-R,y45); g.stroke(); g.setLineDash([]);
    g.fillStyle='#ff9aa6'; g.textAlign='right'; g.fillText('警示線 0.45', W-R-4, y45-4);
  }
  paretoPts = [];
  S5.forEach(m=>{
    if(m[kx]==null||m[ky]==null) return;
    const px = X(m[kx]), py = Y(m[ky]), col = (SERIES_META[m.series]||{}).col||'#ccc';
    const r = (m.series==='ref'||m.series==='leader') ? 6 : 4.8;
    g.fillStyle = col; g.beginPath(); g.arc(px,py,r,0,Math.PI*2); g.fill();
    if(m.series==='ref'||m.series==='leader'){
      g.strokeStyle='#fff'; g.lineWidth=1.5; g.stroke();
      g.fillStyle='#eaf0fa'; g.font='10px sans-serif'; g.textAlign='left';
      g.fillText(m.id, px+7, py+3);
    }
    paretoPts.push({m, px, py, r});
  });
}
document.getElementById('scX').onchange = drawPareto;
document.getElementById('scY').onchange = drawPareto;
document.getElementById('paretoCanvas').onmousemove = ev=>{
  const c = ev.currentTarget, rect = c.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  let hit = null, bestD = 14;
  paretoPts.forEach(p=>{ const d = Math.hypot(mx-p.px, my-p.py); if(d<bestD){ bestD=d; hit=p.m; }});
  document.getElementById('paretoHover').innerHTML = hit
    ? `<b>${hit.id}</b>（${hit.desc}）· Vina ${fmt(hit.Vina)} · Vinardo ${fmt(hit.Vinardo)} · CNNaff ${fmt(hit.CNNaff)} · hERG ${fmt(hit.hERG)} · DILI ${fmt(hit.DILI)} · AMES ${fmt(hit.AMES)} · QED ${fmt(hit.QED)} · SA ${fmt(hit.SA)}`
    : '滑鼠移至圓點查看詳情，點擊直接開啟 3D 結合姿勢。';
};
document.getElementById('paretoCanvas').onclick = ev=>{
  const c = ev.currentTarget, rect = c.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  let hit = null, bestD = 14;
  paretoPts.forEach(p=>{ const d = Math.hypot(mx-p.px, my-p.py); if(d<bestD){ bestD=d; hit=p.m; }});
  if(hit) jumpTo3D(hit.id);
};

/* ---------------- Tab 2: Full SAR Table ---------------- */
let sortKey = 'Vina', sortAsc = true;
function renderSarTable(){
  const q = (document.getElementById('sarSearch').value||'').trim().toLowerCase();
  const sf = document.getElementById('seriesFilter').value;
  const cleanAmes = document.getElementById('onlyCleanAmes').checked;
  const zeroAlert = document.getElementById('onlyZeroAlert').checked;
  let list = S5.filter(m=>{
    if(sf!=='ALL' && m.series!==sf) return false;
    if(cleanAmes && (m.AMES==null || m.AMES>=0.45)) return false;
    if(zeroAlert && (m.BRENK.includes('nitro') || m.BRENK.includes('Michael'))) return false;
    if(q && !(m.id.toLowerCase().includes(q) || m.desc.toLowerCase().includes(q) || m.smiles.toLowerCase().includes(q))) return false;
    return true;
  });
  list.sort((a,b)=>{
    const va = a[sortKey], vb = b[sortKey];
    if(va==null) return 1; if(vb==null) return -1;
    if(typeof va==='string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va-vb : vb-va;
  });
  const cols = [
    ['id','ID'],['series','系列'],['desc','取代基設計'],
    ['Vina','Vina ↓'],['Vinardo','Vinardo ↓'],['CNNaff','CNNaff ↑'],['CNNscore','CNNscr ↑'],
    ['hERG','hERG ↓'],['DILI','DILI ↓'],['AMES','AMES ↓'],['HIA','HIA ↑'],
    ['QED','QED ↑'],['SA','SA ↓'],['cLogP','cLogP'],['MW','MW'],['nF','F'],['contacts_n','CRD']
  ];
  const thead = '<thead><tr>'+cols.map(([k,lbl])=>{
    const arrow = sortKey===k ? (sortAsc?' ▲':' ▼') : '';
    const numCls = ['id','series','desc'].includes(k)?'':'n';
    return `<th class="sort ${numCls}" data-k="${k}">${lbl}${arrow}</th>`;
  }).join('')+'</tr></thead>';
  const tbody = '<tbody>'+list.map(m=>{
    const meta = SERIES_META[m.series]||{name:m.series,cls:'p-pol'};
    return `<tr class="clickable" data-id="${m.id}">
      <td><b>${m.id}</b></td>
      <td><span class="pill ${meta.cls}">${meta.name}</span></td>
      <td class="note" style="max-width:240px">${m.desc}</td>
      <td class="n">${fmt(m.Vina,2)}<span class="note">±${fmt(m.Vina_sd,2)}</span></td>
      <td class="n">${fmt(m.Vinardo,2)}<span class="note">±${fmt(m.Vinardo_sd,2)}</span></td>
      <td class="n">${fmt(m.CNNaff,2)}</td>
      <td class="n">${fmt(m.CNNscore,2)}</td>
      <td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG,3)}</td>
      <td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI,3)}</td>
      <td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES,3)}</td>
      <td class="n ${riskCls(m.HIA,false)}">${fmt(m.HIA,2)}</td>
      <td class="n">${fmt(m.QED,3)}</td>
      <td class="n">${fmt(m.SA,2)}</td>
      <td class="n">${fmt(m.cLogP,2)}</td>
      <td class="n">${fmt(m.MW,0)}</td>
      <td class="n">${m.nF}</td>
      <td class="n">${m.contacts.length}/7</td>
    </tr>`;
  }).join('')+'</tbody>';
  const tbl = document.getElementById('sarTable');
  tbl.innerHTML = thead + tbody;
  tbl.querySelectorAll('th.sort').forEach(th=>{
    th.onclick = ()=>{
      const k = th.dataset.k;
      if(sortKey===k) sortAsc=!sortAsc; else { sortKey=k; sortAsc = !['CNNaff','CNNscore','HIA','QED','contacts_n'].includes(k); }
      renderSarTable();
    };
  });
  tbl.querySelectorAll('tr.clickable').forEach(tr=>{
    tr.onclick = ()=> jumpTo3D(tr.dataset.id);
  });
}
S5.forEach(m=>{ m.contacts_n = m.contacts.length; });
['sarSearch','seriesFilter','onlyCleanAmes','onlyZeroAlert'].forEach(id=>{
  document.getElementById(id).oninput = renderSarTable;
  document.getElementById(id).onchange = renderSarTable;
});
document.getElementById('btnExportCsv').onclick = ()=>{
  const keys = ['id','series','desc','smiles','Vina','Vina_sd','Vinardo','Vinardo_sd','CNNaff','CNNscore','hERG','DILI','AMES','Pgp','HIA','Bioavail','QED','SA','TPSA','cLogP','MW','RotB','nF','BRENK'];
  const csv = [keys.join(','), ...S5.map(m=>keys.map(k=>`"${String(m[k]??'').replace(/"/g,'""')}"`).join(','))].join('\n');
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
  a.download = 'gal3_stage5_aniline_sar.csv';
  a.click();
};

/* ---------------- Tab 3: 3D Viewer ---------------- */
const ALL3D = [
  ...S5.map(m=>({...m, src:'Stage 5 smina Vinardo 最佳姿勢 (pH 7.4)'})),
  ...LR.map(m=>({
    id: 'DB:' + m.id, series: m.group==='ref'?'ref':'leader',
    desc: (m.caption||'Stage 1–4 DynamicBind 姿勢') + ` (DB=${fmt(m.aff,2)})`,
    smiles: m.smiles, MW:m.MW, cLogP:m.cLogP, TPSA:m.TPSA, RotB:m.RotB, nF:m.nF, QED:m.QED, SA:null,
    hERG:m.hERG, DILI:m.DILI, AMES:m.AMES, Pgp:m.Pgp, HIA:m.HIA, Bioavail:m.Bioavail, Sol:m.Sol,
    Vina:m.vina_old, Vinardo:m.vinardo_old, CNNaff:null, CNNscore:null,
    BRENK:'', PAINS:'', contacts:m.contacts, png:m.png, mol:m.mol,
    src:'Stage 1–4 DynamicBind 疊合姿勢 (5E89 Cα Kabsch)'
  }))
];
let cur3D = 0, viewer3d = null, protM = null, ligM = null, surf3d = null, labels3d = [], hb3d = [];
let showSurf=false, showLabel=true, showHb=true, showCartoon=true, spinOn=false;

function initViewer(){
  try{
    viewer3d = $3Dmol.createViewer(document.getElementById('viewer3d'), {backgroundColor:'#0b101b', antialias:true});
    protM = viewer3d.addModel(DATA.protein_pdb, 'pdb');
  } catch(e){
    document.getElementById('viewer3d').innerHTML = '<div style="padding:24px;color:#ffd0a0">WebGL 無法初始化：'+e+'</div>';
  }
  renderVList();
  load3D(cur3D);
}
function renderVList(){
  const q = (document.getElementById('vSearch').value||'').trim().toLowerCase();
  const host = document.getElementById('vList');
  host.innerHTML = ALL3D.map((m,i)=>({m,i})).filter(({m})=>!q || m.id.toLowerCase().includes(q) || (m.desc||'').toLowerCase().includes(q)).map(({m,i})=>{
    const col = (SERIES_META[m.series]||{}).col||'#ccc';
    const score = m.Vinardo!=null ? fmt(m.Vinardo,2) : '—';
    return `<div class="v-item ${i===cur3D?'sel':''}" data-i="${i}">
      <span class="dot" style="background:${col}"></span>
      <span class="mono" style="font-size:11.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${m.id}</span>
      <span class="mono note" style="margin-left:auto">${score}</span>
    </div>`;
  }).join('');
  host.querySelectorAll('.v-item').forEach(el=>el.onclick=()=>load3D(Number(el.dataset.i)));
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
  const set = new Set(contacts);
  KEYRES.forEach(r=>{
    const ats = protM.selectedAtoms({resi:r, atom:'CA'});
    if(!ats.length) return;
    const a = ats[0], hit = set.has(r);
    labels3d.push(viewer3d.addLabel(CRD[String(r)], {
      position:{x:a.x,y:a.y,z:a.z}, fontSize:11,
      fontColor:hit?'#ffe49e':'#b8c8e0',
      backgroundColor:hit?'rgba(95,66,18,.88)':'rgba(16,24,38,.75)',
      backgroundOpacity:0.9, borderThickness:hit?1:0, borderColor:'#f0c766', inFront:true
    }));
  });
}
function drawHb3D(){
  if(!viewer3d) return [];
  hb3d.forEach(s=>viewer3d.removeShape(s)); hb3d=[];
  if(!showHb || !ligM) return [];
  const lig = ligM.selectedAtoms({}).filter(a=>a.elem==='N'||a.elem==='O');
  const pro = protM.selectedAtoms({}).filter(a=>a.elem==='N'||a.elem==='O');
  const found = [];
  lig.forEach(l=>pro.forEach(p=>{
    const d = Math.hypot(l.x-p.x, l.y-p.y, l.z-p.z);
    if(d<=3.4){
      found.push({d, res:(p.resn||'')+p.resi});
      hb3d.push(viewer3d.addCylinder({start:{x:l.x,y:l.y,z:l.z}, end:{x:p.x,y:p.y,z:p.z},
        radius:0.045, dashed:true, dashLength:0.22, gapLength:0.18, fromCap:1, toCap:1, color:'#3ad1be'}));
    }
  }));
  return found;
}
function load3D(idx){
  cur3D = idx;
  const m = ALL3D[idx];
  let hb = [];
  if(viewer3d){
    if(ligM){ viewer3d.removeModel(ligM); ligM=null; }
    if(m.mol){
      ligM = viewer3d.addModel(m.mol, 'sdf');
      ligM.setStyle({}, {stick:{radius:0.17,colorscheme:'cyanCarbon'}, sphere:{radius:0.25,colorscheme:'cyanCarbon'}});
    }
    styleProt3D(m.contacts||[]);
    drawLabels3D(m.contacts||[]);
    hb = drawHb3D();
    if(ligM){ viewer3d.zoomTo({model:ligM}); viewer3d.zoom(0.55); }
    viewer3d.render();
  }
  document.getElementById('vPoseBadge').textContent = m.src;
  const set = new Set(m.contacts||[]);
  const tags = KEYRES.map(r=>`<span class="tag ${set.has(r)?'hit':''}">${CRD[String(r)]}</span>`).join('');
  const hbTxt = hb.length ? hb.sort((a,b)=>a.d-b.d).slice(0,10).map(x=>`${x.res} (${x.d.toFixed(2)}Å)`).join('、') : '3.4 Å 內無 N/O–N/O 幾何氫鍵';
  const meta = SERIES_META[m.series]||{name:m.series,cls:'p-pol'};
  document.getElementById('vRight').innerHTML = `
    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
      <h2 class="mono" style="margin:0;font-size:16px;color:#fff">${m.id}</h2>
      <span class="pill ${meta.cls}">${meta.name}</span>
    </div>
    <div class="note" style="margin:3px 0 6px">${m.desc||''}</div>
    ${m.png?`<img class="png2d" src="data:image/png;base64,${m.png}" alt="2D">`:''}
    <div class="smi-box">${m.smiles}</div>
    <div style="margin:6px 0 10px"><button id="bCopy3D">複製 SMILES</button> <span id="cpMsg" class="note" style="color:var(--teal)"></span></div>
    <div class="card" style="padding:10px;margin-bottom:10px">
      <h3>多引擎對接分數</h3>
      <table>
        <tr><td class="note">smina Vina (pH 7.4)</td><td class="n">${fmt(m.Vina,3)}</td></tr>
        <tr><td class="note">smina Vinardo (pH 7.4)</td><td class="n">${fmt(m.Vinardo,3)}</td></tr>
        <tr><td class="note">GNINA GPU CNNaffinity</td><td class="n">${fmt(m.CNNaff,3)}</td></tr>
        <tr><td class="note">GNINA GPU CNNscore</td><td class="n">${fmt(m.CNNscore,3)}</td></tr>
      </table>
    </div>
    <div class="card" style="padding:10px;margin-bottom:10px">
      <h3>CRD 關鍵殘基接觸 (${(m.contacts||[]).length}/7)</h3>
      <div>${tags}</div>
      <div class="note" style="margin-top:6px"><b>即時氫鍵：</b>${hbTxt}</div>
    </div>
    <div class="card" style="padding:10px;margin-bottom:10px">
      <h3>ADMET-AI 與成藥性</h3>
      <table>
        <tr><td class="note">hERG ↓</td><td class="n ${riskCls(m.hERG,true)}">${fmt(m.hERG,3)}</td><td class="note">DILI ↓</td><td class="n ${riskCls(m.DILI,true)}">${fmt(m.DILI,3)}</td></tr>
        <tr><td class="note">AMES ↓</td><td class="n ${riskCls(m.AMES,true)}">${fmt(m.AMES,3)}</td><td class="note">Pgp ↓</td><td class="n ${riskCls(m.Pgp,true)}">${fmt(m.Pgp,3)}</td></tr>
        <tr><td class="note">HIA ↑</td><td class="n ${riskCls(m.HIA,false)}">${fmt(m.HIA,3)}</td><td class="note">Bioavail ↑</td><td class="n ${riskCls(m.Bioavail,false)}">${fmt(m.Bioavail,3)}</td></tr>
        <tr><td class="note">QED ↑</td><td class="n">${fmt(m.QED,3)}</td><td class="note">SA Score ↓</td><td class="n">${fmt(m.SA,2)}</td></tr>
        <tr><td class="note">TPSA</td><td class="n">${fmt(m.TPSA,1)}</td><td class="note">cLogP</td><td class="n">${fmt(m.cLogP,2)}</td></tr>
        <tr><td class="note">MW</td><td class="n">${fmt(m.MW,1)}</td><td class="note">RotB / F</td><td class="n">${m.RotB??'—'} / ${m.nF??'—'}</td></tr>
      </table>
      <div class="note" style="margin-top:6px"><b>Brenk 警示：</b>${m.BRENK||'無 ✅'}</div>
    </div>`;
  document.getElementById('bCopy3D').onclick = ()=>{
    navigator.clipboard?.writeText(m.smiles);
    document.getElementById('cpMsg').textContent='已複製！';
    setTimeout(()=>document.getElementById('cpMsg').textContent='',1500);
  };
  renderVList();
}
function jumpTo3D(id){
  const idx = ALL3D.findIndex(x=>x.id===id);
  if(idx>=0) cur3D = idx;
  document.querySelector('[data-tab="t3"]').click();
  if(viewerInit) load3D(cur3D);
}
document.getElementById('bReset').onclick = ()=>{ if(viewer3d&&ligM){ viewer3d.zoomTo({model:ligM}); viewer3d.zoom(0.55); viewer3d.render(); }};
document.getElementById('bSurf').onclick = function(){
  showSurf=!showSurf; this.classList.toggle('on',showSurf);
  if(!viewer3d) return;
  if(surf3d){ viewer3d.removeSurface(surf3d); surf3d=null; }
  if(showSurf){ surf3d = viewer3d.addSurface($3Dmol.SurfaceType.MS, {opacity:0.58,color:'#3b5985'}, {model:protM}); if(surf3d?.then) surf3d.then(s=>{surf3d=s;viewer3d.render();}); }
  viewer3d.render();
};
document.getElementById('bLabel').onclick = function(){ showLabel=!showLabel; this.classList.toggle('on',showLabel); drawLabels3D(ALL3D[cur3D].contacts||[]); viewer3d?.render(); };
document.getElementById('bHb').onclick = function(){ showHb=!showHb; this.classList.toggle('on',showHb); drawHb3D(); viewer3d?.render(); };
document.getElementById('bCartoon').onclick = function(){ showCartoon=!showCartoon; this.classList.toggle('on',showCartoon); styleProt3D(ALL3D[cur3D].contacts||[]); viewer3d?.render(); };
document.getElementById('bSpin').onclick = function(){ spinOn=!spinOn; this.classList.toggle('on',spinOn); viewer3d?.spin(spinOn?'y':false); };

/* ---------------- Boot ---------------- */
drawEvolution();
renderSarTable();
window.addEventListener('resize', ()=>{
  if(document.getElementById('t1').classList.contains('active')) drawEvolution();
  if(document.getElementById('t2').classList.contains('active')) drawPareto();
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
