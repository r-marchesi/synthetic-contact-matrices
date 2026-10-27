import os
import sys
import glob
import base64
import argparse
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RES_DIR = os.path.join(EXP_DIR, "results")
DEEP_DIR = os.path.join(RES_DIR, "deep_analysis")
RQ2_DIR = os.path.join(RES_DIR, "rq2_subgroups")
RQ1_DIR = os.path.join(RES_DIR, "rq1_scaling")

sys.path.append(SCRIPT_DIR)
from importlib import import_module
deep07 = import_module("05_deep_analysis_and_exports")


def encode_image_b64(img_path):
    if not img_path or not os.path.exists(img_path):
        return None
    with open(img_path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


def safe_read_csv(path):
    return pd.read_csv(path) if path and os.path.exists(path) else pd.DataFrame()


def find_first_matching(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def df_to_styled_html(df, float_fmt="{:.4f}"):
    if df is None or df.empty:
        return "<p><em>Table not yet available.</em></p>"
    df_disp = df.copy()
    for col in df_disp.select_dtypes(include=[np.floating]).columns:
        df_disp[col] = df_disp[col].apply(lambda x: "—" if pd.isna(x) else float_fmt.format(x))
    return df_disp.to_html(index=False, classes="nb-table", border=0, escape=True)


def df_to_markdown_safe(df, float_fmt="{:.4f}"):
    if df is None or df.empty:
        return "_Table not yet available._"
    df_disp = df.copy()
    for col in df_disp.select_dtypes(include=[np.floating]).columns:
        df_disp[col] = df_disp[col].apply(lambda x: "—" if pd.isna(x) else float_fmt.format(x))
    try:
        return df_disp.to_markdown(index=False)
    except Exception:
        return "```\n" + df_disp.to_string(index=False) + "\n```"


def build_notebook_and_report(args):
    lead_csv = os.path.join(RES_DIR, "leaderboard_07_semantic.csv")
    if not args.skip_refresh:
        if not os.path.exists(lead_csv):
            subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "04_evaluate_and_wsmad.py")], check=False)
        subprocess.run(
            [
                sys.executable,
                os.path.join(SCRIPT_DIR, "05_deep_analysis_and_exports.py"),
                "--pg_epoch", str(args.pg_epoch),
                "--ff_epoch", str(args.ff_epoch),
            ],
            check=False,
        )

    pg_ep, ff_ep = deep07.select_best_epochs_by_wsmad(lead_csv, args.pg_epoch, args.ff_epoch)

    df_lead = safe_read_csv(lead_csv)
    df_wsmad_dims = safe_read_csv(os.path.join(RES_DIR, "wsmad_dimension_breakdown.csv"))
    df_vol = safe_read_csv(os.path.join(DEEP_DIR, "table1_volume_statistics.csv"))
    df_loc_vol = safe_read_csv(os.path.join(DEEP_DIR, "table1b_location_volumes.csv"))
    df_loc_mat = safe_read_csv(find_first_matching(os.path.join(DEEP_DIR, "table2_location_matrices_*.csv")))
    df_sub_mat = safe_read_csv(find_first_matching(os.path.join(DEEP_DIR, "table3_subgroup_matrices_*.csv")))
    df_corr = safe_read_csv(find_first_matching(os.path.join(DEEP_DIR, "table4_pairwise_correlations_*.csv")))
    df_marg = safe_read_csv(find_first_matching(os.path.join(DEEP_DIR, "table4b_marginal_distributions_*.csv")))
    df_c2st = safe_read_csv(os.path.join(DEEP_DIR, "table5_c2st_ablations.csv"))
    df_imp = safe_read_csv(os.path.join(DEEP_DIR, "table5b_c2st_feature_importances.csv"))
    df_rq2 = safe_read_csv(os.path.join(RQ2_DIR, "rq2_subgroup_leaderboard.csv"))
    df_ovf_sum = safe_read_csv(os.path.join(DEEP_DIR, "table7_overfitting_summary.csv"))
    df_ovf_deg = safe_read_csv(os.path.join(DEEP_DIR, "table7b_diary_copy_by_degree.csv"))

    target_flagship = f"100_percent_plan_guided_epoch_{pg_ep}"
    if not df_lead.empty:
        match_rows = df_lead[df_lead["Run"] == target_flagship]
        best_row = match_rows.iloc[0] if not match_rows.empty else df_lead.iloc[0]
        best_run_name = str(best_row["Run"])
        best_wsmad = float(best_row.get("WSMAD_Norm_[0,1]", np.nan))
        best_wsmad_mae = float(best_row.get("WSMAD_Cell_MAE", np.nan))
        best_mae = float(best_row["MAE_Total"])
        best_mae_home = float(best_row.get("MAE_Home", 0.0))
        best_mae_work = float(best_row.get("MAE_Work", 0.0))
        best_mae_school = float(best_row.get("MAE_School", 0.0))
        best_r0 = float(best_row["R0_Ratio"])
        best_tvd = float(best_row["Volume_TVD"])
        best_wass = float(best_row.get("Volume_Wass", 0.0))
        best_corr = float(best_row["Corr_MAE"])
        best_marg_c2st = float(best_row.get("Marginal_C2ST_AUC", np.nan))
        best_joint_c2st = float(best_row["Joint_C2ST_AUC"])
        best_syntax = float(best_row["Syntax_Valid_%"])
    else:
        best_run_name = target_flagship
        best_wsmad = best_wsmad_mae = best_mae = best_mae_home = best_mae_work = best_mae_school = np.nan
        best_r0 = best_tvd = best_wass = best_corr = best_marg_c2st = best_joint_c2st = best_syntax = np.nan

    fig_wsmad_b64 = encode_image_b64(os.path.join(RES_DIR, "wsmad_epoch_stopping_curve.png"))
    fig1_b64 = encode_image_b64(os.path.join(DEEP_DIR, "fig1_volume_analysis.png"))
    fig2_b64 = encode_image_b64(find_first_matching(os.path.join(DEEP_DIR, "fig2_matrices_by_location_*.png")))
    fig3_b64 = encode_image_b64(find_first_matching(os.path.join(DEEP_DIR, "fig3_matrices_by_subgroup_*.png")))
    fig4_b64 = encode_image_b64(find_first_matching(os.path.join(DEEP_DIR, "fig4_marginal_distributions_*.png")))
    fig_corr_b64 = encode_image_b64(find_first_matching(os.path.join(DEEP_DIR, "fig4a_correlation_cramers_v_*.png")))
    fig5_b64 = encode_image_b64(os.path.join(DEEP_DIR, "fig5_c2st_diagnostics.png"))
    fig_rq2_b64 = encode_image_b64(os.path.join(RQ2_DIR, "rq2_conditioning_and_subgroups.png"))
    fig_ovf_b64 = encode_image_b64(os.path.join(DEEP_DIR, "fig7_overfitting_dcr.png"))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def nb_plot(b64_str, fig_num, caption):
        if not b64_str:
            return f"<div class='nb-missing'>[Figure {fig_num} not available: {caption}]</div>"
        return f"""
        <div class="nb-output-plot">
            <div class="nb-prompt out-prompt">Out [{fig_num}]:</div>
            <div class="nb-plot-body">
                <img src="{b64_str}" alt="{caption}" />
                <div class="nb-caption"><strong>Figure {fig_num}:</strong> {caption}</div>
            </div>
        </div>
        """

    notebook_path = os.path.join(RES_DIR, "notebook_exp07.html")
    html_nb = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Research Notebook: Semantic Causal Hierarchy (Exp 07)</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
    :root {{
        --bg: #f8fafc;
        --paper: #ffffff;
        --sidebar: #0f172a;
        --primary: #0f172a;
        --accent: #2563eb;
        --text: #1e293b;
        --muted: #64748b;
        --border: #e2e8f0;
        --code-bg: #0f172a;
        --in-color: #2563eb;
        --out-color: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.7;
        display: flex;
    }}
    .sidebar {{
        width: 295px;
        height: 100vh;
        position: sticky;
        top: 0;
        background: var(--sidebar);
        color: #f8fafc;
        padding: 1.8rem 1.4rem;
        overflow-y: auto;
        flex-shrink: 0;
    }}
    .sidebar h3 {{ font-size: 0.92rem; text-transform: uppercase; letter-spacing: 0.06em; color: #93c5fd; margin: 0 0 1rem 0; }}
    .sidebar nav a {{
        display: block;
        color: #cbd5e1;
        text-decoration: none;
        font-size: 0.85rem;
        padding: 0.42rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.2rem;
    }}
    .sidebar nav a:hover {{ background: rgba(255,255,255,0.1); color: #ffffff; }}
    .sidebar .meta {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; font-size: 0.78rem; color: #94a3b8; }}
    .notebook-main {{
        flex: 1;
        max-width: 1260px;
        margin: 2rem auto;
        background: var(--paper);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        padding: 3rem 3.5rem;
    }}
    .nb-header {{ border-bottom: 3px solid var(--accent); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
    .nb-header h1 {{ margin: 0 0 0.5rem 0; font-size: 2.05rem; color: var(--primary); }}
    .nb-header .subtitle {{ color: var(--muted); font-size: 1.02rem; margin: 0; }}
    .kpi-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.85rem;
        margin: 1.5rem 0 2.5rem 0;
    }}
    .kpi-pill {{
        background: #f8fafc;
        border: 1px solid var(--border);
        border-left: 4px solid var(--accent);
        padding: 0.8rem 0.95rem;
        border-radius: 8px;
    }}
    .kpi-pill .k-lbl {{ font-size: 0.7rem; text-transform: uppercase; color: var(--muted); font-weight: 700; }}
    .kpi-pill .k-val {{ font-size: 1.4rem; font-weight: 800; color: var(--primary); margin: 0.15rem 0; }}
    .kpi-pill .k-sub {{ font-size: 0.75rem; color: #16a34a; font-weight: 600; }}
    .nb-section {{ margin-bottom: 3.2rem; scroll-margin-top: 1.5rem; }}
    h2 {{ color: var(--primary); font-size: 1.48rem; border-bottom: 1px solid var(--border); padding-bottom: 0.45rem; margin-top: 2.2rem; }}
    h3 {{ color: #1e40af; font-size: 1.15rem; margin-top: 1.4rem; }}
    .nb-cell {{ display: flex; gap: 0.8rem; margin: 1.1rem 0; align-items: flex-start; }}
    .nb-prompt {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.78rem; font-weight: 700; width: 68px; flex-shrink: 0; padding-top: 0.6rem; text-align: right; }}
    .in-prompt {{ color: var(--in-color); }}
    .out-prompt {{ color: var(--out-color); }}
    .nb-code {{
        flex: 1;
        background: var(--code-bg);
        color: #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 0.83rem;
        overflow-x: auto;
        white-space: pre;
        line-height: 1.5;
    }}
    .nb-output-table {{ flex: 1; overflow-x: auto; background: #ffffff; border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem; }}
    .nb-output-plot {{ display: flex; gap: 0.8rem; margin: 1.4rem 0; align-items: flex-start; }}
    .nb-plot-body {{ flex: 1; background: #ffffff; border: 1px solid var(--border); border-radius: 8px; padding: 1rem; text-align: center; }}
    .nb-plot-body img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    .nb-caption {{ margin-top: 0.6rem; font-size: 0.86rem; color: var(--muted); }}
    .callout {{ background: #eff6ff; border-left: 4px solid var(--accent); padding: 1rem 1.25rem; border-radius: 6px; margin: 1rem 0; }}
    table.nb-table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    table.nb-table th {{ background: #f1f5f9; color: var(--primary); padding: 0.52rem 0.7rem; text-align: left; border-bottom: 2px solid var(--border); white-space: nowrap; }}
    table.nb-table td {{ padding: 0.45rem 0.7rem; border-bottom: 1px solid var(--border); white-space: nowrap; }}
</style>
</head>
<body>

<aside class="sidebar">
    <h3>Notebook Contents</h3>
    <nav>
        <a href="#sec-overview">1. Study Overview</a>
        <a href="#sec-schema">2. Semantic Dictionary &amp; Attendance</a>
        <a href="#sec-hierarchy">3. Enriched Plan &amp; Causal DSL</a>
        <a href="#sec-wsmad">4. WSMAD Epoch-Stopping Criterion</a>
        <a href="#sec-volumes">5. Volume &amp; Cohabitant Split</a>
        <a href="#sec-matrices">6. 15x15 Contact Matrices</a>
        <a href="#sec-rq2">7. RQ2: Subgroup Performance</a>
        <a href="#sec-corr">8. Multivariate Correlations</a>
        <a href="#sec-c2st">9. C2ST Realism &amp; Drivers</a>
        <a href="#sec-overfit">10. Overfitting &amp; Novelty Audit</a>
    </nav>
    <div class="meta">
        <strong>Engine:</strong> Gemma-2-9B-IT (16-Ep LoRA)<br>
        <strong>WSMAD Flagship:</strong> {best_run_name}<br>
        <strong>Generated:</strong> {timestamp}
    </div>
</aside>

<main class="notebook-main">

    <div class="nb-header">
        <h1>LLM-Generated Synthetic Contact Matrices: The Semantic Causal Hierarchy (Exp 07)</h1>
        <p class="subtitle">Self-Contained Research Notebook — Official MixIT Semantic Encoding, Cohabitant/Non-Cohabitant Macro-Planning, WSMAD Epoch Stopping, and Subgroup Evaluation</p>
    </div>

    <div class="kpi-strip">
        <div class="kpi-pill">
            <div class="k-lbl">WSMAD Norm [0,1]</div>
            <div class="k-val">{best_wsmad:.4f}</div>
            <div class="k-sub">Cell MAE: {best_wsmad_mae:.4f}</div>
        </div>
        <div class="kpi-pill">
            <div class="k-lbl">15x15 Matrix MAE</div>
            <div class="k-val">{best_mae:.4f}</div>
            <div class="k-sub">Home: {best_mae_home:.4f} | Work: {best_mae_work:.4f}</div>
        </div>
        <div class="kpi-pill">
            <div class="k-lbl">R0 Spectral Ratio</div>
            <div class="k-val">{best_r0:.4f}</div>
            <div class="k-sub">Syntax Valid: {best_syntax:.1f}%</div>
        </div>
        <div class="kpi-pill">
            <div class="k-lbl">Volume Overlap</div>
            <div class="k-val">{(1.0 - best_tvd)*100:.1f}%</div>
            <div class="k-sub">TVD: {best_tvd:.4f} | Wass: {best_wass:.2f}</div>
        </div>
        <div class="kpi-pill">
            <div class="k-lbl">Correlation MAE</div>
            <div class="k-val">{best_corr:.4f}</div>
            <div class="k-sub">Cramér's V (153 pairs)</div>
        </div>
        <div class="kpi-pill">
            <div class="k-lbl">Joint C2ST AUC</div>
            <div class="k-val">{best_joint_c2st:.4f}</div>
            <div class="k-sub">5-Col Marginal: {best_marg_c2st:.4f}</div>
        </div>
    </div>

    <!-- SECTION 1 -->
    <section id="sec-overview" class="nb-section">
        <h2>1. Study Overview</h2>
        <p>
            This notebook documents the <strong>Semantic Causal Hierarchy (<code>07_semantic_causal</code>)</strong> for synthesizing individual-level epidemiological contact diaries and 15x15 age-structured contact matrices using parameter-efficient fine-tuning of <code>google/gemma-2-9b-it</code>. Every variable name and categorical value is mapped 1-to-1 to the official <code>MixIT_it_dictionary.xlsx</code> specification, structural attendance <code>NA</code>s are explicitly disambiguated, the macro-plan budgets both household cohabitants and external contacts, and checkpoint selection over a 16-epoch training horizon is governed by the <strong>Weighted Subgroup Mixing Absolute Difference (WSMAD)</strong> criterion.
        </p>
    </section>

    <!-- SECTION 2 -->
    <section id="sec-schema" class="nb-section">
        <h2>2. Official MixIT Semantic Encoding &amp; Explicit Attendance Disambiguation</h2>
        <p>
            All variable names use natural English keys without abbreviations or underscores, and all coded values are mapped bijectively to their official <code>MixIT_it_dictionary.xlsx</code> definitions:
        </p>
        <ul>
            <li><strong>Explicit Work &amp; School Attendance Disambiguation:</strong> In the MixIT survey, <code>presence_work = NA</code> is used structurally for respondents whose occupation is not <code>Employed</code> (2,589 of 2,632 rows), and <code>presence_school = NA</code> is used for respondents whose occupation is not <code>Student</code> (4,346 of 4,483 rows). We explicitly encode structural non-applicability as <code>Not applicable</code>, reserving <code>Unknown</code> strictly for employed/student respondents with unreported attendance.</li>
            <li><strong>Official Household Monthly Net Income Brackets:</strong> Survey codes <code>1..9</code> are encoded as their exact Euro ranges: <code>Less than 1000€</code>, <code>1000€-1499€</code>, <code>1500€-1999€</code>, <code>2000€-2499€</code>, <code>2500€-2999€</code>, <code>3000€-3499€</code>, <code>3500€-3999€</code>, <code>4000€-4999€</code>, and <code>More or equal than 5000€</code>.</li>
            <li><strong>Fortnightly Contact Frequency (<code>frequency_multi</code>):</strong> Survey integers <code>1..14</code> represent the exact number of contact days in the preceding 14 days and are encoded concisely as <code>14/14 days</code> (daily cohabitants), <code>10/14 days</code> (weekday coworkers/schoolmates), down to <code>1/14 days</code>.</li>
        </ul>
    </section>

    <!-- SECTION 3 -->
    <section id="sec-hierarchy" class="nb-section">
        <h2>3. Enriched Cohabitant / Non-Cohabitant Macro-Plan &amp; Causal DSL</h2>
        <p>
            Given a respondent profile \\(\\mathbf{{d}}\\), diary synthesis \\(\\mathcal{{Y}} = (\\mathbf{{m}}, \\mathbf{{C}})\\) is factorized into a <strong>Macro Plan</strong> \\(\\mathbf{{m}}\\) that explicitly partitions total volume into <code>Cohabitants</code> (bounded by <code>Household Size - 1</code>) and <code>Non-cohabitants</code> (external contacts across work, school, and leisure) alongside the 6 MixIT location counts, followed by <strong>Micro Contact Synthesis</strong> \\(\\mathbf{{C}}\\):
        </p>
        <div class="nb-cell">
            <div class="nb-prompt in-prompt">In [1]:</div>
            <div class="nb-code">[PARTICIPANT]
Age: 42 | Gender: Female | Household Size: 3 | Occupation: Employed | Work Attendance: Remote work | School Attendance: Not applicable | Education: Bachelor or above | Household Income: 2000€-2499€ | Region: North-West

[PLAN]
Total: 3 | Cohabitants: 2 | Non-cohabitants: 1 | Home: 2 | Work: 0 | School: 0 | Leisure: 1 | Transport: 0 | Other: 0

[CONTACTS]
- Location: Home | Relationship: Cohabitant | Setting: indoor | Age Difference: Peer | Age: 44 | Gender: Male | Frequency: 14/14 days | Physical Contact: Yes | Distance: Physical contact
- Location: Home | Relationship: Cohabitant | Setting: indoor | Age Difference: Younger | Age: 10 | Gender: Female | Frequency: 14/14 days | Physical Contact: Yes | Distance: Physical contact
- Location: Leisure | Relationship: Friend or acquaintance | Setting: outdoor | Age Difference: Peer | Age: 39 | Gender: Female | Frequency: 2/14 days | Physical Contact: No | Distance: Between 1 and 2 meters</div>
        </div>
        <p>
            Placing <code>Relationship</code> immediately after <code>Location</code> ensures that during Phase 2 of <code>Plan-Guided</code> decoding (which prefix-forces <code>\\n- Location: &lt;Loc&gt; | Relationship:</code>), the very first token predicted on each line resolves whether the contact is a household <code>Cohabitant</code> or an external role.
        </p>
    </section>

    <!-- SECTION 4 -->
    <section id="sec-wsmad" class="nb-section">
        <h2>4. WSMAD Epoch-Stopping Criterion &amp; 16-Epoch Trajectory</h2>
        <p>
            To select the optimal training checkpoint across our 16-epoch cosine schedule, we define the <strong>Weighted Subgroup Mixing Absolute Difference (WSMAD)</strong>. For each conditioning dimension \\(d \\in \\{{1, \\dots, D\\}}\\) and subgroup \\(s \\in \\mathcal{{S}}_d\\), let \\(M^{{(s,\\text{{GT}})}}_{{a,r}}\\) and \\(M^{{(s,\\text{{Gen}})}}_{{a,r}}\\) denote the \\(15 \\times 15\\) mixing matrices where columns \\(r \\in \\{{1,\\dots,15\\}}\\) index respondent age groups (with respondent count \\(N_{{s,r}}\\)) and rows \\(a \\in \\{{1,\\dots,15\\}}\\) index contact age groups. The normalized column discrepancy \\(\\delta^{{(s)}}_r \\in [0, 1]\\) is weighted by the respondent age distribution \\(N_{{s,r}} / N_s\\), aggregated across subgroups, and averaged over all \\(D = 9\\) dimensions:
        </p>
        <p>
            \\[ \\text{{WSMAD}}_{{[0,1]}} = \\frac{{1}}{{D}} \\sum_{{d=1}}^{{D}} \\sum_{{s \\in \\mathcal{{S}}_d}} \\frac{{N_s}}{{N}} \\sum_{{r=1}}^{{15}} \\frac{{N_{{s,r}}}}{{N_s}} \\left( \\frac{{\\sum_{{a=1}}^{{15}} |M^{{(s,\\text{{GT}})}}_{{a,r}} - M^{{(s,\\text{{Gen}})}}_{{a,r}}|}}{{\\sum_{{a=1}}^{{15}} (M^{{(s,\\text{{GT}})}}_{{a,r}} + M^{{(s,\\text{{Gen}})}}_{{a,r}})}} \\right) \\in [0, 1] \\]
        </p>
        {nb_plot(fig_wsmad_b64, 1, "16-Epoch Stopping Trajectory: WSMAD Normalized [0,1], WSMAD Cell MAE, Volume TVD, and Joint C2ST AUC across Training Epochs.")}
        <h3>4.1 Full 16-Epoch Leaderboard (Sorted by WSMAD)</h3>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [1]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_lead)}</div>
        </div>
        <h3>4.2 WSMAD Breakdown by Conditioning Dimension</h3>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [2]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_wsmad_dims)}</div>
        </div>
    </section>

    <!-- SECTION 5 -->
    <section id="sec-volumes" class="nb-section">
        <h2>5. Contact Volume, Right Tail &amp; Cohabitant / Non-Cohabitant Split</h2>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [3]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_vol)}</div>
        </div>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [4]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_loc_vol)}</div>
        </div>
        {nb_plot(fig1_b64, 2, "Discrete Volume PMF, Empirical CDF, Log-Survival Right Tail P(K >= k), and Mean Contacts by Role (Cohabitant vs. Non-Cohabitant) and Location.")}
    </section>

    <!-- SECTION 6 -->
    <section id="sec-matrices" class="nb-section">
        <h2>6. 15x15 Age-Mixing Contact Matrices by Social Location</h2>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [5]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_loc_mat)}</div>
        </div>
        {nb_plot(fig2_b64, 3, "15x15 Age-Structured Contact Matrices (Ground Truth vs. Generated vs. Difference) across Total and all 6 MixIT Social Locations.")}
    </section>

    <!-- SECTION 7 -->
    <section id="sec-rq2" class="nb-section">
        <h2>7. RQ2: Subgroup Performance &amp; Explicit Attendance Analysis</h2>
        {nb_plot(fig_rq2_b64, 4, "RQ2 Conditioning Fidelity: Home Contacts by Household Size, Age Assortativity, Total Contacts by Explicit Work Attendance, and Volume by Age Bracket.")}
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [6]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_sub_mat)}</div>
        </div>
        {nb_plot(fig3_b64, 5, "Stratified 15x15 Contact Matrices across Gender, Explicit Work Attendance, Household Size, and Geographic Macro-Regions.")}
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [7]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_rq2)}</div>
        </div>
    </section>

    <!-- SECTION 8 -->
    <section id="sec-corr" class="nb-section">
        <h2>8. Cross-Variable Correlations &amp; Marginal Attribute Distributions</h2>
        {nb_plot(fig_corr_b64, 6, "Pairwise Cramér's V Association Matrix (Ground Truth vs. Generated vs. Absolute Difference).")}
        {nb_plot(fig4_b64, 7, "Marginal Categorical Distributions and Total Variation Distance (TVD) Across All Contact Attributes.")}
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [8]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_corr.head(20) if not df_corr.empty else df_corr)}</div>
        </div>
    </section>

    <!-- SECTION 9 -->
    <section id="sec-c2st" class="nb-section">
        <h2>9. Classifier Two-Sample Test (C2ST) &amp; Feature Importance Diagnostics</h2>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [9]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_c2st)}</div>
        </div>
        {nb_plot(fig5_b64, 8, "Full 19-Feature Joint C2ST Importance Breakdown vs. Uniform Noise Baseline (5.26%) and Out-of-Fold Predicted Probability Density.")}
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [10]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_imp)}</div>
        </div>
    </section>

    <!-- SECTION 10 -->
    <section id="sec-overfit" class="nb-section">
        <h2>10. Overfitting, Memorization &amp; Novelty Audit</h2>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [11]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_ovf_sum, float_fmt="{:.2f}")}</div>
        </div>
        <div class="nb-cell">
            <div class="nb-prompt out-prompt">Out [12]:</div>
            <div class="nb-output-table">{df_to_styled_html(df_ovf_deg, float_fmt="{:.2f}")}</div>
        </div>
        {nb_plot(fig_ovf_b64, 9, "Overfitting & Memorization Audit: Single-Contact Novelty, Full Diary Match Rate by Degree K, and Distance to Closest Training Record (DCR).")}
    </section>

</main>
</body>
</html>
"""
    with open(notebook_path, "w") as f:
        f.write(html_nb)
    print(f"✅ Saved Exp 07 HTML Computational Notebook -> {notebook_path}")

    md_path = os.path.join(RES_DIR, "DEEP_ANALYSIS_REPORT.md")
    md_report = f"""# Experiment 07: Semantic Causal Hierarchy — Research Report
*Generated: {timestamp} | WSMAD-Optimal Flagship Model: `{best_run_name}`*

## 1. Headline Metrics (`{best_run_name}`)
* **WSMAD Normalized $[0, 1]$:** `{best_wsmad:.4f}` | **WSMAD Cell MAE:** `{best_wsmad_mae:.4f}`
* **Syntax Validity:** `{best_syntax:.2f}%`
* **15x15 Matrix MAE:** Total = `{best_mae:.4f}` | Home = `{best_mae_home:.4f}` | Work = `{best_mae_work:.4f}` | School = `{best_mae_school:.4f}`
* **$R_0$ Spectral Radius Ratio:** `{best_r0:.4f}`
* **Volume TVD:** `{best_tvd:.4f}` | **Volume Wasserstein:** `{best_wass:.4f}`
* **Pairwise Correlation MAE:** `{best_corr:.4f}`
* **5-Col Marginal C2ST AUC:** `{best_marg_c2st:.4f}` | **19-Col Full Joint C2ST AUC:** `{best_joint_c2st:.4f}`

## 2. Full 16-Epoch Leaderboard (Sorted by WSMAD)
{df_to_markdown_safe(df_lead.drop(columns=['Top_5_C2ST_Drivers'], errors='ignore'))}

## 3. WSMAD Dimension Breakdown
{df_to_markdown_safe(df_wsmad_dims)}

## 4. Volume & Cohabitant / Non-Cohabitant Breakdown
{df_to_markdown_safe(df_vol)}

{df_to_markdown_safe(df_loc_vol)}

## 5. 15x15 Contact Matrices by Location
{df_to_markdown_safe(df_loc_mat)}

## 6. RQ2 Subgroup Performance
{df_to_markdown_safe(df_sub_mat)}

{df_to_markdown_safe(df_rq2)}

## 7. C2ST Ablations & Feature Importance
{df_to_markdown_safe(df_c2st)}

## 8. Overfitting & Memorization Audit
{df_to_markdown_safe(df_ovf_sum, float_fmt="{:.2f}")}

{df_to_markdown_safe(df_ovf_deg, float_fmt="{:.2f}")}
"""
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"✅ Saved Exp 07 Markdown Report -> {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg_epoch", type=int, default=0, help="0 = auto-select best epoch by WSMAD")
    parser.add_argument("--ff_epoch", type=int, default=0, help="0 = auto-select best epoch by WSMAD")
    parser.add_argument("--skip_refresh", action="store_true")
    args = parser.parse_args()

    build_notebook_and_report(args)