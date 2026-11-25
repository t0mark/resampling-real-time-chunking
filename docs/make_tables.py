"""논문용 LaTeX 표 생성 (rs_rtc/ 에서 실행: python3 _tmp/paper/make_tables.py)."""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
TABLE_DIR = Path(__file__).resolve().parent / "tables"
BUILD_DIR = TABLE_DIR / "build"
RENDER_DPI = 300
STANDALONE_TEMPLATE = r"""\documentclass[border=6pt]{standalone}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{mathptmx}
\begin{document}
\input{%s}
\end{document}
"""

DELAYS = [0, 1, 2, 4, 7]
MAIN_METHODS = [("naive_async", "Naive async"), ("bid_n64", "BID ($N{=}64$)"),
                ("rtc_paper", "RTC (GPU)"), ("rs_rtc_n64", "RS-RTC (NPU, ours)")]


def format_ci(ci_text):
    low, high = [float(value) for value in ci_text.strip("[]").split(",")]
    return low, high


def table_main(summary):
    """Tab 1: 지연 d 별 성공률·최대 보상·전환 변화량·prefix 오차."""
    lines = [r"\begin{tabular}{cl cccc}", r"\toprule",
             r"$d$ & Method & Success (\%) $\uparrow$ & Max reward $\uparrow$ & Switch jump (px) $\downarrow$ & Prefix error $\downarrow$ \\",
             r"\midrule"]
    for delay in DELAYS:
        rows = summary[summary.delay == delay].set_index("method")
        best_jump = rows.loc[[m for m, _ in MAIN_METHODS], "boundary_jump_px"].min()
        best_prefix = rows.loc[[m for m, _ in MAIN_METHODS], "prefix_cost"].min()
        for index, (method, label) in enumerate(MAIN_METHODS):
            row = rows.loc[method]
            low, high = format_ci(row.success_ci_pct)
            jump = f"{row.boundary_jump_px:.2f}"
            prefix = f"{row.prefix_cost:.4f}"
            jump = rf"\textbf{{{jump}}}" if np.isclose(row.boundary_jump_px, best_jump) else jump
            prefix = rf"\textbf{{{prefix}}}" if np.isclose(row.prefix_cost, best_prefix) else prefix
            delay_cell = rf"\multirow{{4}}{{*}}{{{delay}}}" if index == 0 else ""
            lines.append(f"{delay_cell} & {label} & {row.success_pct:.1f} {{\\scriptsize[{low:.1f}, {high:.1f}]}} & "
                         f"{row.max_reward:.3f} & {jump} & {prefix} \\\\")
        lines.append(r"\midrule" if delay != DELAYS[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def table_latency(summary):
    """Tab 2: 장치, 역전파 필요 여부, 생성 시간, 에피소드당 추론 횟수, d=0 성공률."""
    entries = [("per_step", "No chunking (re-infer every step)", "NPU", "No"),
               ("naive_async", "Naive async", "NPU", "No"),
               ("bid_n64", "BID ($N{=}64$)", "NPU", "No"),
               ("rtc_paper", "RTC", "GPU (H200)", "Yes"),
               ("rs_rtc_n64", "RS-RTC (ours)", "NPU (ATOM-Max)", "No")]
    lines = [r"\begin{tabular}{l l c c c c}", r"\toprule",
             r"Method & Device & Backprop & Latency p50 (ms) & Inferences / episode & Success $d{=}0$ (\%) \\",
             r"\midrule"]
    for method, label, device, backprop in entries:
        row = summary[(summary.method == method) & (summary.delay == 0)].iloc[0]
        episode_rows = pd.read_csv(METRICS_DIR / "eval_async" / f"{method}_d0.csv")
        latency = summary[summary.method == method].generation_p50_ms.mean()
        lines.append(f"{label} & {device} & {backprop} & {latency:.1f} & {episode_rows.chunk_count.mean():.1f} & "
                     f"{row.success_pct:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def sweep_rows(sweep_dir, pattern, parameter_of):
    """검증 sweep CSV 들을 (파라미터, 성공률, 전환 변화량, prefix 오차) 로 집계."""
    rows = []
    for csv_path in sorted(sweep_dir.glob(pattern)):
        episode_rows = pd.read_csv(csv_path)
        prefix_column = "mean_prefix_cost" if "mean_prefix_cost" in episode_rows else "mean_selected_cost"
        rows.append((parameter_of(csv_path.stem), 100 * episode_rows.success.mean(),
                     episode_rows.boundary_jump_px.mean(), episode_rows[prefix_column].mean()))
    return sorted(rows)


def table_tuning():
    """Tab 3: RS-RTC λ, RTC β 선택 (검증 seed, d=4, 50 에피소드)."""
    lambda_rows = sweep_rows(METRICS_DIR / "sweep_lambda_val", "rs_rtc_n64_l*_d4.csv",
                             lambda stem: int(stem.split("_l")[1].split("_")[0]))
    beta_rows = sweep_rows(METRICS_DIR / "sweep_beta_val", "rtc_paper_b*_d4.csv",
                           lambda stem: int(stem.split("_b")[1].split("_")[0]))
    lines = [r"\begin{tabular}{l c c c c}", r"\toprule",
             r"Method & Value & Success (\%) & Switch jump (px) & Prefix error \\", r"\midrule"]
    for name, symbol, rows, chosen in (("RS-RTC", r"\lambda", lambda_rows, 1000), ("RTC", r"\beta", beta_rows, 5)):
        for index, (value, success, jump, prefix) in enumerate(rows):
            method_cell = rf"\multirow{{{len(rows)}}}{{*}}{{{name}}}" if index == 0 else ""
            value_cell = f"${symbol}={value}$" + (r"$^\star$" if value == chosen else "")
            lines.append(f"{method_cell} & {value_cell} & {success:.1f} & {jump:.2f} & {prefix:.4f} \\\\")
        lines.append(r"\midrule" if name == "RS-RTC" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_table_image(table_path):
    """표 tex 를 standalone 문서로 컴파일해 PDF 와 PNG 이미지로 저장."""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_path = BUILD_DIR / f"{table_path.stem}_standalone.tex"
    wrapper_path.write_text(STANDALONE_TEMPLATE % table_path.resolve())
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", wrapper_path.name], cwd=BUILD_DIR,
                   check=True, capture_output=True)
    pdf_path = TABLE_DIR / f"{table_path.stem}.pdf"
    shutil.copy(BUILD_DIR / f"{wrapper_path.stem}.pdf", pdf_path)
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", str(RENDER_DPI), str(pdf_path),
                    str(TABLE_DIR / table_path.stem)], check=True)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(METRICS_DIR / "summary_main500.csv")
    for file_name, content in (("tab1_main.tex", table_main(summary)), ("tab2_latency.tex", table_latency(summary)),
                               ("tab3_tuning.tex", table_tuning())):
        (TABLE_DIR / file_name).write_text(content + "\n")
        render_table_image(TABLE_DIR / file_name)
        print(f"저장: {TABLE_DIR / file_name} (+ .pdf, .png)")


if __name__ == "__main__":
    main()
