"""논문용 PDF 그림 생성 (rs_rtc/ 에서 실행: python3 _tmp/paper/make_figures.py)."""
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RENDER_DIR = PROJECT_ROOT / "results" / "render"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

DELAYS = [0, 1, 2, 4, 7]
METHOD_STYLES = {
    "naive_async": ("Naive async", "#9e9e9e", "o", 0.7),
    "bid_n64": ("BID ($N{=}64$)", "#b39ddb", "s", 0.7),
    "rtc_paper": ("RTC", "#e65100", "^", 1.0),
    "rs_rtc_n64": ("RS-RTC (ours)", "#1565c0", "D", 1.0),
}
FIG3_EPISODE = 0
FIG3_STEPS = [50, 100, 150]
FIG2_SWITCH_INDEX = 3
VIDEO_FPS_PER_STEP = 3
# 영상 프레임 그림은 원본 해상도에 가깝게 저장
PNG_DPI = {"fig3_render_frames.pdf": 750, "fig0_method_overview.pdf": 600}

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def figure_delay_sweep(summary):
    """Fig 1: 지연 d 에 따른 성공률 (95% CI), 전환 변화량, prefix 오차."""
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    panels = (("success_pct", "Success rate (%)"), ("boundary_jump_px", "Boundary discontinuity (px)"),
              ("prefix_cost", "Prefix mismatch"))
    for axis, (column, title) in zip(axes, panels):
        for method, (label, color, marker, alpha) in METHOD_STYLES.items():
            rows = summary[summary.method == method].set_index("delay").loc[DELAYS]
            if column == "success_pct":
                bounds = np.array([[float(v) for v in ci.strip("[]").split(",")] for ci in rows.success_ci_pct])
                axis.fill_between(DELAYS, bounds[:, 0], bounds[:, 1], color=color, alpha=0.12 * alpha, lw=0)
            axis.plot(DELAYS, rows[column], color=color, marker=marker, ms=4, lw=1.4 if alpha == 1.0 else 1.0,
                      alpha=alpha, label=label)
        axis.set_title(title)
        axis.set_xlabel("Inference delay $d$ (steps)")
        axis.set_xticks(DELAYS)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False)
    return figure


def plan_steps(offset):
    """chunk 계획 인덱스 j 의 제어 스텝 (offset + j)."""
    return offset + np.arange(16)


def figure_switch_trajectory():
    """Fig 2: 같은 seed·같은 전환 시점의 이전 계획, 새 chunk, 실제 실행 (x, y 시계열)."""
    records = {method: np.load(RENDER_DIR / f"{method}_d4_ep{FIG3_EPISODE}.npz") for method in ("rtc_paper", "rs_rtc_n64")}
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="col")
    for column, (method, record) in enumerate(records.items()):
        switch_step = int(record["switch_steps"][FIG2_SWITCH_INDEX])
        previous_steps = plan_steps(int(record["previous_plan_offsets"][FIG2_SWITCH_INDEX]))
        new_steps = plan_steps(int(record["new_plan_offsets"][FIG2_SWITCH_INDEX]))
        delay = int(record["delay"])
        window = np.arange(max(switch_step - 8, 0), min(switch_step + 8, len(record["executed_actions"])))
        for row, (coordinate, name) in enumerate(((0, "x"), (1, "y"))):
            axis = axes[row, column]
            axis.axvspan(switch_step - delay - 0.5, switch_step - 0.5, color="#ffb300", alpha=0.15, lw=0,
                         label="inference window" if row == 0 else None)
            axis.axvline(switch_step - 0.5, color="k", lw=0.8, ls=":")
            axis.plot(previous_steps, record["previous_plans"][FIG2_SWITCH_INDEX][:, coordinate], color="#757575",
                      ls="--", lw=1.2, label="previous chunk")
            if record["candidate_plans"].shape[1] > 0:
                for candidate in record["candidate_plans"][FIG2_SWITCH_INDEX][:, :, coordinate]:
                    axis.plot(new_steps, candidate, color="#ff9800", lw=0.4, alpha=0.15)
            axis.plot(new_steps, record["new_plans"][FIG2_SWITCH_INDEX][:, coordinate],
                      color=METHOD_STYLES[method][1], lw=1.6, label="new chunk")
            axis.plot(window, record["executed_actions"][window, coordinate], color="k", lw=0, marker=".", ms=3,
                      label="executed action")
            axis.set_xlim(window[0], window[-1])
            if column == 0:
                axis.set_ylabel(f"action {name} (px)")
            if row == 0:
                axis.set_title(METHOD_STYLES[method][0])
            else:
                axis.set_xlabel("control step")
    axes[0, 1].plot([], [], color="#ff9800", lw=1, alpha=0.6, label="64 candidates")
    axes[0, 1].legend(loc="best", fontsize=6.5, frameon=False)
    figure.tight_layout()
    return figure


def video_frame(video_path, step):
    """mp4 에서 제어 스텝 step 의 프레임 (마지막 프레임 이후면 마지막)."""
    frames = iio.imread(video_path, plugin="pyav")
    return frames[min(step * VIDEO_FPS_PER_STEP, len(frames) - 1)]


def block_coverages(record):
    """기록된 블록 자세마다 gym-pusht 와 같은 방식으로 목표 겹침 비율 (coverage) 을 계산."""
    import gymnasium
    import gym_pusht  # noqa: F401

    environment = gymnasium.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")
    environment.reset(seed=int(record["env_seed"]))
    push_t = environment.unwrapped
    coverages = []
    for block_pose in record["block_poses"]:
        push_t.block.position = (float(block_pose[0]), float(block_pose[1]))
        push_t.block.angle = float(block_pose[2])
        coverages.append(push_t._get_coverage())
    environment.close()
    return np.array(coverages)


def hide_axis_frame(axis):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)


def figure_render_frames():
    """Fig 3: 같은 seed 의 진행도 곡선 (목표 겹침 %) 과 같은 시각의 MuJoCo 프레임 (마지막 열: 에피소드 종료)."""
    episodes = [("rs_rtc_n64", f"rs_rtc_n64_d4_ep{FIG3_EPISODE}", "RS-RTC (NPU)\n$d{=}4$"),
                ("rtc_paper", f"rtc_paper_d4_ep{FIG3_EPISODE}", "RTC (GPU)\n$d{=}4$"),
                ("per_step", f"per_step_d0_ep{FIG3_EPISODE}", "No chunking\n$d{=}0$")]
    colors = {"rs_rtc_n64": METHOD_STYLES["rs_rtc_n64"][1], "rtc_paper": METHOD_STYLES["rtc_paper"][1],
              "per_step": "#616161"}
    column_count = len(FIG3_STEPS) + 1
    figure = plt.figure(figsize=(7.0, 5.4))
    curve_grid = figure.add_gridspec(1, 1, left=0.08, right=0.99, top=0.98, bottom=0.66)
    grid = figure.add_gridspec(len(episodes), column_count, wspace=0.02, hspace=0.03, left=0.08, right=1.0,
                               top=0.57, bottom=0.0)

    # 진행도 곡선 (목표 겹침 %) 과 프레임 시각 표시 (chunk 없음 곡선을 맨 뒤에 그림)
    curve_axis = figure.add_subplot(curve_grid[0, 0])
    for method, stem, label in reversed(episodes):
        record = np.load(RENDER_DIR / f"{stem}.npz")
        coverage_pct = 100 * block_coverages(record)
        seconds = np.arange(len(coverage_pct)) / 10
        success = bool(record["success"])
        curve_axis.plot(seconds, coverage_pct, color=colors[method], lw=1.4, label=label.replace("\n", ", "))
        curve_axis.plot(seconds[-1], coverage_pct[-1], color=colors[method], marker="*" if success else "X", ms=9)
    curve_axis.axhline(95, color="k", ls="--", lw=0.8)
    curve_axis.text(0.3, 96.5, "success (95%)", fontsize=7, va="bottom")
    for step in FIG3_STEPS:
        curve_axis.axvline(step / 10, color="k", ls=":", lw=0.7)
    curve_axis.set_xlabel("time (s)", labelpad=1)
    curve_axis.set_ylabel("goal coverage (%)")
    curve_axis.set_ylim(0, 105)
    curve_axis.set_xlim(0, 30.5)
    handles, labels = curve_axis.get_legend_handles_labels()
    curve_axis.legend(handles[::-1], labels[::-1], fontsize=7, frameon=False, loc="lower right")

    # 같은 시각의 프레임과 마지막 프레임 (각각 겹침 % 표시)
    for row, (method, stem, label) in enumerate(episodes, start=1):
        record = np.load(RENDER_DIR / f"{stem}.npz")
        coverage_pct = 100 * block_coverages(record)
        frames = iio.imread(RENDER_DIR / "mujoco" / f"{stem}.mp4", plugin="pyav")
        last_step = len(coverage_pct) - 1
        result = "success" if bool(record["success"]) else "fail"
        steps = FIG3_STEPS + [last_step]
        for column, step in enumerate(steps):
            axis = figure.add_subplot(grid[row - 1, column])
            frame_index = min(step * VIDEO_FPS_PER_STEP, len(frames) - 1) if column < len(FIG3_STEPS) else len(frames) - 1
            axis.imshow(frames[frame_index], interpolation="none")
            hide_axis_frame(axis)
            caption = f"{coverage_pct[min(step, last_step)]:.0f}%"
            if column == len(FIG3_STEPS):
                caption = f"end t={last_step / 10:.1f}s, {caption} ({result})"
            axis.text(0.98, 0.16, caption, transform=axis.transAxes, ha="right", va="bottom", fontsize=6.5,
                      color="white", bbox=dict(boxstyle="round,pad=0.2", fc=colors[method], ec="none", alpha=0.9))
            if row == 1:
                axis.set_title(f"t = {step / 10:.0f} s" if column < len(FIG3_STEPS) else "episode end", fontsize=8,
                               pad=2)
            if column == 0:
                axis.set_ylabel(label, fontsize=8, color=colors[method])
    return figure


def figure_latency_scaling(summary):
    """Fig 4: NPU particle 수 N 에 따른 chunk 생성 시간과 RTC (GPU) 생성 시간."""
    bench = pd.read_csv(METRICS_DIR / "bench_latency.csv")
    rtc_latency = summary[summary.method == "rtc_paper"].generation_p50_ms.mean()
    figure, axis = plt.subplots(figsize=(3.3, 2.3))
    axis.plot(bench.batch_size, bench.npu_sampling_p50_ms, color=METHOD_STYLES["rs_rtc_n64"][1], marker="D", ms=4,
              label="RS-RTC sampling (NPU)")
    axis.axhline(rtc_latency, color=METHOD_STYLES["rtc_paper"][1], ls="--", lw=1.2,
                 label=f"RTC (GPU H200), {rtc_latency:.0f} ms")
    axis.axvline(64, color="k", ls=":", lw=0.8)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks(bench.batch_size)
    axis.set_xticklabels(bench.batch_size)
    axis.set_xlabel("Particles $N$")
    axis.set_ylabel("Chunk generation p50 (ms)")
    axis.legend(fontsize=7, frameon=False, loc="upper left")
    figure.tight_layout()
    return figure


# Fig 0 개념도: 단일 단 (8.1 cm) 인쇄 크기, 한글 라벨, 색각 이상 친화 팔레트 (Okabe-Ito 계열)
# savefig(bbox_inches="tight") 가 사방에 0.1 inch 여백을 더하므로, 저장 결과 폭이 8.1 cm 가 되도록 뺀다
OVERVIEW_WIDTH_INCH = 8.1 / 2.54 - 0.2
# 라틴 문자·숫자는 Times New Roman, 한글은 글리프 단위 대체로 바탕체
OVERVIEW_FONT = ["Times New Roman", "Batang"]
# 바탕체에 굵은 글꼴이 없어 굵은 제목은 같은 색 외곽선으로 굵게 보이게 한다
OVERVIEW_BOLD_STROKE_WIDTH = 0.35
OVERVIEW_FONT_SIZE = 7
EXECUTED_COLOR = "#333333"
FROZEN_COLOR = "#F0C419"
FROZEN_EDGE_COLOR = "#A87800"
DISCARDED_COLOR = "#9a9a9a"
NAIVE_COLOR = "#CC79A7"
RTC_COLOR = "#D55E00"
RS_RTC_COLOR = "#0072B2"
DROPPED_COLOR = "#bdbdbd"
CONCEPT_SWITCH = 8.0
CONCEPT_DELAY = 4.0
CONCEPT_FROZEN_START = CONCEPT_SWITCH - CONCEPT_DELAY
NAIVE_OFFSET = -1.15


def concept_old_action(times):
    """이전 chunk 의 모식 행동 (시간에 따른 1차원 값)."""
    times = np.asarray(times, dtype=float)
    return 0.9 * np.sin(0.38 * times - 0.6)


def concept_new_action(times):
    """확정 구간과 맞는 새 chunk: 전환 전까지 이전 chunk 와 같고, 이후 새 관측에 따라 위로 꺾인다."""
    times = np.asarray(times, dtype=float)
    bend = np.clip(times - CONCEPT_SWITCH, 0, None)
    return concept_old_action(times) + 0.045 * bend ** 2


def concept_candidate(times, offset, seed):
    """확정 구간과 offset 만큼 어긋난 후보: 낮은 주파수의 흔들림을 더해 서로 다른 궤적처럼 보이게 한다."""
    generator = np.random.default_rng(seed)
    amplitude, phase = generator.uniform(0.05, 0.15), generator.uniform(0, 2 * np.pi)
    return concept_new_action(times) + offset + amplitude * np.sin(0.7 * times + phase)


def draw_concept_context(axis, time_start):
    """공통 배경: 실행된 행동, 확정 구간 띠, 전환 시점 점선."""
    axis.axvspan(CONCEPT_FROZEN_START, CONCEPT_SWITCH, color=FROZEN_COLOR, alpha=0.16, lw=0)
    executed_times = np.linspace(time_start, CONCEPT_FROZEN_START, 30)
    frozen_times = np.linspace(CONCEPT_FROZEN_START, CONCEPT_SWITCH, 30)
    axis.plot(frozen_times, concept_old_action(frozen_times), color=FROZEN_COLOR, lw=5.0, solid_capstyle="butt",
              zorder=2)
    axis.plot(frozen_times, concept_old_action(frozen_times), color=FROZEN_EDGE_COLOR, lw=0.9, zorder=3)
    axis.plot(executed_times, concept_old_action(executed_times), color=EXECUTED_COLOR, lw=1.6,
              solid_capstyle="round", zorder=3)
    axis.axvline(CONCEPT_SWITCH, color="k", lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    axis.axis("off")


def draw_concept_chunk(axis, times, values, color, line_width=1.3, alpha=1.0, zorder=4):
    """새 chunk: 확정 구간과 겹치는 앞부분은 점선, 전환 후 실행되는 부분은 실선."""
    before = times <= CONCEPT_SWITCH
    after = times >= CONCEPT_SWITCH
    axis.plot(times[before], values[before], color=color, lw=line_width * 0.8, ls=(0, (2, 1.2)), alpha=alpha,
              zorder=zorder)
    axis.plot(times[after], values[after], color=color, lw=line_width, alpha=alpha, solid_capstyle="round",
              zorder=zorder)


def draw_problem_panel(axis):
    """(a) 새 chunk 생성 중에도 이전 chunk 가 확정 구간을 실행하고, 이를 무시한 새 chunk 는 전환 시점에 튄다."""
    draw_concept_context(axis, 0.0)
    discarded_times = np.linspace(CONCEPT_SWITCH, 14.5, 30)
    axis.plot(discarded_times, concept_old_action(discarded_times), color=DISCARDED_COLOR, lw=1.0, ls=(0, (3, 2)),
              zorder=2)
    new_times = np.linspace(CONCEPT_FROZEN_START, 14.5, 80)
    draw_concept_chunk(axis, new_times, concept_new_action(new_times) + NAIVE_OFFSET, NAIVE_COLOR, line_width=1.6)

    # 전환 시점의 튐
    old_at_switch = concept_old_action(CONCEPT_SWITCH)
    new_at_switch = concept_new_action(CONCEPT_SWITCH) + NAIVE_OFFSET
    axis.annotate("", xy=(CONCEPT_SWITCH, new_at_switch + 0.06), xytext=(CONCEPT_SWITCH, old_at_switch - 0.08),
                  arrowprops=dict(arrowstyle="-|>", color=NAIVE_COLOR, lw=1.3, mutation_scale=8), zorder=6)

    # 새 chunk 생성 기간 (d 스텝) = 확정 구간
    generation_y = -1.85
    axis.annotate("", xy=(CONCEPT_SWITCH, generation_y), xytext=(CONCEPT_FROZEN_START, generation_y),
                  arrowprops=dict(arrowstyle="|-|", color="#555555", lw=0.8, mutation_scale=2.5))
    axis.text(CONCEPT_FROZEN_START - 0.25, generation_y, "생성 중 (d 스텝)", ha="right", va="center",
              color="#555555")

    # 라벨
    axis.text(5.9, 1.9, "확정 구간", ha="center", va="top", color=FROZEN_EDGE_COLOR, fontweight="bold",
              path_effects=[patheffects.withStroke(linewidth=OVERVIEW_BOLD_STROKE_WIDTH,
                                                   foreground=FROZEN_EDGE_COLOR)])
    axis.text(9.6, concept_old_action(9.6) + 0.18, "이전 chunk", ha="left", va="bottom", color="#777777")
    axis.text(11.2, concept_new_action(11.2) + NAIVE_OFFSET - 0.2, "새 chunk", ha="center", va="top",
              color=NAIVE_COLOR)
    axis.text(14.9, -2.2, "시간 →", ha="right", va="bottom", color="#777777")
    axis.set_xlim(-0.2, 15.0)
    axis.set_ylim(-2.2, 1.95)


SNAPSHOT_TIME_START = 2.2
SNAPSHOT_TIME_END = 13.0
SNAPSHOT_TIMES = np.linspace(CONCEPT_FROZEN_START, SNAPSHOT_TIME_END, 70)


def draw_snapshot_frame(axis):
    draw_concept_context(axis, SNAPSHOT_TIME_START)
    axis.set_xlim(SNAPSHOT_TIME_START - 0.2, SNAPSHOT_TIME_END + 0.2)
    axis.set_ylim(-2.0, 1.65)


def draw_rtc_snapshot(axis, stage):
    """(b) RTC: 후보 하나를 역확산 step 마다 역전파 기울기로 확정 구간 쪽으로 당긴다."""
    draw_snapshot_frame(axis)
    offset = (-1.2, -0.9, 0.0)[stage]
    # 간격이 좁은 중간 단계는 화살표 몸통이 보이도록 양 끝 여백을 없앤다
    arrow_shrink = (2, 1, 0)[stage]
    wiggle_scale = (1.0, 0.5, 0.0)[stage]
    values = concept_new_action(SNAPSHOT_TIMES) + offset + wiggle_scale * 0.15 * np.sin(0.7 * SNAPSHOT_TIMES + 1.0)
    draw_concept_chunk(axis, SNAPSHOT_TIMES, values, RTC_COLOR, line_width=1.8 if stage == 2 else 1.4)
    if stage < 2:
        for arrow_time in (5.0, 7.0):
            index = np.argmin(np.abs(SNAPSHOT_TIMES - arrow_time))
            axis.annotate("", xy=(arrow_time, concept_old_action(arrow_time) - 0.16),
                          xytext=(arrow_time, values[index] + 0.08),
                          arrowprops=dict(arrowstyle="-|>", color=RTC_COLOR, lw=0.9, mutation_scale=6,
                                          shrinkA=arrow_shrink, shrinkB=arrow_shrink), zorder=6)


RS_RTC_STAGE_OFFSETS = (np.array([-1.5, -1.05, -0.6, -0.2, 0.15, 0.55]),
                        np.array([-0.62, -0.3, -0.12, 0.06, 0.22, 0.45]),
                        np.array([-0.16, -0.08, 0.0, 0.07, 0.14]))
RS_RTC_DROP_THRESHOLDS = (0.45, 0.4, 1.0)


def draw_rs_rtc_snapshot(axis, stage):
    """(c) RS-RTC: 후보 N 개 중 확정 구간과 어긋난 후보는 제거, 맞는 후보는 복제 후 잡음으로 다시 퍼진다."""
    draw_snapshot_frame(axis)
    dropped_count = 0
    for index, offset in enumerate(RS_RTC_STAGE_OFFSETS[stage]):
        values = concept_candidate(SNAPSHOT_TIMES, offset, seed=10 * stage + index)
        if stage == 2:
            spread = 1 + 0.6 * np.clip(SNAPSHOT_TIMES - CONCEPT_SWITCH, 0, None) / 5
            values = concept_new_action(SNAPSHOT_TIMES) + offset * spread
        if abs(offset) > RS_RTC_DROP_THRESHOLDS[stage]:
            draw_concept_chunk(axis, SNAPSHOT_TIMES, values, DROPPED_COLOR, line_width=0.9, zorder=3)
            marker_index = np.argmin(np.abs(SNAPSHOT_TIMES - (4.9 + 0.9 * dropped_count)))
            axis.plot(SNAPSHOT_TIMES[marker_index], values[marker_index], marker="x", ms=4, mew=1.1,
                      color="#444444", zorder=7)
            dropped_count += 1
        else:
            draw_concept_chunk(axis, SNAPSHOT_TIMES, values, RS_RTC_COLOR, line_width=0.9,
                               alpha=0.4 if stage == 2 else 0.9)
    if stage == 2:
        draw_concept_chunk(axis, SNAPSHOT_TIMES, concept_new_action(SNAPSHOT_TIMES), RS_RTC_COLOR, line_width=1.8,
                           zorder=5)


def figure_method_overview():
    """Fig 0: 방법 개념도. (a) 문제 정의, (b) RTC 와 (c) RS-RTC 의 디노이징 단계별 비교 (같은 축)."""
    overview_style = {"font.family": OVERVIEW_FONT, "font.size": OVERVIEW_FONT_SIZE, "axes.unicode_minus": False,
                      "axes.grid": False}
    # 세로 배치 (inch): (a) 제목은 패널 왼쪽 위 빈 곳에 겹쳐 놓고, (b), (c) 는 제목 + 스냅샷 행, 단계 라벨
    title_height, problem_height, row_height, gap, step_label_height = 0.14, 0.66, 0.4, 0.03, 0.14
    height = problem_height + gap + 2 * (title_height + row_height + gap) + step_label_height
    with plt.rc_context(overview_style):
        figure = plt.figure(figsize=(OVERVIEW_WIDTH_INCH, height))
        title_style = dict(fontsize=8, fontweight="bold", ha="left", va="top")
        title_colors = {"(a) 문제 정의": "k", "(b) RTC": RTC_COLOR, "(c) RS-RTC (ours)": RS_RTC_COLOR}

        def bold_stroke(title):
            return [patheffects.withStroke(linewidth=OVERVIEW_BOLD_STROKE_WIDTH, foreground=title_colors[title])]
        top = height

        def to_figure(y_inch):
            return y_inch / height

        # (a) 문제 정의
        figure.text(0.01, to_figure(top), "(a) 문제 정의", path_effects=bold_stroke("(a) 문제 정의"), **title_style)
        draw_problem_panel(figure.add_axes((0.0, to_figure(top - problem_height), 1.0, to_figure(problem_height))))
        top -= problem_height + gap

        # (b), (c): 같은 축·같은 배치로 디노이징 단계 3개를 나란히
        snapshot_left, snapshot_width, snapshot_gap = 0.0, 0.32, 0.02
        columns = [snapshot_left + column * (snapshot_width + snapshot_gap) for column in range(3)]
        for title, color, draw_snapshot in (("(b) RTC", RTC_COLOR, draw_rtc_snapshot),
                                            ("(c) RS-RTC (ours)", RS_RTC_COLOR, draw_rs_rtc_snapshot)):
            figure.text(0.01, to_figure(top), title, color=color, path_effects=bold_stroke(title), **title_style)
            top -= title_height
            bottom = to_figure(top - row_height)
            snapshot_axes = []
            for stage, left in enumerate(columns):
                axis = figure.add_axes((left, bottom, snapshot_width, to_figure(row_height)))
                draw_snapshot(axis, stage)
                snapshot_axes.append(axis)
            for left in columns[1:]:
                figure.text(left - snapshot_gap / 2, bottom + to_figure(row_height) / 2, "›", ha="center",
                            va="center", fontsize=11, color="#888888")
            top -= row_height + gap

        # 디노이징 단계 (두 행 공통)
        for left, label in zip(columns, ("역확산 초기", "중간", "최종")):
            figure.text(left + snapshot_width / 2, to_figure(top + gap - 0.03), label, ha="center", va="top",
                        color="#555555")
    return figure


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(METRICS_DIR / "summary_main500.csv")
    figure_builders = (("fig0_method_overview.pdf", figure_method_overview),
                       ("fig1_delay_sweep.pdf", lambda: figure_delay_sweep(summary)),
                       ("fig2_switch_trajectory.pdf", figure_switch_trajectory),
                       ("fig3_render_frames.pdf", figure_render_frames),
                       ("fig4_latency_scaling.pdf", lambda: figure_latency_scaling(summary)))

    # 인자로 그림 이름 일부를 주면 해당 그림만 생성 (예: fig0)
    selected_names = sys.argv[1:]
    for file_name, builder in figure_builders:
        if selected_names and not any(name in file_name for name in selected_names):
            continue
        figure = builder()
        figure.savefig(FIGURE_DIR / file_name, bbox_inches="tight", dpi=200)
        figure.savefig(FIGURE_DIR / file_name.replace(".pdf", ".png"), bbox_inches="tight", dpi=PNG_DPI.get(file_name, 200))
        plt.close(figure)
        print(f"저장: {FIGURE_DIR / file_name}")


if __name__ == "__main__":
    main()
