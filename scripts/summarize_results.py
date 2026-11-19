"""eval_async 결과 CSV 를 방법 × 지연별로 집계한다.

집계 항목: 에피소드 수, 성공률 (Wilson 95% CI), 평균 최대 보상, chunk 전환 변화량 평균 (95% CI), 내부 변화량,
         전환/내부 비율, inpainting 오차 (prefix cost), 생성 지연 p50, (RS-RTC) resampling 횟수·최소 ESS
결과: results/metrics/summary_{이름}.csv, 표는 로그로 출력
실행 (컨테이너 내부 또는 호스트, rs_rtc/ 에서):
    python3 scripts/summarize_results.py
    python3 scripts/summarize_results.py --methods naive_async bid_n64 --name lambda_sweep
"""
import argparse
import math

import pandas as pd

from rs_rtc import RESULTS_DIR
from rs_rtc.utils.logging import setup_logger

EVAL_ASYNC_DIR = RESULTS_DIR / "metrics" / "eval_async"
WILSON_Z = 1.96

LOGGER = setup_logger("rs_rtc", "summarize_results")


def parse_arguments():
    parser = argparse.ArgumentParser(description="비동기 평가 결과 집계")
    parser.add_argument("--methods", nargs="+", default=None, help="집계할 방법 (기본: 전체)")
    parser.add_argument("--name", default="all", help="요약 파일 이름")
    return parser.parse_args()


def wilson_interval(success_count, trial_count):
    """성공률의 Wilson 95% 신뢰구간 (하한, 상한)."""
    success_ratio = success_count / trial_count
    denominator = 1.0 + WILSON_Z ** 2 / trial_count
    center = (success_ratio + WILSON_Z ** 2 / (2 * trial_count)) / denominator
    half_width = WILSON_Z * math.sqrt(success_ratio * (1 - success_ratio) / trial_count
                                      + WILSON_Z ** 2 / (4 * trial_count ** 2)) / denominator
    return center - half_width, center + half_width


def summarize_group(group_rows):
    """방법 1개 × 지연 1개 에피소드 묶음을 요약 행으로 만든다."""
    episode_count = len(group_rows)
    success_count = int(group_rows["success"].sum())
    success_low, success_high = wilson_interval(success_count, episode_count)
    jump_mean = group_rows["boundary_jump_px"].mean()
    jump_half_width = WILSON_Z * group_rows["boundary_jump_px"].std() / math.sqrt(episode_count)
    interior_mean = group_rows["interior_step_px"].mean()

    summary_row = {
        "episodes": episode_count,
        "success_pct": round(100 * success_count / episode_count, 1),
        "success_ci_pct": f"[{100 * success_low:.1f}, {100 * success_high:.1f}]",
        "max_reward": round(group_rows["max_reward"].mean(), 3),
        "boundary_jump_px": round(jump_mean, 2),
        "boundary_jump_ci_px": round(jump_half_width, 2),
        "interior_step_px": round(interior_mean, 2),
        "jump_ratio": round(jump_mean / interior_mean, 2),
        "generation_p50_ms": round(group_rows["generation_p50_ms"].mean(), 1),
    }
    for optional_column in ("mean_prefix_cost", "mean_resample_count", "mean_min_ess", "mean_selected_cost"):
        if optional_column in group_rows:
            summary_row[optional_column.replace("mean_", "")] = round(group_rows[optional_column].mean(), 4)
    return pd.Series(summary_row)


def main():
    arguments = parse_arguments()

    # 결과 CSV 로딩
    result_frames = [pd.read_csv(csv_path) for csv_path in sorted(EVAL_ASYNC_DIR.glob("*_d*.csv"))]
    all_episodes = pd.concat(result_frames, ignore_index=True)
    if arguments.methods:
        all_episodes = all_episodes[all_episodes["method"].isin(arguments.methods)]

    # 방법 × 지연 별 요약
    summary_table = all_episodes.groupby(["method", "delay"]).apply(summarize_group).reset_index()
    summary_path = RESULTS_DIR / "metrics" / f"summary_{arguments.name}.csv"
    summary_table.to_csv(summary_path, index=False)
    LOGGER.info("요약 저장: %s\n%s", summary_path, summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
