"""렌더 기록 (results/render/*.npz) 을 MuJoCo 3D 장면 (테이블, T 블록, Franka) 영상으로 만든다.

기록된 끝단·블록 상태를 control_hz 에서 fps 로 보간해 재생하고 (물리 시뮬레이션 아님),
밀대 끝점이 끝단 궤적을 따라가도록 팔을 역기구학으로 맞춘다. chunk 경로·후보·타임라인을 겹쳐 그리고,
밀대를 따라가며 위에서 본 확대 화면 (로봇팔 숨김) 을 좌하단에 넣는다.
결과: results/render/mujoco/{기록 이름}.mp4, 로그 results/logs/render_mujoco_*.log
실행 (CPU 렌더링, rs_rtc/ 에서):
    python3 scripts/render/render_mujoco.py                                              # 모든 기록
    python3 scripts/render/render_mujoco.py --files results/render/rs_rtc_n64_d4_ep0.npz # 일부 기록
"""
import os

# CPU 소프트웨어 OpenGL 렌더링 (mujoco import 전에 설정해야 적용됨, 외부 지정값 우선)
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import imageio
import mujoco
import numpy as np
import yaml

from arm_ik import ArmIkSolver
from overlay import ChunkOverlay, ChunkTimeline, HudPainter
from scene import PushTScene

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDER_DIR = PROJECT_ROOT / "results" / "render"
VIDEO_DIR = RENDER_DIR / "mujoco"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
PANDA_XML_PATH = PROJECT_ROOT / "assets" / "franka_emika_panda" / "panda.xml"
MAX_OVERLAY_GEOMS = 6000
ARM_VISUAL_GEOM_GROUP = 2
RESULT_HOLD_SECONDS = 1.5
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

LOGGER = logging.getLogger("render_mujoco")


def setup_logger():
    """콘솔과 results/logs/render_mujoco_{시각}_{pid}.log 에 기록."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file_path = LOG_DIR / f"render_mujoco_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file_path)):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.info("로그 파일: %s", log_file_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="렌더 기록 → MuJoCo 3D 영상")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "render.yaml"))
    parser.add_argument("--files", nargs="+", default=None, help="렌더 기록 npz 경로 (기본: results/render/*.npz)")
    return parser.parse_args()


def interpolate_state(agent_positions, block_poses, step):
    """실수 스텝 step 의 끝단 위치와 블록 자세를 인접 기록 사이 선형 보간 (각도는 최단 회전)."""
    lower_index = min(int(math.floor(step)), len(agent_positions) - 1)
    upper_index = min(lower_index + 1, len(agent_positions) - 1)
    fraction = step - lower_index
    agent_position = (1 - fraction) * agent_positions[lower_index] + fraction * agent_positions[upper_index]
    block_pose = (1 - fraction) * block_poses[lower_index] + fraction * block_poses[upper_index]
    angle_difference = math.remainder(block_poses[upper_index][2] - block_poses[lower_index][2], 2 * math.pi)
    block_pose[2] = block_poses[lower_index][2] + fraction * angle_difference
    return agent_position, block_pose


def create_camera(camera_config):
    """설정의 자유 카메라 (바라보는 점, 거리, 방위각, 고도각)."""
    camera = mujoco.MjvCamera()
    camera.lookat[:] = camera_config["lookat"]
    camera.distance = camera_config["distance"]
    camera.azimuth = camera_config["azimuth"]
    camera.elevation = camera_config["elevation"]
    return camera


def create_inset_camera(inset_config):
    """밀대를 따라가며 위에서 내려다보는 확대 카메라 (바라보는 점은 프레임마다 갱신)."""
    camera = mujoco.MjvCamera()
    camera.distance = inset_config["distance"]
    camera.azimuth = inset_config["azimuth"]
    camera.elevation = inset_config["elevation"]
    return camera


def method_label(record_path, method_labels):
    """기록 파일 이름의 방법 이름으로 화면 표시 이름을 찾는다."""
    method_name = record_path.stem.rsplit("_d", 1)[0]
    return method_labels.get(method_name, method_name)


def render_record(record_path, render_config):
    """기록 1 개를 영상으로 렌더링."""
    render_record_data = np.load(record_path)
    width, height, fps = render_config["width"], render_config["height"], render_config["fps"]
    control_hz = render_config["control_hz"]

    # 장면, 역기구학, 렌더러, 표시 도구
    push_t_scene = PushTScene(PANDA_XML_PATH, render_config["scene"], render_record_data["goal_pose"], width, height)
    arm_ik_solver = ArmIkSolver(push_t_scene.model, push_t_scene.data, "pusher_tip_site")
    renderer = mujoco.Renderer(push_t_scene.model, height, width, max_geom=MAX_OVERLAY_GEOMS)
    camera = create_camera(render_config["camera"])
    inset_config = render_config["inset"]
    inset_renderer = mujoco.Renderer(push_t_scene.model, inset_config["size"], inset_config["size"],
                                     max_geom=MAX_OVERLAY_GEOMS)
    inset_camera = create_inset_camera(inset_config)
    inset_scene_option = mujoco.MjvOption()
    inset_scene_option.geomgroup[ARM_VISUAL_GEOM_GROUP] = 0
    timeline = ChunkTimeline(render_record_data)
    chunk_overlay = ChunkOverlay(push_t_scene, timeline)
    hud_painter = HudPainter(timeline, method_label(record_path, render_config["method_labels"]),
                             bool(render_record_data["success"]), control_hz, width, height)

    # 프레임 수 (재생 + 결과 표시 유지)
    agent_positions = render_record_data["agent_positions"]
    block_poses = render_record_data["block_poses"]
    playback_frame_count = int(timeline.step_count * fps / control_hz) + 1
    frame_count = playback_frame_count + int(RESULT_HOLD_SECONDS * fps)
    LOGGER.info("렌더 시작: %s (스텝 %d, chunk %d, 지연 %d, 프레임 %d)", record_path.name, timeline.step_count,
                timeline.chunk_count, timeline.delay, frame_count)

    # 프레임 생성과 영상 저장
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_DIR / f"{record_path.stem}.mp4"
    render_start = time.perf_counter()
    with imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as video_writer:
        for frame_index in range(frame_count):
            step = min(frame_index * control_hz / fps, timeline.step_count)
            agent_position, block_pose = interpolate_state(agent_positions, block_poses, step)
            push_t_scene.set_block_pose(block_pose)
            arm_ik_solver.solve(push_t_scene.table_position(agent_position, push_t_scene.pusher_tip_height))
            # 전체 장면
            renderer.update_scene(push_t_scene.data, camera)
            chunk_overlay.add_to_scene(renderer.scene, agent_positions, step)
            scene_frame = renderer.render()

            # 밀대 중심 확대 화면 (로봇팔 메시 숨김)
            inset_camera.lookat[:] = push_t_scene.table_position(agent_position, 0.0)
            inset_renderer.update_scene(push_t_scene.data, inset_camera, inset_scene_option)
            chunk_overlay.add_to_scene(inset_renderer.scene, agent_positions, step)
            inset_frame = inset_renderer.render()

            # HUD 와 저장
            frame = hud_painter.paint(scene_frame, step, frame_index >= playback_frame_count, inset_frame)
            video_writer.append_data(frame)
            if (frame_index + 1) % 100 == 0:
                LOGGER.info("  %s: %d/%d 프레임 (%.1f s)", record_path.stem, frame_index + 1, frame_count,
                            time.perf_counter() - render_start)

    renderer.close()
    inset_renderer.close()
    LOGGER.info("렌더 완료: %s (%.1f s, 역기구학 최대 오차 %.2f mm)", video_path, time.perf_counter() - render_start,
                arm_ik_solver.max_error * 1e3)


def main():
    arguments = parse_arguments()
    setup_logger()

    # 설정과 대상 기록
    with open(arguments.config) as config_file:
        render_config = yaml.safe_load(config_file)["mujoco"]
    record_paths = [Path(path) for path in arguments.files] if arguments.files else sorted(RENDER_DIR.glob("*.npz"))
    LOGGER.info("설정: %s, 기록 %d 개, MUJOCO_GL=%s", arguments.config, len(record_paths), os.environ["MUJOCO_GL"])

    # 기록별 렌더링
    for record_path in record_paths:
        render_record(record_path, render_config)


if __name__ == "__main__":
    main()
