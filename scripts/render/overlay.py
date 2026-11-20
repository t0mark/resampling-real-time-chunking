"""chunk 표시: 렌더 기록 (npz) 에서 스텝별 chunk 상태를 계산하고 3D 경로와 화면 글자·타임라인을 그린다.

chunk 상태 (실행 스텝 t 기준, M = 전환 횟수, S_k = switch_steps[k], d = 지연)
    chunk 0 (첫 chunk) 은 previous_plans[0], chunk k+1 은 new_plans[k] 이고 [S_k, S_{k+1}) 동안 실행된다.
    chunk k+1 의 생성 구간은 [S_k − d, S_k) 이며, 이 동안 이전 chunk 가 계속 실행된다.
3D 표시 (테이블 위 경로)
    실행할 부분: 파랑 / 계획했지만 버리는 뒷부분: 옅은 파랑 / 생성 중 실행되는 확정 구간: 금색
    생성 중 후보 경로 (RS-RTC particle): 옅은 주황 / 전환 직후 이전 chunk 의 남은 계획: 금색 잔상
"""
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATH_HEIGHT = 0.007
EXECUTE_RGBA = (0.15, 0.45, 0.95, 0.95)
DISCARD_RGBA = (0.45, 0.65, 1.0, 0.35)
COMMITTED_RGBA = (1.0, 0.75, 0.10, 0.95)
CANDIDATE_RGBA = (1.0, 0.45, 0.10, 0.18)
TRAIL_RGBA = (0.35, 0.35, 0.38, 0.3)
GHOST_STEPS = 6
TRAIL_STEPS = 30

HUD_TEXT_RGB = (255, 255, 255)
HUD_PANEL_RGBA = (20, 22, 28, 170)
TIMELINE_CHUNK_RGB = ((40, 110, 230), (90, 150, 245))
TIMELINE_GENERATE_RGB = (245, 140, 30)
LEGEND_ITEMS = (("executed part of chunk", (40, 115, 240)), ("planned, then discarded", (130, 170, 255)),
                ("executed during inference", (255, 190, 25)))
CANDIDATE_LEGEND_ITEM = ("candidate chunks (particles)", (255, 120, 30))


class ChunkTimeline:
    """렌더 기록에서 실행 스텝별 활성 chunk, 생성 구간, 후보 경로를 찾는다."""

    def __init__(self, render_record):
        self._delay = int(render_record["delay"])
        self._step_count = len(render_record["executed_actions"])
        self._switch_steps = render_record["switch_steps"].astype(int)
        self._candidate_plans = render_record["candidate_plans"]

        # chunk 별 계획 경로, offset, 실행 구간 [시작, 끝)
        self._plans = [render_record["previous_plans"][0]] + list(render_record["new_plans"])
        self._plan_offsets = [int(render_record["previous_plan_offsets"][0])] + \
            [int(offset) for offset in render_record["new_plan_offsets"]]
        self._chunk_starts = [0] + list(self._switch_steps)
        self._chunk_ends = list(self._switch_steps) + [self._step_count]

    @property
    def delay(self):
        return self._delay

    @property
    def step_count(self):
        return self._step_count

    @property
    def chunk_count(self):
        return len(self._plans)

    @property
    def chunk_starts(self):
        return self._chunk_starts

    @property
    def chunk_ends(self):
        return self._chunk_ends

    @property
    def has_candidates(self):
        return self._candidate_plans.ndim == 4 and self._candidate_plans.shape[1] > 0

    def active_chunk_index(self, step):
        """실행 스텝 step 에 실행 중인 chunk 번호."""
        return int(np.searchsorted(self._switch_steps, step, side="right"))

    def generating_chunk_index(self, step):
        """step 에 생성 중인 chunk 번호와 진행률 (생성 중이 아니면 (None, 0))."""
        next_chunk_index = self.active_chunk_index(step) + 1
        if next_chunk_index >= self.chunk_count or self._delay == 0:
            return None, 0.0
        generation_start = self._chunk_starts[next_chunk_index] - self._delay
        if step < generation_start:
            return None, 0.0
        return next_chunk_index, (step - generation_start) / self._delay

    def plan_steps(self, chunk_index):
        """chunk 계획 index j 에 대응하는 실행 스텝 [H]."""
        return self._plan_offsets[chunk_index] + np.arange(len(self._plans[chunk_index]))

    def plan(self, chunk_index):
        return self._plans[chunk_index]

    def candidates(self, chunk_index):
        """chunk_index (≥ 1) 를 고른 후보 경로 [N, H, 2]."""
        return self._candidate_plans[chunk_index - 1]


class ChunkOverlay:
    """renderer.scene 에 chunk·후보 경로와 지나온 궤적을 3D 도형으로 추가."""

    def __init__(self, push_t_scene, timeline):
        self._scene = push_t_scene
        self._timeline = timeline

    def add_sphere(self, render_scene, position, radius, rgba):
        """구 1 개 추가."""
        if render_scene.ngeom >= render_scene.maxgeom:
            return
        mujoco.mjv_initGeom(render_scene.geoms[render_scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0, 0]), position, np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        render_scene.ngeom += 1

    def add_segment(self, render_scene, start, end, width, rgba):
        """두 점을 잇는 캡슐 1 개 추가."""
        if render_scene.ngeom >= render_scene.maxgeom:
            return
        geom = render_scene.geoms[render_scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width, start, end)
        render_scene.ngeom += 1

    def add_polyline(self, render_scene, positions, width, rgba, point_radius=0.0):
        """점 배열을 선 (와 점) 으로 추가."""
        for start, end in zip(positions[:-1], positions[1:]):
            self.add_segment(render_scene, start, end, width, rgba)
        if point_radius > 0:
            for position in positions:
                self.add_sphere(render_scene, position, point_radius, rgba)

    def add_active_chunk(self, render_scene, step, generating_chunk_index):
        """실행 중인 chunk: 실행할 부분, 버리는 뒷부분, 생성 중 확정 구간을 색으로 구분."""
        timeline = self._timeline
        chunk_index = timeline.active_chunk_index(step)
        plan_positions = self._scene.table_positions(timeline.plan(chunk_index), PATH_HEIGHT)
        plan_steps = timeline.plan_steps(chunk_index)
        chunk_end = timeline.chunk_ends[chunk_index]
        committed_start = chunk_end - timeline.delay if generating_chunk_index is not None else chunk_end

        # 아직 실행하지 않은 계획 index 를 구간별로 나눔
        future_mask = plan_steps >= int(np.floor(step))
        execute_mask = future_mask & (plan_steps < committed_start)
        committed_mask = future_mask & (plan_steps >= committed_start) & (plan_steps < chunk_end)
        discard_mask = future_mask & (plan_steps >= chunk_end)
        for segment_mask, rgba, width, radius in ((discard_mask, DISCARD_RGBA, 0.0022, 0.0045),
                                                  (execute_mask, EXECUTE_RGBA, 0.0038, 0.0070),
                                                  (committed_mask, COMMITTED_RGBA, 0.0040, 0.0075)):
            indices = np.flatnonzero(segment_mask)
            if len(indices) == 0:
                continue
            segment_indices = np.arange(max(indices[0] - 1, 0), indices[-1] + 1)
            self.add_polyline(render_scene, plan_positions[segment_indices], width, rgba)
            for index in indices:
                self.add_sphere(render_scene, plan_positions[index], radius, rgba)

    def add_candidates(self, render_scene, generating_chunk_index, progress):
        """생성 중인 chunk 의 후보 경로를 진행률에 따라 점점 진하게 추가."""
        candidate_alpha = CANDIDATE_RGBA[3] + 0.2 * progress
        candidate_rgba = CANDIDATE_RGBA[:3] + (candidate_alpha,)
        for candidate_plan in self._timeline.candidates(generating_chunk_index):
            self.add_polyline(render_scene, self._scene.table_positions(candidate_plan, PATH_HEIGHT * 0.8), 0.0012,
                              candidate_rgba)

    def add_switch_ghost(self, render_scene, step):
        """전환 직후 이전 chunk 의 남은 계획을 금색 잔상으로 추가 (새 chunk 와 이어지는지 비교)."""
        timeline = self._timeline
        chunk_index = timeline.active_chunk_index(step)
        elapsed_steps = step - timeline.chunk_starts[chunk_index]
        if chunk_index == 0 or elapsed_steps >= GHOST_STEPS:
            return
        previous_index = chunk_index - 1
        previous_steps = timeline.plan_steps(previous_index)
        remaining_mask = previous_steps >= timeline.chunk_starts[chunk_index]
        ghost_alpha = 0.7 * (1.0 - elapsed_steps / GHOST_STEPS)
        ghost_positions = self._scene.table_positions(timeline.plan(previous_index)[remaining_mask], PATH_HEIGHT)
        self.add_polyline(render_scene, ghost_positions, 0.0018, COMMITTED_RGBA[:3] + (ghost_alpha,), 0.003)

    def add_trail(self, render_scene, agent_positions, step):
        """최근 TRAIL_STEPS 스텝 동안 지나온 끝단 궤적."""
        last_index = int(np.floor(step))
        trail_positions = self._scene.table_positions(agent_positions[max(last_index - TRAIL_STEPS, 0):last_index + 1],
                                                      0.003)
        self.add_polyline(render_scene, trail_positions, 0.0012, TRAIL_RGBA)

    def add_to_scene(self, render_scene, agent_positions, step):
        """step 시점의 모든 3D 표시를 추가."""
        generating_chunk_index, progress = self._timeline.generating_chunk_index(int(np.floor(step)))
        self.add_trail(render_scene, agent_positions, step)
        if generating_chunk_index is not None and self._timeline.has_candidates:
            self.add_candidates(render_scene, generating_chunk_index, progress)
        self.add_switch_ghost(render_scene, int(np.floor(step)))
        self.add_active_chunk(render_scene, step, generating_chunk_index)


class HudPainter:
    """렌더 이미지 위에 방법 이름, 상태, 범례, chunk 타임라인을 그린다."""

    def __init__(self, timeline, method_label, success, control_hz, width, height):
        self._timeline = timeline
        self._method_label = method_label
        self._success = success
        self._control_hz = control_hz
        self._width = width
        self._height = height
        self._title_font = ImageFont.load_default(size=max(height // 26, 16))
        self._text_font = ImageFont.load_default(size=max(height // 42, 12))

    def draw_status(self, draw, step):
        """좌상단: 방법, 지연, 스텝, 현재 chunk 상태."""
        timeline = self._timeline
        integer_step = min(int(np.floor(step)), timeline.step_count)
        active_index = timeline.active_chunk_index(integer_step)
        generating_index, progress = timeline.generating_chunk_index(integer_step)
        delay_seconds = timeline.delay / self._control_hz
        if generating_index is None:
            status_text = f"executing chunk {active_index + 1}"
        else:
            status_text = (f"executing chunk {active_index + 1}  |  inferring chunk {generating_index + 1} "
                           f"({int(progress * timeline.delay) + 1}/{timeline.delay})")
        lines = (f"inference delay d = {timeline.delay} steps ({delay_seconds:.1f} s)",
                 f"step {integer_step} / {timeline.step_count}", status_text)

        # 반투명 패널과 글자
        draw.rounded_rectangle((16, 16, 16 + self._width * 0.42, 16 + self._height * 0.2), radius=12,
                               fill=HUD_PANEL_RGBA)
        draw.text((32, 26), self._method_label, font=self._title_font, fill=HUD_TEXT_RGB)
        line_height = self._text_font.size * 1.35
        for line_index, line in enumerate(lines):
            draw.text((32, 30 + self._title_font.size * 1.4 + line_index * line_height), line, font=self._text_font,
                      fill=HUD_TEXT_RGB)

    def draw_legend(self, draw):
        """우상단: 경로 색 범례."""
        legend_items = LEGEND_ITEMS + ((CANDIDATE_LEGEND_ITEM,) if self._timeline.has_candidates else ())
        line_height = self._text_font.size * 1.5
        panel_width = self._width * 0.25
        panel_left = self._width - panel_width - 16
        draw.rounded_rectangle((panel_left, 16, self._width - 16, 30 + line_height * len(legend_items)), radius=12,
                               fill=HUD_PANEL_RGBA)
        for item_index, (label, rgb) in enumerate(legend_items):
            center_y = 24 + line_height * (item_index + 0.5)
            draw.ellipse((panel_left + 14, center_y - 6, panel_left + 26, center_y + 6), fill=rgb)
            draw.text((panel_left + 36, center_y - self._text_font.size * 0.6), label, font=self._text_font,
                      fill=HUD_TEXT_RGB)

    def draw_timeline(self, draw, step):
        """하단: chunk 실행 구간 (파랑 교대), 생성 구간 (주황), 현재 위치."""
        timeline = self._timeline
        bar_left, bar_right = 40, self._width - 40
        bar_top = self._height - 56
        bar_height = 18
        steps_to_x = (bar_right - bar_left) / max(timeline.step_count, 1)
        draw.rounded_rectangle((bar_left - 14, bar_top - 30, bar_right + 14, self._height - 14), radius=10,
                               fill=HUD_PANEL_RGBA)
        draw.text((bar_left, bar_top - 26), "chunks (blue) / inference windows (orange)", font=self._text_font,
                  fill=HUD_TEXT_RGB)

        # chunk 실행 구간과 생성 구간
        for chunk_index in range(timeline.chunk_count):
            chunk_left = bar_left + timeline.chunk_starts[chunk_index] * steps_to_x
            chunk_right = bar_left + timeline.chunk_ends[chunk_index] * steps_to_x
            draw.rectangle((chunk_left, bar_top, chunk_right, bar_top + bar_height),
                           fill=TIMELINE_CHUNK_RGB[chunk_index % 2])
            if chunk_index > 0 and timeline.delay > 0:
                generate_left = chunk_left - timeline.delay * steps_to_x
                draw.rectangle((generate_left, bar_top + bar_height + 3, chunk_left, bar_top + bar_height + 9),
                               fill=TIMELINE_GENERATE_RGB)

        # 현재 위치
        playhead_x = bar_left + min(step, timeline.step_count) * steps_to_x
        draw.rectangle((playhead_x - 2, bar_top - 6, playhead_x + 2, bar_top + bar_height + 12), fill=HUD_TEXT_RGB)

    def draw_result(self, draw):
        """마지막 구간: 에피소드 결과."""
        result_text = "SUCCESS" if self._success else "FAILED (time limit)"
        result_rgb = (60, 200, 110) if self._success else (230, 70, 60)
        text_width = draw.textlength(result_text, font=self._title_font)
        center_x = self._width / 2
        draw.rounded_rectangle((center_x - text_width / 2 - 24, self._height * 0.42 - 12,
                                center_x + text_width / 2 + 24, self._height * 0.42 + self._title_font.size + 12),
                               radius=12, fill=HUD_PANEL_RGBA)
        draw.text((center_x - text_width / 2, self._height * 0.42), result_text, font=self._title_font,
                  fill=result_rgb)

    def draw_inset(self, frame_image, draw, inset_frame):
        """좌하단: 밀대를 따라가는 위에서 본 확대 화면 (로봇팔 숨김) 과 테두리·제목."""
        inset_height, inset_width = inset_frame.shape[:2]
        inset_left = 24
        inset_top = self._height - inset_height - 96
        draw.rounded_rectangle((inset_left - 8, inset_top - 34, inset_left + inset_width + 8,
                                inset_top + inset_height + 8), radius=12, fill=HUD_PANEL_RGBA)
        draw.text((inset_left + 4, inset_top - 30), "zoom on pusher (arm hidden)", font=self._text_font,
                  fill=HUD_TEXT_RGB)
        frame_image.paste(Image.fromarray(inset_frame).convert("RGBA"), (inset_left, inset_top))

    def paint(self, frame, step, show_result, inset_frame=None):
        """렌더 프레임 [H, W, 3] 에 HUD (와 확대 화면) 를 그려 반환."""
        frame_image = Image.fromarray(frame).convert("RGBA")
        hud_layer = Image.new("RGBA", frame_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(hud_layer)
        if inset_frame is not None:
            self.draw_inset(hud_layer, draw, inset_frame)
        self.draw_status(draw, step)
        self.draw_legend(draw)
        self.draw_timeline(draw, step)
        if show_result:
            self.draw_result(draw)
        return np.asarray(Image.alpha_composite(frame_image, hud_layer).convert("RGB"))
