"""MuJoCo 3D 장면: 테이블 위 PushT 작업 공간, T 블록, 목표 T, 밀대를 잡은 Franka 로봇팔.

PushT 픽셀 좌표 (0~512, y 아래 방향) 를 테이블 좌표 (m, y 위 방향) 로 옮긴다.
    X = s (x − 256),  Y = −s (y − 256),  블록 각도 θ → z 축 회전 −θ   (s = meters_per_pixel)
물리 시뮬레이션 없이 기록된 상태를 그대로 배치하는 재생용 장면이다 (블록은 mocap, 팔은 역기구학).
"""
import math

import mujoco
import numpy as np

PUSHT_CENTER_PX = 256.0
PUSHT_WALL_MIN_PX = 5.0
PUSHT_WALL_MAX_PX = 506.0
PUSHER_RADIUS_PX = 15.0

# T 블록 (gym-pusht add_tee, scale 30, length 4): (중심 x, 중심 y, 반폭, 반높이) 픽셀, 블록 좌표계
TEE_BOXES_PX = ((0.0, 15.0, 60.0, 15.0), (0.0, 75.0, 15.0, 45.0))

TABLE_TOP_HALF_SIZE = (0.75, 0.6)
TABLE_THICKNESS = 0.04
PUSHER_ROD_LENGTH = 0.16
FINGER_OPEN_WIDTH = 0.012

BLOCK_RGBA = (0.36, 0.42, 0.52, 1.0)
GOAL_RGBA = (0.30, 0.78, 0.45, 0.55)
MAT_RGBA = (0.93, 0.93, 0.91, 1.0)
WALL_RGBA = (0.25, 0.27, 0.30, 1.0)
PUSHER_RGBA = (0.85, 0.25, 0.20, 1.0)


def yaw_quaternion(yaw):
    """z 축 회전 yaw (rad) 의 쿼터니언 [w, x, y, z]."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


class PushTScene:
    """Franka 모델에 테이블·작업 공간·T 블록·목표·밀대를 더한 장면과 상태 배치."""

    def __init__(self, panda_xml_path, scene_config, goal_pose_px, width, height):
        self._meters_per_pixel = scene_config["meters_per_pixel"]
        self._block_height = scene_config["block_height_m"]
        self._pusher_tip_height = scene_config["pusher_tip_height_m"]
        self._robot_base_xy = scene_config["robot_base_xy"]

        # 장면 조립 후 컴파일
        model_spec = mujoco.MjSpec.from_file(str(panda_xml_path))
        self.configure_visual(model_spec, width, height)
        self.add_environment(model_spec)
        self.add_workspace(model_spec, goal_pose_px)
        self.add_pusher(model_spec)
        model_spec.body("link0").pos = [self._robot_base_xy[0], self._robot_base_xy[1], 0.0]
        self._model = model_spec.compile()
        self._data = mujoco.MjData(self._model)

        # 초기 자세 (home keyframe, 손가락은 밀대를 잡는 폭)
        mujoco.mj_resetDataKeyframe(self._model, self._data, self._model.key("home").id)
        for finger_joint_name in ("finger_joint1", "finger_joint2"):
            self._data.joint(finger_joint_name).qpos = FINGER_OPEN_WIDTH
        self._block_mocap_index = self._model.body("tee_block").mocapid[0]
        mujoco.mj_forward(self._model, self._data)

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    @property
    def pusher_tip_height(self):
        return self._pusher_tip_height

    def configure_visual(self, model_spec, width, height):
        """오프스크린 해상도, 그림자, 안개, 조명 품질."""
        model_spec.visual.global_.offwidth = width
        model_spec.visual.global_.offheight = height
        model_spec.visual.quality.shadowsize = 4096
        model_spec.visual.quality.offsamples = 8
        model_spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
        model_spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
        model_spec.visual.headlight.specular = [0.05, 0.05, 0.05]

    def add_environment(self, model_spec):
        """하늘 그라데이션, 바닥, 나무색 테이블, 그림자 조명."""
        # 텍스처와 재질
        model_spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.82, 0.86, 0.92],
                               rgb2=[0.35, 0.40, 0.48], width=512, height=3072)
        model_spec.add_texture(name="floor_texture", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.55, 0.56, 0.58],
                               rgb2=[0.50, 0.51, 0.53], width=512, height=512)
        floor_material = model_spec.add_material(name="floor_material", reflectance=0.05, texrepeat=[6, 6])
        floor_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_texture"
        model_spec.add_material(name="table_material", rgba=[0.62, 0.47, 0.33, 1.0], specular=0.2, shininess=0.3,
                                reflectance=0.08)

        # 바닥과 테이블 (상판 윗면 z = 0)
        worldbody = model_spec.worldbody
        worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], pos=[0, 0, -0.75],
                           material="floor_material", contype=0, conaffinity=0)
        worldbody.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=[TABLE_TOP_HALF_SIZE[0], TABLE_TOP_HALF_SIZE[1], TABLE_THICKNESS / 2],
                           pos=[-0.1, 0, -TABLE_THICKNESS / 2], material="table_material", contype=0,
                           conaffinity=0)
        for leg_x in (-0.75, 0.55):
            for leg_y in (-0.5, 0.5):
                worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.355],
                                   pos=[leg_x, leg_y, -0.395], material="table_material", contype=0,
                                   conaffinity=0)

        # 조명
        worldbody.add_light(name="key_light", pos=[1.6, 0.4, 1.8], dir=[-1.0, -0.25, -1.1], castshadow=True,
                            diffuse=[0.75, 0.75, 0.72], specular=[0.2, 0.2, 0.2])
        worldbody.add_light(name="fill_light", pos=[-1.0, 1.0, 1.5], dir=[0.5, -0.5, -1.0], castshadow=False,
                            diffuse=[0.25, 0.25, 0.28])

    def add_workspace(self, model_spec, goal_pose_px):
        """작업 매트, 벽 테두리, 목표 T (반투명), T 블록 (mocap)."""
        worldbody = model_spec.worldbody
        meters_per_pixel = self._meters_per_pixel
        half_extent = (PUSHT_WALL_MAX_PX - PUSHT_WALL_MIN_PX) / 2 * meters_per_pixel

        # 작업 매트와 벽
        worldbody.add_geom(name="mat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half_extent, half_extent, 0.001],
                           pos=[0, 0, 0.001], rgba=MAT_RGBA, contype=0, conaffinity=0)
        wall_half_thickness = 0.006
        for wall_center, wall_size in (([0, half_extent, 0.008], [half_extent, wall_half_thickness, 0.008]),
                                       ([0, -half_extent, 0.008], [half_extent, wall_half_thickness, 0.008]),
                                       ([half_extent, 0, 0.008], [wall_half_thickness, half_extent, 0.008]),
                                       ([-half_extent, 0, 0.008], [wall_half_thickness, half_extent, 0.008])):
            worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=wall_size, pos=wall_center, rgba=WALL_RGBA,
                               contype=0, conaffinity=0)

        # 목표 T (얇은 반투명 판)
        goal_body = worldbody.add_body(name="goal_tee", pos=self.table_position(goal_pose_px[:2], 0.0025),
                                       quat=yaw_quaternion(-float(goal_pose_px[2])))
        self.add_tee_geoms(goal_body, 0.0008, GOAL_RGBA)

        # T 블록 (기록된 자세로 직접 배치)
        block_body = worldbody.add_body(name="tee_block", mocap=True)
        self.add_tee_geoms(block_body, self._block_height / 2, BLOCK_RGBA)

    def add_tee_geoms(self, parent_body, half_height, rgba):
        """T 자 상자 2개를 블록 좌표계 기준으로 추가 (윗면이 half_height·2 높이)."""
        meters_per_pixel = self._meters_per_pixel
        for center_x_px, center_y_px, half_width_px, half_length_px in TEE_BOXES_PX:
            parent_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                                 size=[half_width_px * meters_per_pixel, half_length_px * meters_per_pixel,
                                       half_height],
                                 pos=[center_x_px * meters_per_pixel, -center_y_px * meters_per_pixel, half_height],
                                 rgba=rgba, contype=0, conaffinity=0)

    def add_pusher(self, model_spec):
        """손가락 사이에 수직 밀대와 끝점 site 를 추가."""
        hand_body = model_spec.body("hand")
        rod_radius = PUSHER_RADIUS_PX * self._meters_per_pixel
        rod_bottom_z = 0.11 + PUSHER_ROD_LENGTH
        hand_body.add_geom(name="pusher_rod", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           fromto=[0, 0, 0.06, 0, 0, rod_bottom_z - rod_radius * 0.3], size=[rod_radius * 0.55, 0, 0],
                           rgba=[0.75, 0.76, 0.78, 1.0], contype=0, conaffinity=0)
        hand_body.add_geom(name="pusher_tip", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           fromto=[0, 0, rod_bottom_z - 0.025, 0, 0, rod_bottom_z], size=[rod_radius, 0, 0],
                           rgba=PUSHER_RGBA, contype=0, conaffinity=0)
        hand_body.add_site(name="pusher_tip_site", pos=[0, 0, rod_bottom_z], size=[0.004, 0, 0])

    def table_position(self, position_px, height):
        """PushT 픽셀 좌표 [x, y] → 테이블 좌표 [X, Y, height] (m)."""
        meters_per_pixel = self._meters_per_pixel
        return np.array([(position_px[0] - PUSHT_CENTER_PX) * meters_per_pixel,
                         -(position_px[1] - PUSHT_CENTER_PX) * meters_per_pixel, height])

    def table_positions(self, positions_px, height):
        """PushT 픽셀 좌표 배열 [..., 2] → 테이블 좌표 [..., 3] (m)."""
        positions_px = np.asarray(positions_px, dtype=np.float64)
        table_xy = np.stack([positions_px[..., 0] - PUSHT_CENTER_PX, -(positions_px[..., 1] - PUSHT_CENTER_PX)],
                            axis=-1) * self._meters_per_pixel
        return np.concatenate([table_xy, np.full(table_xy.shape[:-1] + (1,), height)], axis=-1)

    def set_block_pose(self, block_pose_px):
        """T 블록 mocap 을 PushT 자세 [x, y, θ] 로 배치."""
        self._data.mocap_pos[self._block_mocap_index] = self.table_position(block_pose_px[:2], 0.0)
        self._data.mocap_quat[self._block_mocap_index] = yaw_quaternion(-float(block_pose_px[2]))
