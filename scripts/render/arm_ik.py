"""Franka 역기구학: 밀대 끝점이 목표 위치에 오고 손이 아래를 향하도록 팔 관절 7 개를 맞춘다.

감쇠 최소제곱 (damped least squares) 6 자유도 (위치 3 + 자세 3) 반복 해법.
목표 자세는 home 자세에서의 site 자세 (손이 아래를 향함) 이다.
남는 1 자유도는 home 자세 쪽으로 당기는 null-space 항으로 정한다. 이전 프레임 해를 초기값으로 쓴다.
"""
import mujoco
import numpy as np

ARM_JOINT_NAMES = tuple(f"joint{joint_number}" for joint_number in range(1, 8))


class ArmIkSolver:
    """밀대 끝점 site 의 위치·자세 목표에 대한 팔 관절 역기구학."""

    def __init__(self, model, data, site_name, iterations=40, damping=0.02, nullspace_gain=0.05):
        self._model = model
        self._data = data
        self._site_id = model.site(site_name).id
        self._iterations = iterations
        self._damping = damping
        self._nullspace_gain = nullspace_gain

        # 팔 관절 qpos / dof 주소와 범위
        self._qpos_indices = np.array([model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES])
        self._dof_indices = np.array([model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES])
        self._joint_lower = np.array([model.joint(name).range[0] for name in ARM_JOINT_NAMES])
        self._joint_upper = np.array([model.joint(name).range[1] for name in ARM_JOINT_NAMES])
        self._home_qpos = data.qpos[self._qpos_indices].copy()
        self._max_error = 0.0

        # home 자세의 site 자세 (목표 자세)
        mujoco.mj_forward(model, data)
        self._target_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(self._target_quaternion, data.site_xmat[self._site_id])

    @property
    def max_error(self):
        """지금까지 풀이에서 남은 최대 위치 오차 (m)."""
        return self._max_error

    def site_pose_error(self, target_position, target_quaternion):
        """현재 site 와 목표 사이의 오차 [위치 3, 자세 3]."""
        position_error = target_position - self._data.site_xpos[self._site_id]
        site_quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(site_quaternion, self._data.site_xmat[self._site_id])
        orientation_error = np.zeros(3)
        mujoco.mju_subQuat(orientation_error, target_quaternion, site_quaternion)

        # site 좌표계 회전 오차를 월드 좌표계로 변환 (야코비안과 같은 좌표계)
        site_rotation = self._data.site_xmat[self._site_id].reshape(3, 3)
        return np.concatenate([position_error, site_rotation @ orientation_error])

    def solve(self, target_position):
        """목표 위치와 home site 자세로 팔 관절을 맞추고 data 의 순기구학을 갱신 (남은 위치 오차 m 반환)."""
        target_quaternion = self._target_quaternion
        position_jacobian = np.zeros((3, self._model.nv))
        rotation_jacobian = np.zeros((3, self._model.nv))

        for _ in range(self._iterations):
            # 현재 자세의 오차와 야코비안
            mujoco.mj_kinematics(self._model, self._data)
            mujoco.mj_comPos(self._model, self._data)
            pose_error = self.site_pose_error(target_position, target_quaternion)
            mujoco.mj_jacSite(self._model, self._data, position_jacobian, rotation_jacobian, self._site_id)
            jacobian = np.vstack([position_jacobian, rotation_jacobian])[:, self._dof_indices]

            # 감쇠 최소제곱 + home 자세 null-space 항
            damped_inverse = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + self._damping ** 2 * np.eye(6))
            nullspace_projector = np.eye(len(self._dof_indices)) - damped_inverse @ jacobian
            current_qpos = self._data.qpos[self._qpos_indices]
            joint_step = damped_inverse @ pose_error
            joint_step += nullspace_projector @ (self._nullspace_gain * (self._home_qpos - current_qpos))
            self._data.qpos[self._qpos_indices] = np.clip(current_qpos + joint_step, self._joint_lower,
                                                          self._joint_upper)

        # 최종 순기구학과 오차 기록
        mujoco.mj_forward(self._model, self._data)
        position_error = float(np.linalg.norm(target_position - self._data.site_xpos[self._site_id]))
        self._max_error = max(self._max_error, position_error)
        return position_error
