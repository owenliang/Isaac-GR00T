from typing import Generic, List, Optional, Sequence, TypeVar, Union

from gr00t.data.state_action.pose import EndEffectorPose, JointPose, Pose
from gr00t.data.types import ActionFormat
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate
from scipy.spatial.transform import Rotation, Slerp


PoseType = TypeVar("PoseType", bound=Pose)


class ActionChunk(Generic[PoseType]):
    """
    Abstract base class for robot action chunking.

    This class provides common functionality for different action chunking types
    including relative and delta action chunking computation with optional reference frames,
    interpolation, and format conversion.

    【中文】机器人动作片段（trajectory chunk）的抽象基类：
    【中文】- 以一串 Pose（关节或末端位姿）描述一个动作片段；
    【中文】- 提供公共的相对动作/增量动作计算（相对于参考帧或前一帧）；
    【中文】- 支持按时间戳插值、以及将内部表示转换为不同的动作数值格式；
    【中文】具体的关节空间 / 末端空间动作片段由子类 JointActionChunk 和 EndEffectorActionChunk 实现。
    """

    def __init__(
        self,
        poses: Sequence[PoseType],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize action chunking from a list of poses.

        Args:
            poses: Sequence of Pose objects
            times: Optional sequence of timestamps for each pose. If None, assumes
                   uniform spacing starting from 0 with step 1.0

        Raises:
            ValueError: If action chunking is empty or times length doesn't match poses

        【中文】从一串位姿构造动作片段：
        【中文】- poses: 一系列按时间排序的 Pose（JointPose 或 EndEffectorPose）；
        【中文】- times: 对应的时间戳序列，若省略则默认从 0 开始、步长为 1.0 的均匀时间轴；
        【中文】- 若 poses 为空或 times 长度与 poses 不匹配，会抛出 ValueError；
        【中文】时间戳用于插值和对齐，不影响相对/增量运算的几何意义。
        """
        if not poses:
            raise ValueError("ActionChunk must contain at least one pose")

        self._poses: List[PoseType] = list(poses)

        # Set up times
        if times is None:
            self._times = np.arange(len(poses), dtype=np.float64)
        else:
            if len(times) != len(poses):
                raise ValueError("Number of times must match number of poses")
            self._times = np.array(times, dtype=np.float64)

    @property
    def poses(self) -> List[PoseType]:
        """Get the list of poses.

        【中文】返回动作片段内部保存的位姿列表：
        【中文】- 返回值是内部列表的浅拷贝，避免外部直接修改内部状态；
        【中文】- 每个元素都是 JointPose 或 EndEffectorPose，取决于具体子类。
        """
        return self._poses.copy()

    @property
    def times(self) -> NDArray[np.float64]:
        """Get the timestamps.

        【中文】返回与每个位姿对应的时间戳数组：
        【中文】- 通常用于插值或与真实时间对齐；
        【中文】- 同样返回的是拷贝，防止外部修改内部缓存。
        """
        return self._times.copy()

    @property
    def num_poses(self) -> int:
        """Get the number of poses in the action chunking.

        【中文】返回当前动作片段包含的位姿数量，即时间步骤数 N。
        """
        return len(self._poses)

    def relative_chunking(
        self, reference_frame: Optional[PoseType] = None
    ) -> "ActionChunk[PoseType]":
        """
        Compute the relative action chunking with respect to a reference frame.

        If reference_frame is None, uses the first pose in the action chunking as reference.
        All poses are transformed to be relative to the reference frame.

        Args:
            reference_frame: Optional reference pose. If None, uses first pose.

        Returns:
            A new ActionChunk of the same type where all poses are relative to the reference frame.

        【中文】计算“相对于某个参考位姿”的相对动作轨迹：
        【中文】- 若未提供 reference_frame，则默认以轨迹中第一帧为参考；
        【中文】- 对每个 pose 计算 `pose - ref_pose`，得到一条以参考帧为原点的相对轨迹；
        【中文】- 对 JointActionChunk 来说是关节差值，对 EndEffectorActionChunk 则是相对位姿变换；
        【中文】返回新的同类型 ActionChunk，不修改原始轨迹。
        """
        if not self._poses:
            return self.__class__([], times=[])

        # Use the first pose as the reference if one is not provided.
        ref_pose = reference_frame if reference_frame is not None else self._poses[0]

        # Use the polymorphic subtraction defined in the Pose subclasses.
        # The subtraction returns the same type as the operands
        relative_poses: List[PoseType] = [pose - ref_pose for pose in self._poses]  # type: ignore[misc]

        # Return a new instance of the same action chunking class
        # (e.g., JointActionChunk or EndEffectorActionChunk)
        return self.__class__(relative_poses, times=self.times)

    def delta_chunking(self, reference_frame: Optional[PoseType] = None) -> "ActionChunk[PoseType]":
        """
        Compute the delta action chunking where each pose represents the relative
        transformation from the previous frame.

        If reference_frame is provided, it is treated as the first frame, and the
        first delta will be from reference_frame to the first pose in the action chunking.
        Otherwise, the first pose in the delta action chunking will be the identity/zero transformation.

        Args:
            reference_frame: Optional reference pose to use as the first frame.

        Returns:
            A new ActionChunk of the same type where each pose is relative to the previous pose.

        【中文】计算“逐帧增量”形式的动作轨迹（delta 轨迹）：
        【中文】- 若提供 reference_frame，则第一帧增量为 pose[0] - reference_frame；
        【中文】- 否则第一帧增量为 pose[0] - pose[0]，得到零位姿，其后每一帧为当前帧相对上一帧的差值；
        【中文】- 常用于将绝对轨迹变为“速度/增量”形式，便于学习局部变化或做数值稳定处理。
        """
        if not self._poses:
            return self.__class__([], times=[])

        delta_poses: List[PoseType] = []

        # Determine the initial reference for the very first pose.
        # If a reference_frame is given, the first delta is pose[0] - reference_frame.
        # If not, the first delta is pose[0] - pose[0], resulting in an identity/zero pose.
        prev_pose = reference_frame if reference_frame is not None else self._poses[0]

        for current_pose in self._poses:
            delta: PoseType = current_pose - prev_pose  # type: ignore[assignment]
            delta_poses.append(delta)
            prev_pose = current_pose  # Update the reference for the next step

        return self.__class__(delta_poses, times=self.times.tolist())

    def to_absolute_chunking(self, reference_frame: PoseType) -> "ActionChunk[PoseType]":
        """
        Convert a relative action chunking to an absolute action chunking by applying
        the relative poses on top of a reference frame.

        This is the inverse operation of relative_chunking(). Each relative pose
        is composed with the reference frame to produce absolute poses.

        Args:
            reference_frame: The reference pose to apply the relative action chunking on top of.

        Returns:
            A new ActionChunk of the same type with absolute poses.

        Raises:
            NotImplementedError: Must be implemented by subclasses

        【中文】将“相对轨迹”还原为“绝对轨迹”的逆操作：
        【中文】- 以 reference_frame 作为基准，将每一帧的相对位姿叠加回去，得到绝对位姿；
        【中文】- 具体叠加方式（关节相加 / 齐次矩阵相乘）由子类实现；
        【中文】通常与 `relative_chunking` 成对使用，用于在相对/绝对表示之间来回切换。
        """
        raise NotImplementedError("Subclasses must implement to_absolute_chunking")

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "ActionChunk":
        """
        Interpolate the action chunking to generate intermediate poses.
        Must be implemented by subclasses.

        Args:
            num_points: Number of evenly-spaced points to generate
            times: Specific timestamps at which to interpolate

        Returns:
            A new ActionChunk with interpolated poses

        【中文】对动作轨迹做插值，生成中间位姿：
        【中文】- 子类可选择不同的插值策略：关节空间用线性插值，末端位姿用线性 + SLERP 等；
        【中文】- 可通过 num_points 指定需要的均匀采样点数，或直接给定特定时间戳 times；
        【中文】插值结果仍然是同类型的 ActionChunk，不会修改原始轨迹。
        """
        raise NotImplementedError("Subclasses must implement interpolate")

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert action chunking to the specified action format.
        Must be implemented by subclasses.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            NotImplementedError: If not implemented by subclass

        【中文】将内部的 Pose 轨迹转换为指定的数值动作格式：
        【中文】- 对关节轨迹通常导出为 (N, num_joints) 的关节角数组；
        【中文】- 对末端轨迹可以导出为齐次矩阵、xyz+rot6d、xyz+rotvec 等多种形式；
        【中文】具体支持的格式和转换逻辑由各个子类实现。
        """
        raise NotImplementedError("Subclasses must implement to method")

    def __len__(self) -> int:
        """Return the number of poses in the action chunking.

        【中文】返回轨迹长度（位姿数量）。
        """
        return len(self._poses)

    def __getitem__(self, index: int) -> PoseType:
        """Get a pose by index.

        【中文】按索引访问第 index 个位姿（支持切片访问单个 Pose）。
        """
        return self._poses[index]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_poses={len(self._poses)}, "
            f"time_range=[{self._times[0]:.2f}, {self._times[-1]:.2f}])"
        )


class JointActionChunk(ActionChunk[JointPose]):
    """
    Represents action chunking in joint space as a sequence of joint configurations.

    Examples:
        # Create a joint action chunking
        joint_poses = [
            JointPose([0.0, 0.0, 0.0]),
            JointPose([0.5, 0.5, 0.5]),
            JointPose([1.0, 1.0, 1.0]),
        ]
        action_chunking = JointActionChunk(joint_poses)

        # Get relative trajectory (all poses relative to first pose)
        relative_traj = action_chunking.relative_chunking()

        # Get relative trajectory with custom reference
        reference = JointPose([0.1, 0.1, 0.1])
        relative_traj = action_chunking.relative_chunking(reference_frame=reference)

        # Get delta trajectory (incremental changes)
        delta_traj = action_chunking.delta_chunking()

        # Convert relative trajectory back to absolute
        reference = JointPose([0.1, 0.1, 0.1])
        absolute_traj = relative_traj.to_absolute_chunking(reference_frame=reference)

        # Interpolate trajectory
        interpolated = action_chunking.interpolate(num_points=10)

        # Convert to desired format
        from gr00t.data.types import ActionFormat
        array_data = action_chunking.to(ActionFormat.DEFAULT)  # Returns joint array

    【中文】关节空间的动作片段表示：
    【中文】- 内部保存一串 JointPose，描述机器人在关节空间的运动轨迹；
    【中文】- 提供相对/增量轨迹、插值、以及导出为 (N, num_joints) 数组的能力；
    【中文】适合直接驱动关节级控制器，或作为网络关节动作的输出/标签。
    """

    def __init__(
        self,
        poses: Sequence[JointPose],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize a joint trajectory from a list of joint poses.

        Args:
            poses: Sequence of JointPose objects
            times: Optional sequence of timestamps for each pose

        Raises:
            TypeError: If poses are not all JointPose objects

        【中文】从一组 JointPose 构造关节轨迹：
        【中文】- 会检查所有元素类型是否都是 JointPose，若出现其他类型会抛出 TypeError；
        【中文】- 可选的 times 参数用于指定时间戳，不提供则使用等间隔时间；
        【中文】通常由 StateActionProcessor 的关节动作分支调用。
        """
        # Validate all poses are JointPose
        if not all(isinstance(p, JointPose) for p in poses):
            raise TypeError("All poses must be JointPose objects for JointActionChunk")

        super().__init__(poses, times)

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "JointActionChunk":
        """
        Interpolate the joint action chunking to generate intermediate configurations.

        Uses linear interpolation for joint values.

        Args:
            num_points: Number of evenly-spaced points to generate (including endpoints).
                       Only used if times is None.
            times: Specific timestamps at which to interpolate. If provided,
                  num_points is ignored.

        Returns:
            A new JointActionChunk with interpolated poses

        Raises:
            ValueError: If neither num_points nor times is provided, or if
                       interpolation times are outside the trajectory range

        【中文】对关节轨迹做线性插值，生成中间关节配置：
        【中文】- 关节值按时间轴做逐维线性插值；
        【中文】- 若存在非单调递增的时间戳，会打印并丢弃对应点，保证插值的时间单调性；
        【中文】- 可指定总采样点数 num_points，或显式给出插值时间 times，超出时间范围会报错。
        """
        if num_points is None and times is None:
            raise ValueError("Must provide either num_points or times")

        if len(self._poses) < 2:
            raise ValueError("Need at least 2 poses for interpolation")

        # Prepare data: extract joint values
        timestamps = self._times.copy()
        joint_values = np.array([pose.joints for pose in self._poses])  # (N, num_joints)

        # Find and remove non-monotonic timestamps
        drop_indices = [
            idx for idx in range(1, len(timestamps)) if timestamps[idx] <= timestamps[idx - 1]
        ]

        if drop_indices:
            for idx in drop_indices:
                print(
                    f"Dropping timestamp pair - Previous: {timestamps[idx - 1]}, "
                    f"Current: {timestamps[idx]} at index {idx}"
                )
            timestamps = np.delete(timestamps, drop_indices)
            joint_values = np.delete(joint_values, drop_indices, axis=0)

        # Check if we still have enough poses after cleanup
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 poses with monotonic timestamps for interpolation")

        # Create interpolator
        joint_interp = interpolate.interp1d(timestamps, joint_values, kind="linear", axis=0)

        # Generate interpolation times if not provided
        if times is None:
            assert num_points is not None  # Type narrowing for type checker
            interp_times = np.linspace(timestamps[0], timestamps[-1], num_points)
        else:
            interp_times = np.array(times, dtype=np.float64)

        # Check that interpolation times are within bounds
        if np.any(interp_times < timestamps[0]) or np.any(interp_times > timestamps[-1]):
            raise ValueError(
                f"Interpolation times must be within [{timestamps[0]}, {timestamps[-1]}]"
            )

        # Interpolate joint values
        interp_joint_values = joint_interp(interp_times)

        # Create interpolated poses
        joint_names = self._poses[0].joint_names
        interpolated_poses = [
            JointPose(joints=interp_joint_values[i], joint_names=joint_names)
            for i in range(len(interp_times))
        ]

        return JointActionChunk(interpolated_poses, times=interp_times)

    def to_array(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of joint values.

        Returns:
            Array with shape (N, num_joints) where N is the number of poses

        【中文】将关节轨迹导出为纯数值数组：
        【中文】- 返回形状 (N, num_joints)，其中 N 为时间步数；
        【中文】- 每一行对应一个时间步的所有关节角，可直接用于模型输入或保存到文件。
        """
        return np.array([pose.joints for pose in self._poses])

    def to_absolute_chunking(self, reference_frame: JointPose) -> "JointActionChunk":
        """
        Convert a relative joint action chunking to an absolute action chunking by adding
        the relative joint positions to the reference frame.

        This is the inverse operation of relative_chunking(). Each relative joint
        configuration is added to the reference frame to produce absolute joint positions.

        Args:
            reference_frame: The reference joint pose to apply the relative trajectory on top of.

        Returns:
            A new JointActionChunk with absolute joint positions.

        Raises:
            ValueError: If joint dimensions don't match

        【中文】将“相对关节轨迹”还原为“绝对关节轨迹”：
        【中文】- 要求相对轨迹与参考帧的关节维度一致，否则抛出 ValueError；
        【中文】- 对每一帧执行 `absolute = reference_frame.joints + relative_pose.joints`；
        【中文】常用于把模型预测的相对关节增量叠加回某个已知姿态上。
        """
        if not self._poses:
            return JointActionChunk([], times=[])

        if len(self._poses[0].joints) != len(reference_frame.joints):
            raise ValueError(
                f"Cannot apply relative trajectory: "
                f"joint dimensions don't match ({len(self._poses[0].joints)} vs {len(reference_frame.joints)})"
            )

        # Add each relative pose to the reference frame
        absolute_poses: List[JointPose] = []
        for relative_pose in self._poses:
            absolute_joints = reference_frame.joints + relative_pose.joints
            absolute_pose = JointPose(
                joints=absolute_joints, joint_names=reference_frame.joint_names
            )
            absolute_poses.append(absolute_pose)

        return JointActionChunk(absolute_poses, times=self.times)

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert trajectory to the desired format.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            ValueError: If the action format is not supported for joint trajectories

        【中文】将关节轨迹转换为指定的动作格式：
        【中文】- 当前仅支持 ActionFormat.DEFAULT，对应 `to_array()` 输出的关节角数组；
        【中文】- 若请求其他格式，会抛出 ValueError，提醒调用方不支持该格式。
        """
        if action_format == ActionFormat.DEFAULT:
            return self.to_array()
        else:
            raise ValueError(
                f"ActionFormat {action_format} is not supported for JointActionChunk. "
                f"Only {ActionFormat.DEFAULT} is supported."
            )


class EndEffectorActionChunk(ActionChunk[EndEffectorPose]):
    """
    Represents action chunking in Cartesian space as a sequence of end-effector poses.

    Examples:
        # Create an end-effector action chunking
        ee_poses = [
            EndEffectorPose(translation=[0, 0, 0], rotation=[1, 0, 0, 0],
                          rotation_type="quat", rotation_order="wxyz"),
            EndEffectorPose(translation=[1, 0, 0], rotation=[0.707, 0, 0, 0.707],
                          rotation_type="quat", rotation_order="wxyz"),
            EndEffectorPose(translation=[2, 0, 0], rotation=[0, 0, 0, 1],
                          rotation_type="quat", rotation_order="wxyz"),
        ]
        action_chunking = EndEffectorActionChunk(ee_poses)

        # Get relative trajectory (all poses relative to first pose)
        relative_traj = action_chunking.relative_chunking()

        # Get relative trajectory with custom reference frame
        reference = EndEffectorPose(translation=[0.5, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
        relative_traj = action_chunking.relative_chunking(reference_frame=reference)

        # Get delta trajectory
        delta_traj = action_chunking.delta_chunking()

        # Convert relative trajectory back to absolute
        reference = EndEffectorPose(translation=[0.5, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
        absolute_traj = relative_traj.to_absolute_chunking(reference_frame=reference)

        # Interpolate trajectory
        interpolated = action_chunking.interpolate(num_points=10)

        # Convert to desired format
        from gr00t.data.types import ActionFormat
        homo_matrices = action_chunking.to(ActionFormat.DEFAULT)      # (N, 4, 4) homogeneous matrices
        xyz_rot6d = action_chunking.to(ActionFormat.XYZ_ROT6D)        # (N, 9) xyz + rot6d
        xyz_rotvec = action_chunking.to(ActionFormat.XYZ_ROTVEC)      # (N, 6) xyz + rotvec

    【中文】末端执行器在笛卡尔空间的动作片段表示：
    【中文】- 内部保存一串 EndEffectorPose，描述手/工具在空间中的轨迹；
    【中文】- 相对/增量轨迹都在 SE(3) 上计算，支持插值和平滑；
    【中文】- 可导出为齐次矩阵或 xyz + 旋转（rot6d/rotvec）等多种格式，方便网络输入或控制器使用。
    """

    def __init__(
        self,
        poses: Sequence[EndEffectorPose],
        times: Optional[Union[Sequence[float], NDArray[np.float64]]] = None,
    ):
        """
        Initialize an end-effector trajectory from a list of end-effector poses.

        Args:
            poses: Sequence of EndEffectorPose objects
            times: Optional sequence of timestamps for each pose

        Raises:
            TypeError: If poses are not all EndEffectorPose objects

        【中文】从一组 EndEffectorPose 构造末端轨迹：
        【中文】- 检查所有元素是否都是 EndEffectorPose，否则抛出 TypeError；
        【中文】- 可选的 times 用于时间轴定义，用于插值与对齐，不会改变几何轨迹本身。
        """
        # Validate all poses are EndEffectorPose
        if not all(isinstance(p, EndEffectorPose) for p in poses):
            raise TypeError("All poses must be EndEffectorPose objects for EndEffectorActionChunk")

        super().__init__(poses, times)

    def interpolate(
        self, num_points: Optional[int] = None, times: Optional[NDArray[np.float64]] = None
    ) -> "EndEffectorActionChunk":
        """
        Interpolate the action chunking to generate intermediate poses.

        Uses linear interpolation for translation and SLERP (Spherical Linear
        Interpolation) for rotation.

        Args:
            num_points: Number of evenly-spaced points to generate (including endpoints).
                       Only used if times is None.
            times: Specific timestamps at which to interpolate. If provided,
                  num_points is ignored.

        Returns:
            A new EndEffectorActionChunk with interpolated poses

        Raises:
            ValueError: If neither num_points nor times is provided, or if
                       interpolation times are outside the trajectory range

        【中文】对末端轨迹做插值：
        【中文】- 平移部分使用线性插值，保证轨迹在空间中平滑移动；
        【中文】- 旋转部分使用四元数 SLERP 球面线性插值，避免欧拉角插值带来的奇异和非匀速；
        【中文】- 若时间戳不单调，会先剔除不合法的时间点，再进行插值，确保数值稳定。
        """
        if num_points is None and times is None:
            raise ValueError("Must provide either num_points or times")

        if len(self._poses) < 2:
            raise ValueError("Need at least 2 poses for interpolation")

        # Prepare data: extract positions and rotations
        timestamps = self._times.copy()
        homogeneous_matrices = np.array([pose.homogeneous for pose in self._poses])
        positions = homogeneous_matrices[:, :3, 3]
        rotations = Rotation.from_matrix(homogeneous_matrices[:, :3, :3])

        # Find indices where timestamps are not monotonically increasing
        drop_indices = [
            idx for idx in range(1, len(timestamps)) if timestamps[idx] <= timestamps[idx - 1]
        ]

        # Remove the problematic timestamps and corresponding data
        if drop_indices:
            for idx in drop_indices:
                print(
                    f"Dropping timestamp pair - Previous: {timestamps[idx - 1]}, "
                    f"Current: {timestamps[idx]} at index {idx}"
                )
            timestamps = np.delete(timestamps, drop_indices)
            positions = np.delete(positions, drop_indices, axis=0)
            rotations = Rotation.from_matrix(
                np.delete(homogeneous_matrices[:, :3, :3], drop_indices, axis=0)
            )

        # Check if we still have enough poses after cleanup
        if len(timestamps) < 2:
            raise ValueError("Need at least 2 poses with monotonic timestamps for interpolation")

        # Create interpolators
        pos_interp = interpolate.interp1d(timestamps, positions, kind="linear", axis=0)
        rot_interp = Slerp(timestamps, rotations)

        # Generate interpolation times if not provided
        if times is None:
            assert num_points is not None  # Type narrowing for type checker
            interp_times = np.linspace(timestamps[0], timestamps[-1], num_points)
        else:
            interp_times = np.array(times, dtype=np.float64)

        # Check that interpolation times are within bounds
        if np.any(interp_times < timestamps[0]) or np.any(interp_times > timestamps[-1]):
            raise ValueError(
                f"Interpolation times must be within [{timestamps[0]}, {timestamps[-1]}]"
            )

        # Interpolate positions and rotations
        interp_positions = pos_interp(interp_times)
        interp_rotations = rot_interp(interp_times)

        # Create interpolated poses
        interpolated_poses = []
        for i in range(len(interp_times)):
            pose = EndEffectorPose(
                translation=interp_positions[i],
                rotation=interp_rotations[i].as_matrix(),
                rotation_type="matrix",
            )
            interpolated_poses.append(pose)

        return EndEffectorActionChunk(interpolated_poses, times=interp_times)

    def to_homogeneous_matrices(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of homogeneous transformation matrices.

        Returns:
            Array of homogeneous matrices with shape (N, 4, 4) where N is the number of poses

        【中文】将末端轨迹导出为一组 4x4 齐次变换矩阵：
        【中文】- 形状为 (N, 4, 4)，每一帧都是 SE(3) 中的一个变换；
        【中文】- 适合直接用于几何运算或下游控制模块。
        """
        return np.array([pose.homogeneous for pose in self._poses])

    def to_translation_rot6d(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of translations and 6D rotations.

        Returns:
            Array with shape (N, 9) - 3 for xyz + 6 for rot6d

        【中文】将末端轨迹导出为 xyz + rot6d 格式：
        【中文】- 每一帧得到长度为 9 的向量：[x, y, z, rot6d...]；
        【中文】- rot6d 来自内部 Pose 的 6D 旋转表示，适合作为模型输入或动作表示。
        """
        translations = np.array([pose.translation for pose in self._poses])  # (N, 3)
        rotations = np.array([pose.rot6d for pose in self._poses])  # (N, 6)

        # Concatenate translation and rotation
        xyz_rot6d = np.concatenate([translations, rotations], axis=1)  # (N, 9)

        return xyz_rot6d

    def to_translation_rotvec(self) -> NDArray[np.float64]:
        """
        Convert trajectory to array of translations and rotation vectors.

        Returns:
            Array with shape (N, 6) - 3 for xyz + 3 for rotvec

        【中文】将末端轨迹导出为 xyz + 旋转向量 格式：
        【中文】- 每一帧得到长度为 6 的向量：[x, y, z, rx, ry, rz]；
        【中文】- 适合与以轴-角形式建模旋转的算法或网络对接。
        """
        translations = np.array([pose.translation for pose in self._poses])  # (N, 3)
        rotations = np.array([pose.rotvec for pose in self._poses])  # (N, 3)

        # Concatenate translation and rotation
        xyz_rotvec = np.concatenate([translations, rotations], axis=1)  # (N, 6)

        return xyz_rotvec

    def to_absolute_chunking(self, reference_frame: EndEffectorPose) -> "EndEffectorActionChunk":
        """
        Convert a relative end-effector action chunking to an absolute action chunking by
        composing each relative transformation with the reference frame.

        This is the inverse operation of relative_chunking(). Each relative pose
        represents a transformation that is applied on top of the reference frame
        to produce absolute poses.

        Args:
            reference_frame: The reference end-effector pose to apply the relative trajectory on top of.

        Returns:
            A new EndEffectorActionChunk with absolute poses.

        【中文】将“相对末端轨迹”还原为“绝对末端轨迹”：
        【中文】- 先取参考帧的齐次矩阵 T_ref；
        【中文】- 对每一帧相对位姿 T_relative 执行 T_absolute = T_ref @ T_relative；
        【中文】- 将组合后的绝对变换重新封装为 EndEffectorPose 序列返回。
        """
        if not self._poses:
            return EndEffectorActionChunk([], times=[])

        # Get reference frame as homogeneous matrix
        T_ref = reference_frame.homogeneous

        # Compose each relative transformation with the reference frame
        absolute_poses: List[EndEffectorPose] = []
        for relative_pose in self._poses:
            # Get relative transformation as homogeneous matrix
            T_relative = relative_pose.homogeneous

            # Compose transformations: T_absolute = T_ref @ T_relative
            T_absolute = T_ref @ T_relative

            # Create absolute pose from composed transformation
            absolute_pose = EndEffectorPose(homogeneous=T_absolute)
            absolute_poses.append(absolute_pose)

        return EndEffectorActionChunk(absolute_poses, times=self.times)

    def to(self, action_format: ActionFormat) -> NDArray[np.float64]:
        """
        Convert trajectory to the desired format.

        Args:
            action_format: The desired output format

        Returns:
            Array in the requested format

        Raises:
            ValueError: If the action format is not supported

        【中文】按指定的 ActionFormat 导出末端轨迹：
        【中文】- DEFAULT: 返回 (N, 4, 4) 的齐次矩阵序列；
        【中文】- XYZ_ROT6D: 返回 (N, 9) 的 xyz+rot6d 轨迹；
        【中文】- XYZ_ROTVEC: 返回 (N, 6) 的 xyz+rotvec 轨迹；
        【中文】- 其他未支持的格式会抛出 ValueError，提醒调用方使用受支持的几种编码方式。
        """
        if action_format == ActionFormat.DEFAULT:
            return self.to_homogeneous_matrices()
        elif action_format == ActionFormat.XYZ_ROT6D:
            return self.to_translation_rot6d()
        elif action_format == ActionFormat.XYZ_ROTVEC:
            return self.to_translation_rotvec()
        else:
            raise ValueError(f"Unsupported action format: {action_format}")
