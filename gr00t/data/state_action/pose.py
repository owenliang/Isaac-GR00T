from enum import Enum
from typing import Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


# TypeVar for self-type preservation in Pose operations
PoseT = TypeVar("PoseT", bound="Pose")


def invert_transformation(T: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Invert a homogeneous transformation matrix.

    Args:
        T: A 4x4 homogeneous transformation matrix

    Returns:
        The inverse of the transformation matrix (4x4)

    【中文】对 4x4 齐次变换矩阵求逆：
    【中文】- 输入 T 由旋转 R 和平移 t 组成，形如 [[R, t], [0, 1]]；
    【中文】- 先对旋转部分转置 R^T，作为逆旋转；
    【中文】- 再计算新的平移向量 -R^T @ t，表示在逆旋转坐标系下的反向位移；
    【中文】- 最终返回的仍是一个 4x4 齐次变换矩阵，可与其他位姿做左乘/右乘运算。
    """
    R = T[:3, :3]  # Extract the rotation matrix
    t = T[:3, 3]  # Extract the translation vector

    # Inverse of the rotation matrix is its transpose (since it's orthogonal)
    R_inv = R.T

    # Inverse of the translation is -R_inv * t
    t_inv = -R_inv @ t

    # Construct the inverse transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv

    return T_inv


def relative_transformation(
    T0: NDArray[np.float64], Tt: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Compute the relative transformation between two poses.

    Args:
        T0: Initial 4x4 homogeneous transformation matrix
        Tt: Current 4x4 homogeneous transformation matrix

    Returns:
        The relative transformation matrix (4x4) from T0 to Tt

    【中文】计算两个齐次位姿之间的相对变换：
    【中文】- 输入 T0 表示参考帧（起始位姿），Tt 表示当前帧；
    【中文】- 通过 T0^{-1} @ Tt 得到“从 T0 到 Tt”需要施加的位姿增量；
    【中文】- 在末端位姿计算中，这个结果可用来表示相对运动，用于相对动作或误差反馈。
    """
    # Relative transformation is T0^{-1} * Tt
    T_relative = invert_transformation(T0) @ Tt
    return T_relative


class RotationType(Enum):
    """Supported rotation representation types.

    【中文】支持的“旋转表示”类型，用于在不同数值格式之间做统一转换：
    【中文】- QUAT: 四元数；
    【中文】- EULER: 欧拉角；
    【中文】- ROTVEC: 旋转向量（轴-角）；
    【中文】- MATRIX: 3x3 旋转矩阵；
    【中文】- ROT6D: 6 维旋转表示（矩阵前两行展平），常用于深度学习中避免万向节锁。
    """

    QUAT = "quat"
    EULER = "euler"
    ROTVEC = "rotvec"
    MATRIX = "matrix"
    ROT6D = "rot6d"


class EulerOrder(Enum):
    """Common Euler angle conventions.

    【中文】常见的欧拉角旋转顺序约定：
    【中文】- 如 XYZ、ZYX 等，对应 `Rotation.from_euler` / `as_euler` 中的顺序字符串；
    【中文】不同顺序对应的旋转含义不同，必须与生成/消费该角度的下游模块保持一致。
    """

    XYZ = "xyz"
    ZYX = "zyx"
    XZY = "xzy"
    YXZ = "yxz"
    YZX = "yzx"
    ZXY = "zxy"


class QuatOrder(Enum):
    """Quaternion ordering conventions.

    【中文】四元数分量顺序约定：
    【中文】- WXYZ: 标量在前 (w, x, y, z)，常见于某些机器人中间件；
    【中文】- XYZW: 标量在后 (x, y, z, w)，scipy 默认使用该顺序；
    【中文】本模块在与 scipy 交互时会根据约定做顺序转换，避免混用导致的旋转错误。
    """

    WXYZ = "wxyz"  # scalar-first (w, x, y, z)
    XYZW = "xyzw"  # scalar-last (x, y, z, w)


class Pose:
    """
    Abstract base class for robot poses.

    This class provides common functionality for different pose representations
    including relative pose computation via the subtraction operator.

    【中文】机器人“位姿”抽象基类：
    【中文】- 为关节空间位姿 (JointPose) 和末端位姿 (EndEffectorPose) 提供统一接口；
    【中文】- 约定减法运算符 `self - other` 表示“相对位姿”，具体计算由子类实现；
    【中文】- 通过 `pose_type` 区分不同位姿类型，便于上层逻辑做分支处理。
    """

    pose_type: str

    def __sub__(self: PoseT, other: PoseT) -> PoseT:
        """
        Compute relative transformation between two poses.

        For EndEffectorPose: Computes the relative transformation from other to self.
        Result represents the transformation needed to go from other's frame to self's frame.

        For JointPose: Computes the joint-space difference (self - other).

        Args:
            other: The reference pose to compute relative transformation from

        Returns:
            Relative pose (same type as self)

        Raises:
            TypeError: If poses are not of the same type

        Examples:
            # End-effector poses
            pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                                   rotation_type="quat", rotation_order="wxyz")
            relative = pose2 - pose1  # Transformation from pose1 to pose2

            # Joint poses
            joint1 = JointPose([0.0, 0.5, 1.0])
            joint2 = JointPose([0.1, 0.6, 1.2])
            joint_diff = joint2 - joint1  # Joint differences: [0.1, 0.1, 0.2]

        【中文】统一的“位姿相减”接口：
        【中文】- 若是 EndEffectorPose，则计算从 `other` 到 `self` 的相对位姿（T_other^{-1} * T_self）；
        【中文】- 若是 JointPose，则按元素做关节向量差值，得到关节空间的相对位移；
        【中文】- 要求两者类型一致，否则抛出 TypeError，避免混用关节/笛卡尔空间位姿。
        """
        if type(self) is not type(other):
            raise TypeError(
                f"Cannot compute relative transformation between different pose types: "
                f"{type(self).__name__} and {type(other).__name__}"
            )

        return self._compute_relative(other)

    def _compute_relative(self: PoseT, other: PoseT) -> PoseT:
        """
        Internal method to compute relative transformation.
        Must be implemented by subclasses.

        Args:
            other: The reference pose

        Returns:
            Relative pose

        【中文】子类需要实现的“相对位姿计算”内部接口：
        【中文】- JointPose 中实现为关节向量相减；
        【中文】- EndEffectorPose 中实现为齐次矩阵相对变换；
        【中文】基类不提供默认实现，强制要求子类根据各自语义明确相对运算的定义。
        """
        raise NotImplementedError("Subclasses must implement _compute_relative")

    def copy(self: PoseT) -> PoseT:
        """
        Create a deep copy of this pose.
        Must be implemented by subclasses.

        Returns:
            New Pose instance with copied data

        【中文】创建当前位姿对象的深拷贝：
        【中文】- 返回一个独立的新实例，内部数值（关节/位姿矩阵）不与原对象共享内存；
        【中文】- 具体拷贝逻辑由子类负责实现，以保证各自内部缓存/属性的一致性。
        """
        raise NotImplementedError("Subclasses must implement copy")


class JointPose(Pose):
    """
    Represents a robot configuration in joint space.

    This class stores joint angles/positions for a robot manipulator.
    Unlike end-effector poses, joint poses represent the configuration
    of all joints in the kinematic chain.

    Examples:
        # Create a 6-DOF joint configuration
        joint_pose = JointPose(
            joints=[0.0, -np.pi/4, np.pi/2, 0.0, np.pi/4, 0.0],
            joint_names=["shoulder_pan", "shoulder_lift", "elbow",
                        "wrist_1", "wrist_2", "wrist_3"]
        )

        # Create with default joint names
        joint_pose = JointPose(joints=[0.0, 0.5, 1.0])

        # Get as dictionary
        joint_dict = joint_pose.to_dict()  # {"joint_0": 0.0, ...}

        # Access individual joints
        first_joint = joint_pose.joints[0]
        num_joints = joint_pose.num_joints

        # Compute relative joint displacement
        joint1 = JointPose([0.0, 0.5, 1.0])
        joint2 = JointPose([0.1, 0.6, 1.2])
        relative = joint2 - joint1  # [0.1, 0.1, 0.2]

    【中文】关节空间下的位姿表示类，用一个向量保存机器人各关节的角度/位置。
    【中文】与末端位姿不同，JointPose 描述的是整个关节链在关节空间中的构型，可用于：
    【中文】1）直接表示当前关节角配置；2）两帧相减得到相对关节位移；3）作为相对轨迹的参考帧。
    """

    pose_type = "joint"

    def __init__(
        self,
        joints: Union[list, np.ndarray],
        joint_names: Optional[list] = None,
    ):
        """
        Initialize a joint pose.

        Args:
            joints: Joint angles/positions as array-like of shape (n,)
            joint_names: Optional list of names for each joint. If None,
                        defaults to ["joint_0", "joint_1", ...]
        """
        super().__init__()
        self.joints = np.array(joints, dtype=np.float64)

        # Set defaults and validate joint_names
        if joint_names is None:
            self.joint_names = [f"joint_{i}" for i in range(len(self.joints))]
        else:
            if len(joint_names) != len(self.joints):
                raise ValueError(
                    f"Number of joint names ({len(joint_names)}) must match "
                    f"number of joints ({len(self.joints)})"
                )
            self.joint_names = joint_names

    @property
    def num_joints(self) -> int:
        """
        Get the number of joints.

        Returns:
            Number of joints in the configuration
        """
        return len(self.joints)

    def to_dict(self) -> dict:
        """
        Convert joint configuration to dictionary.

        Returns:
            Dictionary mapping joint names to joint values
        """
        return dict(zip(self.joint_names, self.joints))

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative joint displacement.

        Args:
            other: Reference joint pose

        Returns:
            JointPose representing the joint-space difference (self - other)

        Raises:
            ValueError: If joint dimensions don't match

        【中文】计算当前关节姿态相对于参考姿态的“关节位移”：self - other。
        【中文】要求两者关节维度一致，否则抛出异常；结果仍然是一个 JointPose，可继续参与后续运算。
        """
        if len(self.joints) != len(other.joints):
            raise ValueError(
                f"Cannot compute relative joint pose: "
                f"joint dimensions don't match ({len(self.joints)} vs {len(other.joints)})"
            )

        relative_joints = self.joints - other.joints
        return JointPose(joints=relative_joints, joint_names=self.joint_names)

    def copy(self) -> "JointPose":
        """
        Create a deep copy of this joint pose.

        Returns:
            New JointPose instance with copied data
        """
        return JointPose(
            joints=self.joints.copy(),
            joint_names=self.joint_names.copy(),
        )

    def __repr__(self) -> str:
        if len(self.joints) <= 6:
            joints_str = np.array2string(self.joints, precision=4, suppress_small=True)
        else:
            joints_str = (
                f"[{self.joints[0]:.4f}, ..., {self.joints[-1]:.4f}] ({len(self.joints)} joints)"
            )

        return f"JointPose(joints={joints_str})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, JointPose):
            return False
        return np.allclose(self.joints, other.joints) and self.joint_names == other.joint_names

    def __getitem__(self, index) -> Union[float, NDArray[np.float64]]:
        """Allow indexing: joint_pose[0] returns first joint value"""
        return self.joints[index]

    def __len__(self) -> int:
        """Allow len(): len(joint_pose) returns number of joints"""
        return len(self.joints)


class EndEffectorPose(Pose):
    """
    Represents a single end-effector pose with translation and rotation components.

    This class handles Cartesian space representations of robot end-effector poses,
    supporting multiple rotation representations (quaternions, Euler angles, rotation
    vectors, rotation matrices, etc.).

    Examples:
        # Create with quaternion (wxyz order)
        pose = EndEffectorPose(
            translation=[1.0, 2.0, 3.0],
            rotation=[1.0, 0.0, 0.0, 0.0],
            rotation_type="quat",
            rotation_order="wxyz"
        )

        # Create with Euler angles (degrees by default)
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, 90],
            rotation_type="euler",
            rotation_order="xyz"
        )

        # Create with Euler angles in radians
        pose = EndEffectorPose(
            translation=[1, 2, 3],
            rotation=[0, 0, np.pi/2],
            rotation_type="euler",
            rotation_order="xyz",
            degrees=False
        )

        # Create from homogeneous matrix
        H = np.eye(4)
        H[:3, 3] = [1, 2, 3]
        pose = EndEffectorPose(homogeneous=H)

        # Convert between representations
        quat_wxyz = pose.to_rotation("quat", "wxyz")
        euler_zyx = pose.to_rotation("euler", "zyx")
        rot6d = pose.to_rotation("rot6d")

        # Compute relative transformation
        pose1 = EndEffectorPose(translation=[1, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        pose2 = EndEffectorPose(translation=[2, 0, 0], rotation=[1,0,0,0],
                               rotation_type="quat", rotation_order="wxyz")
        relative = pose2 - pose1  # Transformation from pose1's frame to pose2's frame

    【中文】末端执行器在笛卡尔空间下的位姿表示：
    【中文】- 同时包含平移向量和平移矩阵（齐次变换）两部分信息；
    【中文】- 内部统一用 scipy 的 Rotation 表示旋转，但对外支持多种表示格式（quat/euler/rotvec/matrix/rot6d）；
    【中文】- 支持从齐次矩阵构造，以及不同旋转表示之间的互相转换；
    【中文】- 通过减法运算可以方便地得到相对末端位姿，用于相对动作或误差计算。
    """

    pose_type = "end_effector"

    def __init__(
        self,
        translation: Optional[Union[list, np.ndarray]] = None,
        rotation: Optional[Union[list, np.ndarray]] = None,
        rotation_type: Optional[str] = None,
        rotation_order: Optional[str] = None,
        homogeneous: Optional[np.ndarray] = None,
        degrees: bool = True,
    ):
        """
        Initialize an end-effector pose.

        Args:
            translation: Translation vector [x, y, z]
            rotation: Rotation in specified format
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            homogeneous: Homogeneous transformation matrix (4, 4)
                        If provided, overrides translation and rotation
            degrees: For Euler angles, whether the input is in degrees (default True)

        【中文】末端位姿的构造函数：
        【中文】- 可以通过齐次矩阵 `homogeneous` 一次性指定平移和旋转，此时忽略 translation/rotation 参数；
        【中文】- 也可以分别传入平移向量和旋转（配合 rotation_type / rotation_order 指明表示方式）；
        【中文】- 对于欧拉角，`degrees=True` 表示输入为角度制，False 表示弧度制；
        【中文】无论何种方式，内部都会统一成 Rotation 对象和 3 维平移向量，方便后续转换和相对运算。
        """
        super().__init__()

        # Cache for homogeneous matrix
        self._homogeneous_cache: Optional[NDArray[np.float64]] = None
        self._cache_valid = False

        # Handle homogeneous matrix input
        if homogeneous is not None:
            self._from_homogeneous(homogeneous)
            return

        # Store translation
        self._translation = np.array(translation) if translation is not None else np.zeros(3)

        # Store rotation as scipy Rotation object internally
        if rotation is not None:
            if rotation_type is None:
                raise ValueError("rotation_type must be specified when rotation is provided")
            self._set_rotation(rotation, rotation_type, rotation_order, degrees)
        else:
            self._rotation = Rotation.identity()

    def _from_homogeneous(self, homogeneous: np.ndarray):
        """Initialize from homogeneous transformation matrix.

        【中文】从 4x4 齐次变换矩阵初始化末端位姿：
        【中文】- 直接从矩阵最后一列前三个元素提取平移向量；
        【中文】- 从左上角 3x3 子矩阵提取旋转矩阵，并构造 Rotation 对象；
        【中文】用于在外部算法已经以齐次矩阵形式给出位姿时，快速载入到统一的 Pose 表示中。
        """
        homogeneous = np.array(homogeneous)

        # Extract translation (last column, first 3 rows)
        self._translation = homogeneous[:3, 3]

        # Extract rotation matrix (top-left 3x3)
        rotation_matrix = homogeneous[:3, :3]

        # Create Rotation object from matrix
        self._rotation = Rotation.from_matrix(rotation_matrix)

    @staticmethod
    def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation to rotation matrix.

        Args:
            rot6d: 6D rotation as (6,) array - first two rows of rotation matrix flattened

        Returns:
            Rotation matrix (3, 3)

        【中文】将 6 维旋转表示还原为 3x3 旋转矩阵：
        【中文】- 输入是将旋转矩阵的前两行展平成长度为 6 的向量；
        【中文】- 先对第一行做归一化，视作正交基中的第一个向量；
        【中文】- 再对第二行做 Gram-Schmidt 正交化并归一化，得到第二个基向量；
        【中文】- 第三个基向量由前两者叉乘得到，保证矩阵正交、行列式为 1。
        """
        rot6d = rot6d.reshape(2, 3)

        # First two rows
        row1 = rot6d[0]
        row2 = rot6d[1]

        # Normalize first row
        row1 = row1 / np.linalg.norm(row1)

        # Gram-Schmidt orthogonalization for second row
        row2 = row2 - np.dot(row1, row2) * row1
        row2 = row2 / np.linalg.norm(row2)

        # Third row is cross product
        row3 = np.cross(row1, row2)

        # Construct rotation matrix
        rotation_matrix = np.vstack([row1, row2, row3])

        return rotation_matrix

    @staticmethod
    def _matrix_to_rot6d(rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to 6D rotation representation.

        Args:
            rotation_matrix: Rotation matrix (3, 3)

        Returns:
            6D rotation - (6,) array (first two rows flattened)

        【中文】将 3x3 旋转矩阵转换为 6 维旋转表示：
        【中文】- 直接取旋转矩阵的前两行并展平为长度为 6 的向量；
        【中文】- 这一表示常用于神经网络输出空间，因为它天然满足正交性约束，数值更稳定。
        """
        return rotation_matrix[:2, :].flatten()

    def _set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """Internal method to set rotation from various representations.

        【中文】内部方法：根据不同的输入旋转表示设置内部的 Rotation 对象：
        【中文】- 根据 rotation_type 选择四元数 / 欧拉角 / 旋转向量 / 矩阵 / 6D 表示的解析路径；
        【中文】- 对四元数会根据 rotation_order 处理 wxyz/xyzw 顺序转换；
        【中文】- 设置完成后会将缓存的齐次矩阵标记为无效，等待下次访问时重算。
        """
        rotation = np.array(rotation)
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            if quat_order == QuatOrder.WXYZ:
                # scipy uses xyzw order, so convert
                quat_xyzw = np.array([rotation[1], rotation[2], rotation[3], rotation[0]])
            else:
                quat_xyzw = rotation
            self._rotation = Rotation.from_quat(quat_xyzw)

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            self._rotation = Rotation.from_euler(euler_order.value, rotation, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            self._rotation = Rotation.from_rotvec(rotation)

        elif rot_type == RotationType.MATRIX:
            self._rotation = Rotation.from_matrix(rotation)

        elif rot_type == RotationType.ROT6D:
            rotation_matrix = self._rot6d_to_matrix(rotation)
            self._rotation = Rotation.from_matrix(rotation_matrix)

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

        # Invalidate cache
        self._cache_valid = False

    @property
    def translation(self) -> np.ndarray:
        """
        Get translation vector.

        Returns:
            Translation array - shape (3,)

        【中文】获取当前末端位姿的平移向量 [x, y, z]：
        【中文】- 返回的是内部副本的拷贝，调用方修改返回值不会影响内部状态；
        【中文】- 常与 `rot6d`/`rotvec` 等一起拼接成网络输入特征。
        """
        return self._translation.copy()

    @property
    def quat_wxyz(self) -> np.ndarray:
        """Get rotation as quaternion in wxyz order (w, x, y, z).

        【中文】以 wxyz 顺序返回旋转四元数（标量在前）：
        【中文】- 便于与某些机器人中间件或控制栈对接，这些系统多采用 wxyz 约定；
        【中文】- 内部会从 scipy 默认的 xyzw 顺序转换而来。
        """
        return self.to_rotation("quat", "wxyz")

    @property
    def quat_xyzw(self) -> np.ndarray:
        """Get rotation as quaternion in xyzw order (x, y, z, w).

        【中文】以 xyzw 顺序返回四元数（标量在后）：
        【中文】- 这是 scipy Rotation 的默认输出格式；
        【中文】- 适合直接与科学计算或学习框架交互。
        """
        return self.to_rotation("quat", "xyzw")

    @property
    def euler_xyz(self) -> np.ndarray:
        """Get rotation as Euler angles in xyz order (degrees).

        【中文】以 xyz 顺序返回欧拉角（默认单位为角度）：
        【中文】- 对应按 X→Y→Z 依次旋转的欧拉角约定；
        【中文】- 若需要弧度，可在 `to_rotation("euler", "xyz", degrees=False)` 中自行指定。
        """
        return self.to_rotation("euler", "xyz")

    @property
    def rotvec(self) -> np.ndarray:
        """Get rotation as rotation vector (axis-angle).

        【中文】以旋转向量（轴-角）形式返回旋转：
        【中文】- 向量方向表示旋转轴，模长表示旋转角度（弧度）；
        【中文】- 这种表示在插值和优化中较为常见。
        """
        return self.to_rotation("rotvec")

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get rotation as 3x3 rotation matrix.

        【中文】以 3x3 旋转矩阵形式返回当前旋转：
        【中文】- 适合与齐次变换或外部几何库对接；
        【中文】- 也是构造 `homogeneous` 变换矩阵时使用的内部格式。
        """
        return self.to_rotation("matrix")

    @property
    def rot6d(self) -> np.ndarray:
        """Get rotation as 6D representation (first two rows of rotation matrix).

        【中文】以 6 维向量形式返回旋转（旋转矩阵前两行展平）：
        【中文】- 这一表示既能表示任意 SO(3) 旋转，又消除了四元数归一化等约束问题；
        【中文】- 常用于神经网络的姿态回归任务中，数值稳定性更好。
        """
        return self.to_rotation("rot6d")

    @property
    def xyz_rot6d(self) -> np.ndarray:
        """Get pose as concatenated translation and 6D rotation (9,).

        【中文】将平移向量和 6D 旋转拼接成长度为 9 的姿态向量 [x, y, z, rot6d...]：
        【中文】- 这是本项目中常用的末端位姿数值表示形式；
        【中文】- 方便直接作为网络输入或输出目标。
        """
        return np.concatenate([self._translation, self.rot6d])

    @property
    def xyz_rotvec(self) -> np.ndarray:
        """Get pose as concatenated translation and rotation vector (6,).

        【中文】将平移向量和旋转向量拼接成长度为 6 的姿态向量 [x, y, z, rx, ry, rz]：
        【中文】- 适合需要轴-角形式的下游模块；
        【中文】- 可视为 xyz + SO(3) 的另一种参数化方式。
        """
        return np.concatenate([self._translation, self.rotvec])

    @property
    def homogeneous(self) -> np.ndarray:
        """
        Get homogeneous transformation matrix.

        Returns:
            Homogeneous matrix - shape (4, 4)

        【中文】获取对应的 4x4 齐次变换矩阵：
        【中文】- 上三行三列为旋转矩阵，最后一列前三个元素为平移向量；
        【中文】- 为避免重复计算，内部使用缓存 `_homogeneous_cache`，当位姿更新时会自动失效重算。
        """
        if not self._cache_valid:
            self._homogeneous_cache = self._compute_homogeneous()
            self._cache_valid = True
        assert self._homogeneous_cache is not None
        return self._homogeneous_cache.copy()

    def _compute_homogeneous(self) -> np.ndarray:
        """Compute homogeneous transformation matrix.

        【中文】根据当前内部的旋转和平移，显式构造 4x4 齐次变换矩阵：
        【中文】- 旋转部分来自 `self._rotation.as_matrix()`；
        【中文】- 平移部分来自 `self._translation`；
        【中文】该方法仅在缓存失效时被调用，其结果会被缓存以提高后续访问效率。
        """
        H = np.eye(4)
        H[:3, :3] = self._rotation.as_matrix()
        H[:3, 3] = self._translation
        return H

    def to_rotation(
        self, rotation_type: str, rotation_order: Optional[str] = None, degrees: bool = True
    ) -> np.ndarray:
        """
        Get rotation in specified representation.

        Args:
            rotation_type: Desired type ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, return in degrees (default True)

        Returns:
            Rotation in requested format
            - Shape (4,) for quat
            - Shape (3,) for euler/rotvec
            - Shape (6,) for rot6d
            - Shape (3, 3) for matrix

        【中文】按指定的表示形式导出当前末端旋转：
        【中文】- 通过 `rotation_type` 选择输出格式，必要时结合 `rotation_order` 指定顺序约定；
        【中文】- 对欧拉角可通过 `degrees` 控制返回角度制或弧度制；
        【中文】- 是本模块所有旋转 getter（如 quat_wxyz/euler_xyz）的底层实现。
        """
        rot_type = RotationType(rotation_type.lower())

        if rot_type == RotationType.ROT6D:
            rotation_matrix = self._rotation.as_matrix()
            return self._matrix_to_rot6d(rotation_matrix)

        if rot_type == RotationType.QUAT:
            quat_order = QuatOrder(rotation_order.lower()) if rotation_order else QuatOrder.WXYZ
            quat_xyzw = self._rotation.as_quat()
            if quat_order == QuatOrder.WXYZ:
                return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            else:
                return quat_xyzw

        elif rot_type == RotationType.EULER:
            euler_order = EulerOrder(rotation_order.lower()) if rotation_order else EulerOrder.XYZ
            return self._rotation.as_euler(euler_order.value, degrees=degrees)

        elif rot_type == RotationType.ROTVEC:
            return self._rotation.as_rotvec()

        elif rot_type == RotationType.MATRIX:
            return self._rotation.as_matrix()

        else:
            raise ValueError(f"Unknown rotation type: {rotation_type}")

    def to_homogeneous(self) -> np.ndarray:
        """
        Convert pose to homogeneous transformation matrix.
        (Alias for the homogeneous property)

        Returns:
            Homogeneous matrix - shape (4, 4)

        【中文】将当前位姿转换为 4x4 齐次变换矩阵的便捷接口：
        【中文】- 等价于访问 `self.homogeneous` 属性；
        【中文】- 便于在需要函数调用形式的场景中使用（例如与外部库接口对齐）。
        """
        return self.homogeneous

    def set_rotation(
        self,
        rotation: Union[list, np.ndarray],
        rotation_type: str,
        rotation_order: Optional[str] = None,
        degrees: bool = True,
    ):
        """
        Set rotation from specified representation.

        Args:
            rotation: Rotation data
            rotation_type: Type of rotation ("quat", "euler", "rotvec", "matrix", "rot6d")
            rotation_order: Order/convention for the rotation type
            degrees: For Euler angles, whether the input is in degrees (default True)

        【中文】根据给定的旋转数据更新末端姿态：
        【中文】- 会调用内部 `_set_rotation` 完成解析与缓存失效；
        【中文】- 支持所有与构造函数相同的旋转表示，用于在运行中重设姿态朝向。
        """
        self._set_rotation(rotation, rotation_type, rotation_order, degrees)

    def _compute_relative(self, other):  # type: ignore[override]
        """
        Compute relative transformation from other to self.

        The result represents the transformation needed to go from other's frame to self's frame.
        Mathematically: T_relative = T_other^{-1} * T_self

        Args:
            other: Reference end-effector pose

        Returns:
            EndEffectorPose representing the relative transformation

        【中文】计算“从参考末端姿态 other 到当前姿态 self”的相对位姿：
        【中文】- 先将两者转换为齐次矩阵 T_other 和 T_self；
        【中文】- 再通过 T_other^{-1} @ T_self 得到在 other 坐标系下的位姿增量；
        【中文】- 返回一个新的 EndEffectorPose 实例，表示这一步的相对运动，可用于相对动作编码。
        """
        # Get homogeneous matrices
        T_self = self.homogeneous
        T_other = other.homogeneous

        # Compute relative transformation: T_other^{-1} * T_self
        T_relative = relative_transformation(T_other, T_self)

        # Create new EndEffectorPose from relative transformation
        return EndEffectorPose(homogeneous=T_relative)

    def copy(self) -> "EndEffectorPose":
        """
        Create a deep copy of this end-effector pose.

        Returns:
            New EndEffectorPose instance with copied data

        【中文】创建当前末端位姿的深拷贝：
        【中文】- 拷贝平移向量与四元数旋转，生成一个新的 EndEffectorPose 实例；
        【中文】- 新旧实例之间互不共享内部缓存或数组，适合在需要保存历史姿态时使用。
        """
        return EndEffectorPose(
            translation=self._translation.copy(),
            rotation=self._rotation.as_quat(),
            rotation_type="quat",
            rotation_order="xyzw",
        )

    def __repr__(self) -> str:
        quat = self.to_rotation("quat", "wxyz")
        return f"EndEffectorPose(translation={self.translation}, rotation_quat_wxyz={quat})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, EndEffectorPose):
            return False
        return np.allclose(self._translation, other._translation) and np.allclose(
            self._rotation.as_quat(), other._rotation.as_quat()
        )
