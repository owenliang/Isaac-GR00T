from concurrent.futures import Future, ThreadPoolExecutor
import time

import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from gr00t.data.interfaces import BaseProcessor, ShardedDataset


def merge_statistics(
    per_dataset_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
    dataset_sampling_weights: list[float] | np.ndarray,
    is_relative_stats: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """
    Compute overall statistics from per-dataset statistics using weighted averaging.

    This function combines statistics from multiple datasets according to their sampling
    weights, computing weighted means and variances while preserving min/max/quantile
    information across all datasets.

    The weighted variance computation uses the formula:
    Var_combined = Σ(w_i * (σ_i² + μ_i²)) - (Σ(w_i * μ_i))²

    Args:
        per_dataset_stats: List of per-dataset statistics dictionaries.
            Each element has structure: {modality: {joint_group: {stat_type: values}}}
            Example: {"state": {"gripper": {"mean": [0.1, 0.2], "std": [0.5, 0.3]}}}
        dataset_sampling_weights: Weights for combining dataset statistics.
            Should sum to 1.0 or will be normalized.
        is_relative_stats: Whether the statistics are relative (affects merging logic).

    Returns:
        Combined statistics dictionary with same structure as input, containing
        weighted averages for mean/std and global min/max/quantiles across datasets.

    【中文】将多个数据集各自的统计量（按模态/关节组划分）按采样权重做加权合并。
    【中文】mean/std 使用加权均值与方差公式合并，min/max/分位数则在各数据集之间取全局极值（保守边界），
    【中文】得到一个统一的统计字典，供跨数据集、跨具身形态训练时做一致的归一化。
    """
    # Normalize sampling weights to sum to 1
    dataset_sampling_weights = np.array(dataset_sampling_weights)
    normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    # Initialize overall statistics dict
    overall_stats: dict[str, dict[str, list[float]]] = {}

    # Process each modality (e.g., "state", "action")
    for modality in per_dataset_stats[0]:
        # Get dimensionality from first dataset (assumed consistent)
        dim = (
            [len(per_dataset_stats[0][modality]["mean"])]
            if not is_relative_stats
            else np.array(per_dataset_stats[0][modality]["mean"]).shape
        )

        # Initialize accumulators for weighted mean and variance computation
        weighted_means = np.zeros(dim)
        weighted_squares = np.zeros(dim)

        # Collect min/max/quantiles from all datasets for global computation
        min_list = []
        max_list = []
        q01_list = []
        q99_list = []

        # Accumulate weighted statistics across datasets
        for dataset_idx, dataset_stats in enumerate(per_dataset_stats):
            w_i = normalized_weights[dataset_idx]
            stats = dataset_stats[modality]
            means = np.array(stats["mean"])
            stds = np.array(stats["std"])

            # Update weighted sums for mean and variance calculation
            weighted_means += w_i * means
            weighted_squares += w_i * (stds**2 + means**2)

            # Collect extremes and quantiles for global computation
            min_list.append(stats["min"])
            max_list.append(stats["max"])
            q01_list.append(stats["q01"])
            q99_list.append(stats["q99"])

        # Compute final combined statistics
        overall_mean = weighted_means.tolist()
        overall_variance = weighted_squares - weighted_means**2
        overall_std = np.sqrt(overall_variance).tolist()

        # Global min/max across all datasets
        overall_min = np.min(np.array(min_list), axis=0).tolist()
        overall_max = np.max(np.array(max_list), axis=0).tolist()

        # Global quantiles (conservative bounds across datasets)
        q01_array = np.array(q01_list)
        q99_array = np.array(q99_list)
        weighted_q01 = np.min(q01_array, axis=0).tolist()
        weighted_q99 = np.max(q99_array, axis=0).tolist()

        # Store combined statistics for this modality
        overall_stats[modality] = {
            "min": overall_min,
            "max": overall_max,
            "mean": overall_mean,
            "std": overall_std,
            "q01": weighted_q01,
            "q99": weighted_q99,
        }

    return overall_stats


class ShardedMixtureDataset(IterableDataset):
    """
    Iterable dataset that combines multiple sharded datasets with configurable mixing ratios.

    This dataset provides the core functionality for multi-dataset training in VLA systems:
    1. Combines multiple ShardedDataset instances with specified mixing weights
    2. Implements intelligent shard sampling that accounts for dataset sizes
    3. Provides efficient background shard caching for continuous data loading
    4. Handles distributed training across multiple workers and processes
    5. Merges dataset statistics for consistent normalization

    Key features:
    - Weighted sampling across datasets normalized by shard sizes
    - Background shard caching with ThreadPoolExecutor for efficiency
    - Distributed training support with proper shard allocation
    - Automatic epoch management and shard reshuffling
    - Per-embodiment statistics merging for cross-embodiment training

    The sampling strategy ensures that datasets are sampled proportionally to their
    weights while accounting for differences in shard sizes, preventing bias toward
    datasets with smaller shards.

    Args:
        datasets: List of ShardedDataset instances to combine
        weights: Mixing weights for each dataset (will be normalized)
        processor: Data processor to apply to all datasets
        seed: Random seed for reproducible sampling
        training: Whether in training mode (affects sampling strategy)
        num_shards_per_epoch: Number of shards to sample per epoch during training

    Example:
        >>> mixture = ShardedMixtureDataset(
        ...     datasets=[dataset1, dataset2, dataset3],
        ...     weights=[0.5, 0.3, 0.2],
        ...     processor=my_processor,
        ...     num_shards_per_epoch=10000,
        ... )
        >>> for batch in mixture:
        ...     # batch contains processed data from mixed datasets
        ...     pass

    【中文】将多个 `ShardedDataset` 按给定权重混合成一个可迭代数据集，用于多数据源联合训练。
    【中文】核心能力包括：按 shard 级别做加权采样、在后台预取/缓存 shard、支持分布式与多 worker 切分，
    【中文】并在具身形态维度上合并统计量，保证不同数据集间归一化策略一致。
    """

    def __init__(
        self,
        datasets: list[ShardedDataset],
        weights: list[float],
        processor: BaseProcessor,
        seed: int = 42,
        training: bool = True,
        num_shards_per_epoch: int = int(1e5),
        override_pretraining_statistics: bool = False,
    ):
        """Initialize mixture dataset with datasets, weights, and configuration.

        【中文】初始化混合数据集：保存各个底层 `ShardedDataset` 与其混合权重、全局 Processor、
        【中文】以及训练/评估模式和每个 epoch 中要采样的 shard 数量，并立即生成首个 shard 采样计划与合并统计量。
        """
        self.datasets = datasets
        self.weights = weights
        self.seed = seed
        self.training = training
        self.num_shards_per_epoch = num_shards_per_epoch
        self.epoch = 0
        self.processor = processor
        self.override_pretraining_statistics = override_pretraining_statistics

        # Generate initial shard sampling schedule
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()

        # Merge statistics across datasets and configure processor
        self.merge_statistics()

        # Initialize distributed training parameters
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.worker_id = None
        self.num_workers = None

        # Initialize shard caching system
        self.curr_shard = None
        self._executor = None
        self._cache_job: Future | None = None

    def merge_statistics(self):
        """
        Merge dataset statistics across all datasets, grouped by embodiment.

        Combines statistics from datasets with the same embodiment tag using
        weighted averaging, then configures the processor with merged statistics.
        This ensures consistent normalization across datasets within each embodiment.

        【中文】按具身形态（embodiment_tag）分组，将同一具身下不同数据集的统计量按权重合并，
        【中文】然后把合并后的统计配置到全局 Processor 和各个底层数据集上，
        【中文】从而在多数据集训练时保持同一具身内部的一致归一化策略。
        """
        # Group datasets and weights by embodiment
        all_stats_by_emb: dict[str, list] = {}
        weights_by_emb: dict[str, list[float]] = {}
        for ds, w in zip(self.datasets, self.weights):
            emb = getattr(ds, "embodiment_tag", None)
            if emb is None:
                continue
            emb = emb.value
            if emb not in all_stats_by_emb:
                all_stats_by_emb[emb] = []
                weights_by_emb[emb] = []
            stats = ds.get_dataset_statistics()  # type: ignore
            all_stats_by_emb[emb].append(stats)
            weights_by_emb[emb].append(w)

        # Merge statistics within each embodiment group
        stats_by_emb = {}
        for emb, stats in all_stats_by_emb.items():
            stats_by_emb[emb] = {}
            for modality in ["state", "action", "relative_action"]:
                if modality in stats[0]:
                    modality_stats = [s[modality] for s in stats]
                    stats_by_emb[emb][modality] = merge_statistics(
                        per_dataset_stats=modality_stats,
                        dataset_sampling_weights=weights_by_emb[emb],
                        is_relative_stats=(modality == "relative_action"),
                    )

        # Configure processor and datasets with merged statistics
        self.global_stats = stats_by_emb
        self.processor.set_statistics(
            self.global_stats, override=self.override_pretraining_statistics
        )
        for ds in self.datasets:
            ds.set_processor(self.processor)

    def get_dataset_statistics(self):
        """Get the merged dataset statistics."""
        return self.global_stats

    def generate_shard_sampling_schedule(self) -> list[tuple[int, int]]:
        """
        Generate a schedule of (dataset_index, shard_index) pairs for shard sampling.

        For training: Uses weighted random sampling normalized by average shard sizes
        to ensure fair representation regardless of shard size differences.

        For evaluation: Samples every shard from every dataset exactly once
        for comprehensive evaluation coverage.

        Returns:
            List of (dataset_index, shard_index) tuples defining the sampling order

        【中文】生成一个 `(dataset_index, shard_index)` 列表，描述本 epoch 中 shard 的采样顺序：
        【中文】- 训练模式：按“权重 ÷ 平均 shard 长度”归一化后进行加权随机采样，避免小 shard 的数据集被过采样；
        【中文】- 评估模式：遍历所有数据集的所有 shard，各采样一次，保证评估覆盖完整数据。
        """
        if self.training:
            # 【训练模式】使用基于 epoch 的随机种子,确保每个 epoch 有不同的采样顺序
            rng = np.random.default_rng(self.seed + self.epoch)

            # ==================== 步骤 1: 计算每个数据集的平均 shard 大小 ====================
            # 【目的】用于归一化权重,避免 shard 大小差异导致采样不均
            # 【示例】假设数据集 A 有 3 个 shard,大小分别为 [100, 150, 200],则平均 shard 大小为 150
            average_shard_sizes = []
            for dataset in self.datasets:
                # 遍历当前数据集的所有 shard,求和后计算平均值
                average_shard_size = sum(
                    dataset.get_shard_length(i) for i in range(len(dataset))
                ) / len(dataset)
                average_shard_sizes.append(average_shard_size)

            # ==================== 步骤 2: 归一化采样权重 ====================
            # 【关键逻辑】权重 ÷ 平均 shard 大小,再归一化至和为 1
            # 【原因】如果数据集 A 的权重为 0.6,但平均 shard 大小为 150,
            #        数据集 B 的权重为 0.4,平均 shard 大小为 50,
            #        那么按 shard 采样时,A 的每个 shard 包含更多数据,应该降低 A 的 shard 采样频率
            normalized_weights = np.array(
                [w / s for w, s in zip(self.weights, average_shard_sizes)]
            )
            # 归一化到和为 1,使其成为合法的概率分布
            normalized_weights = normalized_weights / normalized_weights.sum()

            # ==================== 步骤 3: 按归一化权重随机抽取数据集索引 ====================
            # 【输出】一个长度为 num_shards_per_epoch 的数组,每个元素表示要从哪个数据集采样
            # 【示例】如果有 2 个数据集,归一化权重为 [0.7, 0.3],采样 10 次,
            #        可能得到 [0, 0, 1, 0, 0, 1, 0, 0, 0, 1](数据集 0 出现约 7 次,数据集 1 出现约 3 次)
            dataset_sampling_schedule = rng.choice(
                len(self.datasets), size=self.num_shards_per_epoch, p=normalized_weights
            )

            # ==================== 步骤 4: 为每个数据集生成打乱的 shard 索引池 ====================
            # 【目的】为每个数据集准备一个随机顺序的 shard 索引队列,后续从中依次弹出
            # 【示例】数据集 0 有 5 个 shard,打乱后可能是 [2, 0, 4, 1, 3]
            shard_sampling_schedule = []  # 最终输出: [(dataset_idx, shard_idx), ...]
            shards_to_sample = []  # 每个数据集的 shard 索引池(打乱后的列表)
            for dataset in self.datasets:
                shard_ids = list(range(len(dataset)))  # 生成 [0, 1, 2, ..., num_shards-1]
                rng.shuffle(shard_ids)  # 随机打乱顺序
                shards_to_sample.append(shard_ids)

            # ==================== 步骤 5: 根据数据集采样计划生成最终的 shard 采样计划 ====================
            # 【循环逻辑】遍历步骤 3 中采样得到的数据集索引序列
            for i in dataset_sampling_schedule:  # i 是数据集索引
                # 【处理 shard 耗尽情况】如果当前数据集的 shard 池已空,重新打乱并填充
                # 【示例】数据集 0 有 3 个 shard,但被采样了 5 次,前 3 次用完后需要重新打乱并重用
                if len(shards_to_sample[i]) == 0:
                    shard_ids = list(range(len(self.datasets[i])))
                    rng.shuffle(shard_ids)
                    shards_to_sample[i] = shard_ids
                
                # 从当前数据集的 shard 池中弹出第一个 shard 索引
                shard_idx = shards_to_sample[i].pop(0)
                # 添加到最终采样计划: (数据集索引, shard 索引)
                shard_sampling_schedule.append((i, shard_idx))

        else:
            # 【评估模式】遍历所有数据集的所有 shard,不做随机采样,保证覆盖全部数据
            # 【输出示例】如果有 2 个数据集,分别有 3 和 2 个 shard,则输出:
            #            [(0,0), (0,1), (0,2), (1,0), (1,1)]
            shard_sampling_schedule = []
            for i, dataset in enumerate(self.datasets):
                shard_sampling_schedule.extend([(i, j) for j in range(len(dataset))])
        return shard_sampling_schedule

    def filter_shard_sample_schedule(self):
        """
        Filter the shard sampling schedule for distributed training.

        Distributes shards across world_size processes and num_workers per process,
        ensuring each worker gets a unique subset of shards for parallel processing.

        Returns:
            Filtered list of (dataset_index, shard_index) pairs for this worker

        【中文】在分布式 + 多 worker 场景下，根据 rank 和 worker_id 从全局 shard 采样计划中过滤出本 worker 负责的子集，
        【中文】通过对序号取模的方式将 shard 均匀划分给各个进程/worker，保证同一 shard 不会被多个 worker 重复处理。
        
        【Worker 概念说明】
        - Worker 是 PyTorch DataLoader 中的数据加载子进程，用于并行加载数据
        - 每个 GPU 进程(rank)可以有多个 worker 并行加载数据，加速 I/O
        - 例如：num_workers=4 表示每个 GPU 进程有 4 个子进程同时读取数据
        
        【分布式训练层次结构】
        - 第 1 层：多个 GPU 进程(world_size 个)，每个进程有一个 rank (0 到 world_size-1)
        - 第 2 层：每个 GPU 进程内有多个 worker 子进程(num_workers 个)，每个有 worker_id (0 到 num_workers-1)
        - 总 worker 数 = world_size × num_workers
        
        【示例】假设有 2 个 GPU，每个 GPU 有 3 个 worker：
        - GPU 0 (rank=0): worker 0, worker 1, worker 2
        - GPU 1 (rank=1): worker 0, worker 1, worker 2
        - 全局共 6 个 worker，需要将 shard 平均分配给这 6 个 worker
        """
        filtered_schedule = []
        
        # ==================== 步骤 1: 获取当前 worker 的配置信息 ====================
        # 【PyTorch DataLoader】通过 get_worker_info() 获取当前 worker 的 ID 和总数
        worker_info = get_worker_info()

        # 【判断是否在 worker 子进程中】
        # - 如果 worker_info 不为 None: 说明在 DataLoader 的 worker 子进程中运行
        # - 如果 worker_info 为 None: 说明在主进程中运行(num_workers=0 的情况)
        if worker_info is not None:
            worker_id = worker_info.id          # 当前 worker 的 ID (0 到 num_workers-1)
            num_workers = worker_info.num_workers  # 总 worker 数量
        else:
            # 【单进程模式】没有使用多 worker，主进程自己加载数据
            worker_id = 0
            num_workers = 1

        # ==================== 步骤 2: 缓存并验证 worker 配置的一致性 ====================
        # 【目的】确保在整个训练过程中 worker 配置不会改变，防止数据分配混乱
        if self.worker_id is None:
            # 【首次调用】缓存 worker 配置信息
            assert self.num_workers is None
            self.worker_id = worker_id
            self.num_workers = num_workers
        else:
            # 【后续调用】验证 worker 配置是否与之前一致
            # 【为什么需要验证】如果 worker 配置改变，会导致不同 worker 可能处理到相同的 shard，造成数据重复
            assert self.worker_id == worker_id and self.num_workers == num_workers, (
                "Worker ID or number of workers has been changed since it was set. This is not allowed."
            )

        # ==================== 步骤 3: 使用取模算法分配 shard 给当前 worker ====================
        # 【分配策略】轮流分配，确保每个 shard 只被一个 worker 处理
        # 【全局 worker 编号计算】global_worker_id = rank * num_workers + worker_id
        # 
        # 【示例】假设有 2 个 GPU(world_size=2)，每个 GPU 有 2 个 worker(num_workers=2)：
        #   GPU 0, worker 0: global_id = 0*2 + 0 = 0
        #   GPU 0, worker 1: global_id = 0*2 + 1 = 1
        #   GPU 1, worker 0: global_id = 1*2 + 0 = 2
        #   GPU 1, worker 1: global_id = 1*2 + 1 = 3
        # 
        # 【分配逻辑】如果有 10 个 shard，分配结果：
        #   - shard 0: 0 % 4 == 0 → worker (0,0)
        #   - shard 1: 1 % 4 == 1 → worker (0,1)
        #   - shard 2: 2 % 4 == 2 → worker (1,0)
        #   - shard 3: 3 % 4 == 3 → worker (1,1)
        #   - shard 4: 4 % 4 == 0 → worker (0,0)  # 循环回第一个 worker
        #   ... 以此类推
        for i, shard in enumerate(self.shard_sampling_schedule): 
            # 【取模公式】i % (总worker数) == 当前worker的全局编号
            # - i % (self.world_size * num_workers): 将 shard 序号映射到 [0, 总worker数-1]
            # - self.rank * num_workers + worker_id: 当前 worker 的全局编号
            if i % (self.world_size * num_workers) == self.rank * num_workers + worker_id:
                filtered_schedule.append(shard)
        
        return filtered_schedule

    def __iter__(self):
        """
        Iterate over the mixture dataset with background shard caching.

        Implements an efficient iteration strategy:
        1. Filter shards for this worker's portion
        2. Start background caching of the first shard
        3. For each shard: wait for cache, start caching next, yield current
        4. Shuffle timesteps within each shard for additional randomization
        5. Handle epoch transitions and schedule regeneration

        【中文】迭代逻辑：
        【中文】1）先根据 rank/worker 过滤出本 worker 负责的 shard 列表；
        【中文】2）用单线程线程池在后台异步预取下一个 shard（get_shard）；
        【中文】3）前台等待当前 shard 预取完成，打乱该 shard 内的样本次序后逐个 yield；
        【中文】4）用完后释放内存，并立即预取下一个 shard；
        【中文】5）若本轮计划耗尽，则切换到下一 epoch，重建采样计划并继续循环。
        """
        # Start background thread pool
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Initialize worker-specific shard schedule
        self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
        self.curr_shard_index = -1
        self.cache_next_shard()
        rng = np.random.default_rng(self.seed + self.epoch)

        # Continuous iteration with epoch management
        while True:
            self.curr_shard_index += 1

            # Wait for background caching to complete
            wait_start = time.time()
            self.finish_cache_shard()
            wait_end = time.time()

            dataset_index, shard_index = self.worker_shard_sampling_schedule[self.curr_shard_index]
            print(
                f"Rank {self.rank}, Worker {self.worker_id}: Wait for shard {shard_index} in dataset {dataset_index} in {wait_end - wait_start:.2f} seconds"
            )

            # Start caching next shard immediately
            self.cache_next_shard()

            # Yield shuffled timesteps from current shard
            assert self.curr_shard is not None
            indices_in_shard = np.arange(len(self.curr_shard))
            rng.shuffle(indices_in_shard)
            for index in indices_in_shard:
                yield self.curr_shard[index]

            # Clean up cached shard to free memory
            self.delete_cached_shard()

    def cache_next_shard(self):
        """
        Start background caching of the next shard using ThreadPoolExecutor.

        Handles epoch transitions by regenerating the sampling schedule when
        the current schedule is exhausted.

        【中文】使用线程池在后台异步加载"下一个将要使用的 shard"：
        【中文】若当前 worker 的 shard 列表已经用完，则进入下一 epoch，重新生成采样计划并重置索引，
        【中文】然后提交 `dataset.get_shard(shard_idx)` 任务到线程池，供前台迭代时等待和消费。
        
        【工作机制】
        - 在主迭代循环中，当前 shard 正在被消费时，这个函数在后台预加载下一个 shard
        - 利用单线程线程池实现异步 I/O，避免阻塞主迭代流程
        - 实现了"流水线"式数据加载：计算和 I/O 并行，提高训练效率
        
        【调用时机】
        - 在 __iter__ 中的每个 shard 处理完后立即调用，启动下一个 shard 的预加载
        - 第一次调用在迭代器初始化时，预加载第一个 shard
        """
        # 【前置检查】确保线程池已初始化
        assert self._executor is not None
        
        # ==================== 步骤 1: 检查是否需要切换到下一个 epoch ====================
        # 【判断条件】下一个要加载的 shard 索引是否超出当前 worker 的 shard 列表长度
        # 【curr_shard_index】当前正在处理的 shard 在 worker_shard_sampling_schedule 中的位置
        # 【示例】如果 worker 负责 5 个 shard (索引 0-4)，当 curr_shard_index=4 时，
        #        下一个索引 5 >= 5，说明本 epoch 的 shard 已经用完
        if self.curr_shard_index + 1 >= len(self.worker_shard_sampling_schedule):
            # 【Epoch 切换】本 epoch 的所有 shard 已经处理完毕，进入下一个 epoch
            self.epoch += 1
            
            # 【重新生成全局采样计划】调用之前的函数生成新的 shard 采样顺序
            # 这会使用新的随机种子 (seed + epoch)，确保每个 epoch 的数据顺序不同
            self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
            
            # 【过滤出本 worker 负责的部分】从全局计划中筛选出当前 worker 要处理的 shard
            self.worker_shard_sampling_schedule = self.filter_shard_sample_schedule()
            
            # 【重置索引】从头开始新的 epoch，索引设为 -1（下一步会变成 0）
            self.curr_shard_index = -1

        # ==================== 步骤 2: 获取下一个要加载的 shard 信息 ====================
        # 【日志输出】记录当前 worker 正在预加载 shard
        print(f"Rank {self.rank}, Worker {self.worker_id}: Caching shard...")
        
        # 【获取下一个 shard 的坐标】
        # - next_dataset_idx: 数据集索引，表示从哪个数据集加载
        # - next_shard_idx: shard 索引，表示加载该数据集的第几个 shard
        # 【示例】如果 worker_shard_sampling_schedule[5] = (2, 7)，
        #        表示要从数据集 2 中加载第 7 个 shard
        next_dataset_idx, next_shard_idx = self.worker_shard_sampling_schedule[
            self.curr_shard_index + 1
        ]
        
        # ==================== 步骤 3: 提交后台加载任务到线程池 ====================
        # 【异步提交】将 get_shard 方法提交到线程池执行，不阻塞当前线程
        # 【关键点】主线程可以继续处理当前 shard 的数据，同时后台线程在加载下一个 shard
        # 【返回 Future 对象】self._cache_job 是一个 Future，可以用 .result() 等待完成
        # 【类比】就像在餐厅点餐：你在吃当前这道菜时，厨房已经在准备下一道菜了
        self._cache_job = self._executor.submit(
            self.datasets[next_dataset_idx].get_shard, next_shard_idx
        )

    def finish_cache_shard(self):
        """Wait for the background caching job to complete and retrieve the shard.

        【中文】阻塞等待后台预取任务完成，将其结果（一个 shard 中的样本列表）赋值给 `self.curr_shard`，
        【中文】并清除 `_cache_job` 句柄，表示当前 shard 已经准备就绪可供迭代使用。
        """
        assert self._cache_job is not None
        self.curr_shard = self._cache_job.result()
        self._cache_job = None

    def delete_cached_shard(self):
        """Delete the current cached shard to free memory.

        【中文】显式删除当前缓存的 shard 引用，提示 Python 释放其占用的内存，
        【中文】避免长时间持有大批量样本造成内存压力。
        """
        del self.curr_shard

    def reset_seed(self, seed: int):
        """
        Reset the random seed and regenerate sampling schedules.

        Used for deterministic training restarts or seed changes during training.

        Args:
            seed: New random seed to use

        【中文】重置随机种子，并回到 epoch 0，重新生成 shard 采样计划和内部游标，
        【中文】方便在训练重启或需要改变随机性时保持可控和可复现。
        """
        self.seed = seed
        self.epoch = 0
        self.shard_sampling_schedule = self.generate_shard_sampling_schedule()
        self.curr_shard_index = -1
        self.curr_shard = None
        self._cache_job = None

    def print_dataset_statistics(self):
        """Print formatted dataset statistics for debugging and monitoring.

        【中文】以人类可读的表格形式打印各底层数据集的路径、shard 数量和混合比例，
        【中文】以及涉及到的具身形态集合和数据集数量，便于调试混合配置是否符合预期。
        """
        print("=" * 100)
        print("ShardedMixtureDataset Statistics")
        print("=" * 100)

        # Print header
        print(f"{'Dataset Path':<60} {'Length':<10} {'Mix Ratio':<12}")
        print("-" * 100)

        # Print dataset details
        for i, ds in enumerate(self.datasets):
            dataset_path = str(ds.dataset_path)
            # Truncate long paths for better display
            if len(dataset_path) > 55:
                dataset_path = "..." + dataset_path[-52:]

            length = len(ds)
            mix_ratio = self.weights[i] * 100

            print(f"{dataset_path:<60} {length:<10,} {mix_ratio:<12.2f}")

        # Print additional metadata
        embodiments = set(
            ds.embodiment_tag.value
            for ds in self.datasets
            if hasattr(ds, "embodiment_tag")  # type: ignore
        )
        print(f"Embodiments: {', '.join(sorted(embodiments))}")
        print(f"Number of datasets: {len(self.datasets)}")
        print("=" * 100)

    def get_initial_actions(self):
        """
        Collect initial actions from all datasets.

        Returns:
            Combined list of initial actions from all constituent datasets

        【中文】遍历所有底层数据集，收集它们各自提供的 initial_actions（如果实现了该接口），
        【中文】合并成一个列表返回，用于策略初始化或部署阶段的通用起始动作配置。
        """
        initial_actions = []
        for dataset in self.datasets:
            if hasattr(dataset, "get_initial_actions"):
                initial_actions.extend(dataset.get_initial_actions())  # type: ignore
        return initial_actions
