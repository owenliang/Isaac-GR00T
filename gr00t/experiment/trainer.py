"""自定义 Trainer，支持简单的性能分析。

这个 HuggingFace Trainer 的子类测量：
1. 数据加载延迟（上一次 training_step 结束到当前 training_step 开始的时间）
2. 前向传播延迟（training_step 内部的耗时，包括模型 forward 和 loss 计算）

每隔 profile_log_interval 步通过 self.log 记录统计信息。
这不是一个完整的 profiler，而是一个轻量级的工具，用于快速确认训练管道的瓶颈是在数据加载还是模型计算。
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Optional

import torch
from transformers.trainer import TRAINER_STATE_NAME, Trainer, TrainerState, get_last_checkpoint
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


class ProfCallback(TrainerCallback):
    """Performance profiling callback for torch.profiler step-by-step triggering.
    【中文】性能分析回调，用于 torch.profiler 的每步触发。
    """
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step.
        【中文】每个训练步骤结束时调用。
        【中文】prof.step() 通知 profiler 进入下一个阶段，profiler 会根据 schedule 自动切换状态。
        【中文】状态循环：wait → warmup → active → wait → ...（没有 stop，自动管理）
        """
        self.prof.step()


class _BatchIterator:
    """轻量级迭代器，产生预拼接的 batch。"""

    def __init__(self, buffer, bs, collator, total_steps):
        self._buffer = buffer
        self._bs = bs
        self._collate = collator
        self._total_steps = total_steps
        self._produced = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self._total_steps

    def __next__(self):
        if self._produced >= self._total_steps:
            raise StopIteration

        # Fast path – single lock acquisition inside ``sample_batch``.
        batch_samples = self._buffer.sample_batch(self._bs)  # type: ignore[attr-defined]
        self._produced += 1
        return self._collate(batch_samples)


class _PrefetchIterator:
    """后台预取迭代器，在单独的线程中预先拼接 batch。"""
    def __init__(self, buffer, bs, collate_fn, total_steps):
        self.buffer = buffer
        self.bs = bs
        self.collate = collate_fn
        self.total = total_steps
        self.produced = 0

        self._q = queue.Queue(maxsize=4)
        self._stop = False

        # Start background worker
        self._worker = threading.Thread(target=self._fill)
        self._worker.daemon = True
        self._worker.start()

    def _fill(self):
        while not self._stop:
            if self.produced + self._q.qsize() >= self.total:
                break
            # block if queue is full
            samples = self.buffer.sample_batch(self.bs)
            batch = self.collate(samples)
            self._q.put(batch)

    def __iter__(self):
        return self

    def __len__(self):
        return self.total

    def __next__(self):
        if self.produced >= self.total:
            self._stop = True
            # in case worker is blocked on put()
            raise StopIteration
        batch = self._q.get()  # this will block until the next batch is ready
        self.produced += 1
        return batch


def _batch_accuracy(
    preds: torch.Tensor, labels: torch.Tensor, action_offset: Optional[int] = None
) -> torch.Tensor:
    """
    计算 token 级别的准确率，忽略 -100 标签位置。
    
    Args:
        preds: 预测 token ID，形状 (batch, seq_len)
        labels: Ground-truth label ID，形状与 preds 相同
    
    Returns:
        当前 batch 中正确预测的标签比例
    """
    # causal prediction：位置 i 预测位置 i+1
    # Shift so that tokens < n predict n
    # https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py#L60
    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # 忽略 -100 标签位置（HuggingFace 约定）
    mask = labels != -100

    if action_offset is not None:
        # 将标签偏移到 action token 范围，普通 token 为负值
        labels = labels - action_offset

    correct = (preds == labels) & mask

    # Avoid division by zero for empty masks (should not happen in practice)
    denom = mask.sum().clamp(min=1)
    accuracy = correct.sum().float() / denom.float()
    return accuracy


# 全局变量：用于 batch 评估指标的累计
_eval_accuracy_accumulated_correct = 0
_eval_accuracy_accumulated_total = 0


def compute_eval_accuracy(
    eval_pred: EvalPrediction, compute_result: bool, action_offset: Optional[int] = None
):
    """计算评估准确率，支持批量累计。"""
    logits = eval_pred.predictions[0]
    if action_offset is not None:
        logits = logits[..., action_offset:]
    preds = logits.argmax(axis=-1)
    labels = eval_pred.label_ids

    preds = preds[:, :-1]
    labels = labels[:, 1:]

    # Ignore positions with label == -100 (HF convention)
    mask = labels != -100

    if action_offset is not None:
        # we offset the labels to the action tokens range, with normal tokens in the negatives
        labels = labels - action_offset

    correct = ((preds == labels) & mask).sum()
    total = mask.sum()

    global _eval_accuracy_accumulated_correct, _eval_accuracy_accumulated_total
    _eval_accuracy_accumulated_correct += correct
    _eval_accuracy_accumulated_total += total

    if compute_result:
        accuracy = _eval_accuracy_accumulated_correct / max(_eval_accuracy_accumulated_total, 1)
        _eval_accuracy_accumulated_correct = 0
        _eval_accuracy_accumulated_total = 0
        return {"eval_accuracy": accuracy}
    else:
        return {}


class Gr00tTrainer(Trainer):
    """
    自定义 Trainer，绕过 torch dataloader，并使 data collator 异步化。
    关键特性：
    - 直接从 shard-based 数据集采样，无需 PyTorch DataLoader
    - 支持从 checkpoint 恢复时重置随机种子
    - 每步计算并记录 token 级别的准确率
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """初始化 Trainer。"""
        self.action_offset = kwargs.pop("action_offset", None)
        self.multiprocessing_context = kwargs.pop("multiprocessing_context", "fork")
        super().__init__(
            *args,
            **kwargs,
            # compute_metrics=partial(compute_eval_accuracy, action_offset=self.action_offset),
        )

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # 隐藏 epoch 信息（对 Iterable datasets 不准确）
        epoch = self.state.epoch
        self.state.epoch = None
        super().log(logs, start_time=start_time)
        self.state.epoch = epoch

    def get_train_dataloader(self):
        """
        返回训练 dataloader。
        关键设计：
        - 从 checkpoint 恢复时，不 skip 数据，而是重置随机种子
        - 使用 persistent_workers 提高效率
        """

        # Fall back to default behaviour if not using the custom buffer.
        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            # 恢复训练时重置种子，使数据顺序不同
            new_seed = self.train_dataset.seed + curr_global_step
            self.train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # 处理 IterableDataset 情况
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )
        # 如果 num_workers > 0，使用 persistent workers
        persistent_workers = self.args.dataloader_num_workers > 0

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        # multiprocessing_context 仅在 num_workers > 0 时可用
        if self.args.dataloader_num_workers > 0:
            dataloader_params["multiprocessing_context"] = self.multiprocessing_context

        return torch.utils.data.DataLoader(self.train_dataset, **dataloader_params)

    def train(
        self,
        resume_from_checkpoint=None,
        **kwargs,
    ):
        """Training entry point.
        Properly handle checkpoint resumption: load self.state from checkpoint,
        so that get_train_dataloader can read global_step.
        
        【中文】训练入口。
        【中文】正确处理 checkpoint 恢复：从 checkpoint 加载 self.state，
        【中文】以便 get_train_dataloader 可以读取 global_step。
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # 【中文】如果传入 True，则自动查找 output_dir 中的最新 checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                logging.warning(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        # 【中文】如果找到了 checkpoint，加载训练状态（包含 global_step、epoch 等）
        if resume_from_checkpoint is not None:
            logging.info(f"Resuming from checkpoint {resume_from_checkpoint}")
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            # 【中文】加载 trainer_state.json，包含训练进度信息
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )

        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    # ------------------------------------------------------------------
    # Loss / accuracy 计算重载
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):  # type: ignore[override]
        """
        计算 loss 并记录 token 级别的准确率。
        
        关键逻辑：
        - 委托给父类 compute_loss 计算 loss
        - 每隔 logging_steps 计算并记录 accuracy
        - 支持分布式训练时的 accuracy 聚合
        """

        # 使用父类实现保留内置功能
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        # import ipdb; ipdb.set_trace()
        # # save the model's embedding for the first step
        # input_embeddings = model.get_input_embeddings().weight.data.cpu()
        # output_embeddings = model.get_output_embeddings().weight.data.cpu()
        # torch.save(input_embeddings, f"input_embeddings_{self.state.global_step}.pt")
        # torch.save(output_embeddings, f"output_embeddings_{self.state.global_step}.pt")

        # 记录最后一次 loss（用于测试）
        self.loss = loss

        # --------------------------------------------------------------
        # 准确率计算
        # --------------------------------------------------------------
        if (
            self.state.global_step % self.args.logging_steps == 0
            and model.training
            and "labels" in inputs
        ):
            if self.action_offset is not None:
                preds = outputs.logits.detach()[:, :, self.action_offset :].argmax(dim=-1).cpu()
            else:
                preds = outputs.logits.detach().argmax(dim=-1).cpu()
            with torch.no_grad():
                acc_local = _batch_accuracy(
                    preds, inputs["labels"].to(device=preds.device), self.action_offset
                )
            acc_tensor = torch.tensor(acc_local.item(), device=loss.device)
            acc_mean = self._nested_gather(acc_tensor).mean().item()

            if self.args.local_rank in (-1, 0):
                self.log({"train_accuracy": acc_mean})

        return (loss, outputs) if return_outputs else loss
