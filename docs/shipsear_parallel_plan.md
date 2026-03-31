# ShipsEar 并行改造方案（保留 baseline）

## 1) baseline 并行保留原则
- 不改已有 `scripts/pretraining_AS2M.sh`、`scripts/finetuning_ESC50.sh`、`config/pretraining_AS2M.yaml`、`config/finetuning.yaml`。
- 新方案全部采用新增命名：`*_shipsear_light*`。
- baseline 与 light 版可并行训练与评估。

## 2) 创新点 1（已落地第一版）
- 新增轻量骨干：`models/modules.py::LightweightConvAttentionBlock`。
- 挂接入口：`models/EAT_pretraining.py`，通过 `model.backbone_variant=lightweight` 切换。
- 核心思想：Depthwise Separable Conv(局部) + Self-Attention(全局) + 轻量 MLP。
- 配置：
  - `config/pretraining_shipsear_light.yaml`
  - `config/finetuning_shipsear_light.yaml`
- 脚本：
  - `scripts/pretraining_SHIPSEAR_LIGHT.sh`
  - `scripts/finetuning_SHIPSEAR_LIGHT.sh`

## 3) 创新点 2（接入位置与实现草案）
- 推荐接入文件：`models/images.py::ImageEncoder.compute_mask`。
- 方案：新增 `mask_strategy` 配置（`baseline` / `phys_guided`），在 `phys_guided` 下对低频 patch 行提高掩码率或使用条状 mask。
- 建议新增 util：`utils/phys_masking.py`，输出与 `compute_block_mask_2d` 同形状 mask。
- 保留原 `inverse_mask` 流程，默认仍走 baseline。

## 4) 创新点 3（接入位置与实现草案）
- 推荐接入文件：`models/EAT_pretraining.py` 的 loss 汇总段。
- 方案：新增 `physical_loss_weight` 与 `physical_loss_type` 配置。
- 第一版 physical loss：
  - 频带能量摘要损失（低频/中频/高频能量比例约束）
  - 可选 Mel 轮廓摘要 L1
- 只在训练时计算，不增加推理分支。

## 5) 创新点 4（后续预留）
- 新增蒸馏脚本与占位配置：
  - `config/finetuning_shipsear_light_kd.yaml`（后续）
  - `scripts/distill_SHIPSEAR_CNN.sh`（后续）
- teacher 为 light finetune 模型，student 为小 CNN。

## 6) 对比实验建议
- Baseline: `finetuning_SHIPSEAR.sh`
- Light only: `finetuning_SHIPSEAR_LIGHT.sh`
- 后续：Light+Mask, Light+Mask+Phy
- 统计脚本：`utils/profile_eat_model.py`（参数量/模型大小/推理延迟，FLOPs 可选依赖）
