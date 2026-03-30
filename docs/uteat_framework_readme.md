# UT-EAT v1 框架说明（baseline 并行保留）

## 1) baseline 与 UT-EAT 并行
- baseline 保持不变：`config/finetuning_shipsear.yaml`, `config/finetuning_deepship.yaml` 及对应脚本。
- UT-EAT 新增命名：`*_uteat.yaml` / `*_UTEAT.sh`。

## 2) 三条核心改造映射

### A. 轻量骨干（已实现）
- 开关：`model.backbone_variant`
  - `baseline`：原始 AltBlock
  - `uteat`：轻量 Conv+Attention block（复用已存在 lightweight 实现）
- 位置：`models/EAT_pretraining.py` + `models/modules.py`

### B. 物理引导掩码（已实现 v1）
- 开关：`model.modalities.image.mask_strategy`
  - `baseline`
  - `uteat_phys`
- v1 策略：低频区域额外掩码概率增强（规则驱动）
- 位置：`models/images.py::compute_mask`

### C. 物理约束目标（已实现 v1）
- 新增项：
  - `model.physical_loss_weight`
  - `model.physical_num_bands`
- 在原 UFO 汇总上增加 coarse band-energy consistency loss
- 位置：`models/EAT_pretraining.py`

## 3) 训练入口
- ShipsEar 预训练：`scripts/pretraining_SHIPSEAR_UTEAT.sh`
- DeepShip 预训练：`scripts/pretraining_DEEPSHIP_UTEAT.sh`
- ShipsEar 微调：`scripts/finetuning_SHIPSEAR_UTEAT.sh`
- DeepShip 微调：`scripts/finetuning_DEEPSHIP_UTEAT.sh`

## 4) profile
- 直接复用：`utils/profile_eat_model.py`

## 5) 对比实验建议
1. Baseline EAT (ShipsEar/DeepShip)
2. UT-EAT backbone only（mask_strategy=baseline, physical_loss_weight=0）
3. UT-EAT + phys mask（mask_strategy=uteat_phys）
4. UT-EAT + phys mask + phys loss（physical_loss_weight>0）
