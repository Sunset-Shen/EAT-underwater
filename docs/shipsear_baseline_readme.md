# 原始 EAT Baseline（ShipsEar）归档说明

> 本文档仅对应“原始 EAT baseline on ShipsEar”，作为第四章后续 LA-EAT 改造的对照组。

## 1. 数据与配置
- manifest: `/hy-tmp/exp/eat/manifests/shipsear`
- 配置: `config/finetuning_shipsear.yaml`
- 训练脚本: `scripts/finetuning_SHIPSEAR.sh`
- 初始化权重: `/hy-tmp/model_zoo/eat_checkpoints/EAT-base_epoch30_pt.pt`

## 2. 当前已知输出（服务器已有）
- run 目录: `/hy-tmp/exp/eat/runs/shipsear_baseline`
- best checkpoint: `/hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_best.pt`
- latest checkpoint: `/hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_last.pt`

## 3. 复现实验命令（单卡 12GB 默认）
```bash
bash /hy-tmp/eat_work/EAT/scripts/finetuning_SHIPSEAR.sh 0 8 6
```

## 4. 模型 profile 命令
```bash
python /hy-tmp/eat_work/EAT/utils/profile_eat_model.py \
  --checkpoint /hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_best.pt \
  --target-length 1024 --batch-size 1 --warmup 10 --iters 30
```

## 5. 论文定位
- 该实验是第四章第一阶段“原始 EAT 水声迁移基线（ShipsEar）”。
- 后续 lightweight / masking / physical loss 等方法均需与该基线对比。
