# 原始 EAT Baseline（DeepShip）运行说明

> 本文档仅对应“原始 EAT baseline on DeepShip”，用于第四章第一阶段对照组构建。

## 1. manifest 生成
```bash
python /hy-tmp/eat_work/EAT/utils/prepare_deepship_manifest.py \
  --data-root /hy-tmp/DeepShip_16kHz_10s \
  --output-dir /hy-tmp/exp/eat/manifests/deepship \
  --seed 42
```

输出文件：
- `train.tsv`, `valid.tsv`, `test.tsv`
- `train.lbl`, `valid.lbl`, `test.lbl`
- `label_descriptors.csv`, `label_mapping.json`

## 2. baseline 训练
```bash
bash /hy-tmp/eat_work/EAT/scripts/finetuning_DEEPSHIP.sh 0 8 6
```

## 3. profile
```bash
python /hy-tmp/eat_work/EAT/utils/profile_eat_model.py \
  --checkpoint /hy-tmp/exp/eat/runs/deepship_baseline/checkpoint_best.pt \
  --target-length 1024 --batch-size 1 --warmup 10 --iters 30
```

## 4. 默认保守参数（RTX 3060 12GB）
- micro batch: 8
- update_freq: 6
- valid subset: `valid`
- allocator hint: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
