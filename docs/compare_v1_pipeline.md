# compare_v1 实验链（ShipsEar / DeepShip, EAT-base vs UT-EAT）

## 目标
- 基线：`EAT-base_epoch30_pt.pt` 初始化后分别在 ShipsEar/DeepShip 微调。
- 对比：UT-EAT 在两个数据集上评测与 profile（有 checkpoint 则复用，无则最小重跑）。
- 统一输出：`/hy-tmp/exp/eat/results/compare_v1`。

## 一键入口
```bash
bash /hy-tmp/eat_work/EAT/scripts/compare_v1/run_compare_v1.sh 0 8 4000
```
参数：
- `$1` GPU ID，默认 `0`
- `$2` throughput 的首选 batch size，默认 `8`（OOM 自动回退到 `4`）
- `$3` baseline(EAT-base) 的 `max_update`，默认 `4000`

## 分条运行（推荐）
1) 仅训练 ShipsEar / EAT-base（max_update=4000）：
```bash
bash /hy-tmp/eat_work/EAT/scripts/finetuning_SHIPSEAR.sh 0 8 6 4000
```

2) 仅训练 DeepShip / EAT-base（max_update=4000）：
```bash
bash /hy-tmp/eat_work/EAT/scripts/finetuning_DEEPSHIP.sh 0 8 6 4000
```

3) 已有 checkpoint 后，再统一跑评测/profile/汇总：
```bash
bash /hy-tmp/eat_work/EAT/scripts/compare_v1/run_compare_v1.sh 0 8 4000
```

## UT-EAT 预训练 -> 微调（建议顺序）
### ShipsEar
```bash
# 1) smoke pretrain (100 updates)
bash /hy-tmp/eat_work/EAT/scripts/pretraining_SHIPSEAR_UTEAT_SMOKE.sh 0 8 2 1
# 2) formal pretrain (10000 updates)
bash /hy-tmp/eat_work/EAT/scripts/pretraining_SHIPSEAR_UTEAT_FORMAL.sh 0 8 2 1
# 3) finetune (4000 updates)
bash /hy-tmp/eat_work/EAT/scripts/finetuning_SHIPSEAR_UTEAT.sh 0 8 6 4000
```

### DeepShip
```bash
# 4) smoke pretrain (100 updates)
bash /hy-tmp/eat_work/EAT/scripts/pretraining_DEEPSHIP_UTEAT_SMOKE.sh 0 8 2 1
# 5) formal pretrain (10000 updates)
bash /hy-tmp/eat_work/EAT/scripts/pretraining_DEEPSHIP_UTEAT_FORMAL.sh 0 8 2 1
# 6) finetune (4000 updates)
bash /hy-tmp/eat_work/EAT/scripts/finetuning_DEEPSHIP_UTEAT.sh 0 8 6 4000
```

## 目录结构
```text
/hy-tmp/exp/eat/results/compare_v1/
  shipsear/
    eat_base/{train,eval,profile}
    ut_eat/{eval,profile}
  deepship/
    eat_base/{train,eval,profile}
    ut_eat/{eval,profile}
  summary/
    compare_main.csv
    compare_main.md
```

## 结果文件
- eval: `eval/eval_metrics.json`
  - dataset, model, init_checkpoint, finetuned_checkpoint, accuracy, macro_f1, precision_macro, recall_macro
- profile: `profile/profile_metrics.json`
  - dataset, model, finetuned_checkpoint, trainable_params, checkpoint_size_mb, flops, latency_ms_per_sample, throughput_samples_per_sec, batch_size_for_throughput, device, precision_mode

## 协议
- latency: batch=1, preloaded input, `torch.no_grad()`, warmup + iters, CUDA sync, FP32, 无 dataloader/I/O 计入。
- throughput: 优先 batch=8，OOM 回退到 4。
- FLOPs: 统一使用 `fvcore.FlopCountAnalysis(model, (source,)).total()`，source 为同一协议下的 batch=1 预加载输入。

## UT-EAT checkpoint 约定路径
- ShipsEar finetune: `/hy-tmp/exp/eat/runs/shipsear_uteat/checkpoint_best.pt`（或 `checkpoint_last.pt`）
- DeepShip finetune: `/hy-tmp/exp/eat/runs/deepship_uteat/checkpoint_best.pt`（或 `checkpoint_last.pt`）

若不存在，脚本会在对应目录写入 `retrain_required.txt`，给出最小重跑命令。

## DeepShip UT-EAT 8000-step 最小追加实验（不覆盖 4000-step）
```bash
# 独立目录启动 8000-step（推荐从 formal pretrain 干净重启）
bash /hy-tmp/eat_work/EAT/scripts/finetuning_DEEPSHIP_UTEAT.sh \
  0 8 6 8000 1 /hy-tmp/exp/eat/runs/deepship_uteat_8k

# 用 compare_v1 复用评测/profile/summary，但只覆盖 deepship/ut_eat 输入 checkpoint
DEEP_UT_CKPT_OVERRIDE=/hy-tmp/exp/eat/runs/deepship_uteat_8k/checkpoint_best.pt \
bash /hy-tmp/eat_work/EAT/scripts/compare_v1/run_compare_v1.sh 0 8 4000
```

## ShipsEar 4.3.3 主实验数据导出与绘图
```bash
bash /hy-tmp/eat_work/EAT/scripts/compare_v1/run_shipsear_main_analysis.sh
```

默认输入：
- baseline ckpt: `/hy-tmp/exp/eat/runs/shipsear_baseline/checkpoint_best.pt`
- UT-EAT ckpt: `/hy-tmp/exp/eat/runs/shipsear_uteat/checkpoint_best.pt`
- compare 主表: `/hy-tmp/exp/eat/results/compare_v1/summary/compare_main.csv`
- pretrain 日志: `/hy-tmp/exp/eat/results/compare_v1/shipsear_formal_pretrain_2026-03-31_164952.log`
- baseline finetune 日志: `/hy-tmp/exp/eat/runs/shipsear_baseline/train.log`
- UT-EAT finetune 日志: `/hy-tmp/exp/eat/results/compare_v1/shipsear_uteat_finetune_2026-04-01_005407.log`

默认输出：
- 数据: `/hy-tmp/exp/eat/results/compare_v1/shipsear/analysis`
- 图: `/hy-tmp/exp/eat/results/compare_v1/figures`
