# compare_v1 实验链（ShipsEar / DeepShip, EAT-base vs UT-EAT）

## 目标
- 基线：`EAT-base_epoch30_pt.pt` 初始化后分别在 ShipsEar/DeepShip 微调。
- 对比：UT-EAT 在两个数据集上评测与 profile（有 checkpoint 则复用，无则最小重跑）。
- 统一输出：`/hy-tmp/exp/eat/results/compare_v1`。

## 一键入口
```bash
bash /hy-tmp/eat_work/EAT/scripts/compare_v1/run_compare_v1.sh 0 8
```
参数：
- `$1` GPU ID，默认 `0`
- `$2` throughput 的首选 batch size，默认 `8`（OOM 自动回退到 `4`）

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
