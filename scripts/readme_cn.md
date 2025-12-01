# SAM3D 医学图像微调流程

医学图像分割完整教程：SAM3D + LoRA

## 数据集结构

本教程使用 **TS (TotalSegmentator) 心脏数据集**，位于：
```
/mnt/nas1/disk01/weidongguo/dataset/TS
```

**数据集组织**：
```
TS/
├── TS_heart_cropped_resize_train/  (596 个病例)
│   ├── s0331/
│   │   ├── s0331-image.nii.gz   # 3D CT 体数据 (64x64x64)
│   │   └── s0331-label.nii.gz   # 多类分割标签 (6类: 0-5)
│   ├── s0332/
│   └── ...
├── TS_heart_cropped_resize_test/   (150 个病例)
│   ├── s0004/
│   │   ├── s0004-image.nii.gz
│   │   └── s0004-label.nii.gz
│   └── ...
└── TS_heart_cropped_resize/        (总共 746 个病例)
```

**体数据属性**：
- **尺寸**：64×64×64 体素（各向同性）
- **间距**：3mm 各向同性（来自仿射矩阵）
- **类别**：6 类（背景=0，5 个心脏结构）
- **格式**：压缩 NIfTI (.nii.gz)

## 环境准备

激活 conda 环境：
```shell
conda activate sam3d-objects
```

确认 CUDA 可用：
```shell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 步骤 1：下载预训练权重

从 Hugging Face 下载 SAM3D 预训练模型：
```shell
python scripts/download_hf_checkpoints.py \
    --repo facebook/sam-3d-objects \
    --out checkpoints/hf
```

**注意**：如果仓库需要授权，请先登录：
```shell
export HF_TOKEN=你的token
# 或使用: huggingface-cli login
```

预期在 `checkpoints/hf/` 下包含：
- `ss_encoder.safetensors` - SAM3D 图像编码器
- `ss_generator.ckpt` - SAM3D 生成器
- `slat_decoder_*.ckpt` - SLAT 解码器（mesh/gaussian）

## 步骤 2：数据预处理

将原始 NIfTI 医学影像转换为预处理缓存：
```shell
# 处理 TS 心脏数据集（训练集）
python scripts/reprocess_ts_nifti.py \
    --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_train \
    --out ./dataset/ts_processed \
    --classes 5 \
    --extract_mesh \
    --spacing 3.0

# 处理测试集
python scripts/reprocess_ts_nifti.py \
    --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_test \
    --out ./dataset/ts_processed_test \
    --classes 5 \
    --extract_mesh \
    --spacing 3.0
```

**关键参数**：
- `--original_nifti_dir`: 原始 `.nii` 或 `.nii.gz` 文件目录（每个病例在子文件夹中包含 `*-image.nii.gz` 和 `*-label.nii.gz`）
- `--out`: 输出预处理后的 `.npz` 切片缓存目录
- `--classes`: 前景分割类别数（TS 心脏数据集为 5 类：左心房、左心室、右心房、右心室、心肌）
- `--extract_mesh`: 使用 marching cubes 提取表面网格
- `--spacing`: 各向同性体素间距（mm），TS 心脏数据集为 3.0

**输出结构**：
```
dataset/ts_processed/
├── s0001.nii_axis0_slice0050.npz
├── s0001.nii_axis1_slice0128.npz
├── s0002.nii_axis2_slice0064.npz
└── ...
```

每个 `.npz` 包含：`image`, `mask`, `pointmap`, `affine`, `slice_idx`, `axis`, `gt_sdf_path`

## 步骤 3：LoRA 微调训练

### 快速开始（简化脚本）：
```shell
./scripts/run_medical_pipeline.sh \
    --gpu 1 \
    --batch_size 4 \
    --epochs 50 \
    --preprocess_crop_size 256,256
```

### 完全控制（Hydra 配置）：
```shell
python scripts/train_medical_hydra.py \
    data.preprocess_crop_size=[256,256] \
    training.epochs=50 \
    training.batch_size=4 \
    lora.rank=8 \
    lora.alpha=16
```

**常用参数**：
- `--gpu <id>`: GPU 设备编号
- `--batch_size <n>`: 批次大小（默认 4）
- `--epochs <n>`: 训练轮数（默认 50）
- `--preprocess_crop_size H,W`: 归一化切片尺寸为 (H, W)
- `--lora_rank <n>`: LoRA 秩（默认 8）
- `--resume <path>`: 从检查点恢复训练

**恢复训练**：
```shell
./scripts/run_medical_pipeline.sh \
    --gpu 1 \
    --resume checkpoints/medical/epoch_10.pt \
    --epochs 100
```

**训练输出**：
- `checkpoints/medical/best.pt` - 最佳模型（最低损失）
- `checkpoints/medical/epoch_*.pt` - 各轮次检查点
- `outputs/YYYY-MM-DD/*/train.log` - 训练日志

## 步骤 4：模型评估

在测试集上评估训练好的模型：
```shell
python scripts/eval_medical.py \
    --checkpoint checkpoints/medical/best.pt \
    --data_root dataset/ts_processed \
    --output_dir results/evaluation \
    --batch_size 8
```

**计算指标**：
- **Dice 系数**：体积重叠度（越高越好，0-1）
- **HD95**：Hausdorff 距离 95 分位数（越低越好，mm）
- **Chamfer 距离**：表面重建质量（越低越好）
- **Surface Dice**：1mm 容差下的边界准确度

**输出**：
- `results/evaluation/metrics.json` - 逐案例和总体指标
- `results/evaluation/visualizations/` - 预测叠加图（如果指定 `--save_viz`）

## 步骤 5：结果可视化

可视化微调模型的预测结果：
```shell
python scripts/visualize_finetuned.py \
    --lora_checkpoint checkpoints/medical/best.pt \
    --image /path/to/test_image.png \
    --mask_dir /path/to/masks \
    --mask_index 0 \
    --output_dir results/visualizations
```

**医学体数据可视化**：
```shell
python scripts/visualize_comparison.py \
    --checkpoint checkpoints/medical/best.pt \
    --volume /path/to/test.nii.gz \
    --slice_idx 64 \
    --axis 2 \
    --output results/comparison.png
```

**输出**：
- PNG 预测叠加图
- 3D 网格 `.obj` 文件（如适用）
- 逐切片对比网格图

## 故障排查

**DataLoader 尺寸错误**：
```
RuntimeError: stack expects each tensor to be equal size
```
→ 增大 `--preprocess_crop_size` 或重新运行预处理以保证尺寸一致

**CUDA 显存不足**：
→ 减小 `--batch_size` 或 `--preprocess_crop_size`

**nvdiffrast 警告**：
```
Cannot import nvdiffrast
```
→ 可选的 GPU 光栅化依赖，可安全忽略

**缺少检查点文件**：
→ 确保步骤 1 成功完成且文件存在于 `checkpoints/hf/`

## 进阶：Hydra 配置

`configs/train.yaml` 中的所有训练参数均可通过命令行覆盖：
```shell
python scripts/train_medical_hydra.py \
    data.slice_cache_dir=dataset/custom \
    data.preprocess_crop_size=[192,192] \
    training.learning_rate=1e-4 \
    training.weight_decay=0.01 \
    lora.rank=16 \
    lora.alpha=32 \
    loss.w_mask_loss=1.0 \
    loss.w_sdf_loss=0.5
```

完整参数列表见 `configs/train.yaml`。

---

**快速参考**：
```shell
# 1. 下载预训练权重
python scripts/download_hf_checkpoints.py --repo facebook/sam-3d-objects --out checkpoints/hf

# 2. 数据预处理（TS 心脏数据集）
python scripts/reprocess_ts_nifti.py --original_nifti_dir /mnt/nas1/disk01/weidongguo/dataset/TS/TS_heart_cropped_resize_train --out dataset/ts_processed --classes 5 --spacing 3.0

# 3. 训练
./scripts/run_medical_pipeline.sh --gpu 0 --batch_size 4 --epochs 50 --preprocess_crop_size 256,256

# 4. 评估
python scripts/eval_medical.py --checkpoint checkpoints/medical/best.pt --data_root dataset/processed --output_dir results

# 5. 可视化
python scripts/visualize_finetuned.py --lora_checkpoint checkpoints/medical/best.pt --image test.png --output_dir viz
```
