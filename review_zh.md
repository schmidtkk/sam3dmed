# LoRA 微调实现综述（中文）

✅ 摘要

- 为 SAM3D 的医学图像（仅训练 Mesh 解码器）实现了 LoRA（低秩适配）支持。
- 关键文件：`sam3d_objects/model/lora.py` —— 包含 LoRA 封装和一系列工具函数。
- 训练集成：在 `scripts/train_medical.py` 的 `MedicalTrainer._setup_lora()` 中调用 `setup_lora_for_medical_finetuning()`。
- 检查点管理会保存 LoRA-only 权重（通过 `get_lora_state_dict` 与 `load_lora_state_dict`），以减小检查点体积。
- 测试：`tests/test_lora.py` 覆盖注入、合并、冻结、保存/加载和参数计数等功能。

---

## 重构/新增内容

1. `sam3d_objects/model/lora.py`
   - 新增的 LoRA 工具模块，包含：
     - `LoRALinear`：将 `nn.Linear` 封装为添加 LoRA 路径的模块。
     - `_find_modules_by_name`：根据名称模式（字符串匹配）查找目标 Linear 层。
     - `inject_lora`：在模型中就地注入 LoRA（替换部分 `nn.Linear` 为 `LoRALinear`）。
     - `freeze_base_params`：冻结基础参数（只有 LoRA 的参数可训练）。
     - `get_lora_params`：返回 LoRA 要训练的参数，用于构建优化器。
     - `get_lora_state_dict` / `load_lora_state_dict`：仅保存/加载 LoRA 权重，保持检查点小而干净。
     - `merge_lora_weights`：把 LoRA 的 delta 合并到基础权重中，并替换 LoRA 模块为普通 `nn.Linear`（用于推理）。
     - `count_parameters`：统计参数数量（总数、可训练、LoRA）。
     - `setup_lora_for_medical_finetuning`：一键注入 LoRA、冻结基础参数并可选解冻输出层的小工具。

2. `scripts/train_medical.py`
   - `MedicalTrainer._setup_lora()` 现在调用 `setup_lora_for_medical_finetuning()`，实现一致的 LoRA 设置流程。
   - 训练使用 `trainable_params = [p for p in self.model.parameters() if p.requires_grad]`，以保证优化器只管理 LoRA 参数（或额外被解冻的输出层）。
   - 检查点保存会持久化 `lora_state_dict`，加载检查点会使用 `load_lora_state_dict()` 恢复 LoRA 权重。

3. 管道与文档调整
   - `scripts/run_medical_pipeline.sh`：简化为直接调用 `scripts/train_medical.py --use_dataset` 而非内联 Python。
   - `markdown/quickstart_training.md`：新增快速启动文档，示例命令演示预处理、训练和评估流程。

4. 单元测试
   - `tests/test_lora.py` 增加/完善：
     - `LoRALinear` 初始化、前向测试与 `merge_weights` 功能。
     - `inject_lora` 的注入和行为验证。
     - `freeze_base_params` 确保只有 LoRA 参数被训练。
     - `get_lora_state_dict` 与 `load_lora_state_dict` 的 round-trip 测试。
     - `merge_lora_weights` 的输出与 LoRA 模式一致性（合并前后保持输出相等）。

---

## 设计与实现细节

### LoRALinear
- 封装一个被冻结的 `nn.Linear`，并添加一个低秩路径：
  - `A` 矩阵形状为 (rank, in_features)
  - `B` 矩阵形状为 (out_features, rank)
- 计算方式：
  - `out = W x + (alpha / r) * B @ (A @ x)`（可选 LoRA dropout）
- 初始化方式：
  - `A` 使用 kaiming_uniform 初始值（更稳定），`B` 初始化为 0，以确保注入后初始行为等于基础线性变换。
- 支持：
  - `merge_weights()`：返回合并后的 `nn.Linear`（用于推理）；
  - `weight` 属性：动态计算合并权重（但不修改模型本体，除非显式调用 `merge_weights()`）。

### 注入/发现
- `_find_modules_by_name` 会递归检查命名子模块，以字符串 `in` 匹配目标名称，比如 `to_qkv` 或 `to_out`。
- `inject_lora` 将匹配的 `nn.Linear` 替换为 `LoRALinear`，在模型结构和命名上依赖一致性。

### 冻结 / 优化器
- `freeze_base_params` 将所有参数 `requires_grad=False`，仅保留 `lora_A` 与 `lora_B` 可训练。
- 使用 `get_lora_params` 构建优化器，仅包含 LoRA 参数。

### 检查点
- `get_lora_state_dict` 仅提取 LoRA 参数以便存储小型检查点；`load_lora_state_dict` 也只加载 LoRA 部分（当 LoRA 已注入时）。
- `MedicalTrainer` 的 `save_checkpoint`/`load_checkpoint` 通过这些 API 管理 LoRA 部分的保存与恢复。

### 推理合并
- `merge_lora_weights` 会在模型上就地替换 `LoRALinear` 模块为合并后的 `nn.Linear`，从而在推理上无需 LoRA 开销。

---

## 集成点

- `scripts/train_medical.py`：
  - `MedicalTrainer._setup_lora()` 使用 `setup_lora_for_medical_finetuning()`，从而使训练代码与 LoRA 的注入和冻结一致。
  - 优化器和检查点代码使用 `get_lora_state_dict` / `load_lora_state_dict`。

- `scripts/eval_medical.py`：
  - evaluator 在加载 LoRA-only 检查点时，会注入 LoRA（`inject_lora`）并载入 LoRA 权重（`load_lora_state_dict`），或可以使用合并后的模型。

- `TS_SAM3D_Dataset`, pipeline：
  - 未改变数据定义；`TS_SAM3D_Dataset` 仍对训练/验证提供样本及 `data_collate`。

---

## 测试
- `tests/test_lora.py` 覆盖上述 LoRA 基本功能与边缘情况。
- `tests/test_train_medical.py` 验证 Trainer 的 LoRA 配置、冻结、检查点和训练循环行为。
- 当前 repo 所有测试均通过（共 69 个测试），涵盖 LoRA 功能。

---

## 风险与边界情况

- 名称匹配注入方法依赖模型层命名的一致性（`to_qkv`, `to_out` 等）。如果模型的属性或命名变更，LoRA 可能无法注入到预期层。
- `load_lora_state_dict` 默认为宽松模式（不会强制检查所有键），这便于向后兼容，但可能掩盖参数维度或命名不匹配的问题；建议增加 `strict` 模式选项以便在必要时抛出错误。
- `merge_lora_weights` 就地修改模型，可能不适合训练时使用；建议提供 `inplace=False` 的选项返回合并模型副本以增强安全性。
- 保存检查点使用 PyTorch 默认序列化（`weights_only=False`），未来版本中可能更安全（`weights_only=True`）；在不受信任的上下文下，需注意安全性。

---

## 改进建议

1. `inject_lora` 支持更稳健和可配置的模块选择（如通过明确配置或回调）以替代纯字符串匹配。
2. 支持 per-module LoRA 配置（为不同层指定不同 rank）以获得更灵活的能力分配。
3. 增加 LoRA 的训练-合并-推理的整合测试（端到端）以防回归。
4. 提供 `merge_lora_weights(inplace=False)` 的副本模式以便更安全地将 LoRA 转为合并权重用于推理。
5. 为 `load_lora_state_dict` 增加 `strict` 模式（缺失/不匹配时抛错），便于早期捕获问题。
6. 提供一个“导出合并后模型”脚本示例，包含进入 ONNX/torchscript 的建议步骤。

---

## 示例用法（典型场景）

1. 注入 LoRA 并冻结基础参数（Trainer 会自动处理）:
```python
from sam3d_objects.model.lora import setup_lora_for_medical_finetuning
setup_lora_for_medical_finetuning(model, rank=4, alpha=8, dropout=0.0, unfreeze_output_layers=True)
```

2. 使用仅训练的参数创建优化器:
```python
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
```

3. 保存 LoRA-only 检查点:
```python
from sam3d_objects.model.lora import get_lora_state_dict
ckpt = {
    'epoch': 10,
    'lora_state_dict': get_lora_state_dict(model)
}
torch.save(ckpt, 'lora_checkpoint.pt')
```

4. 加载 LoRA-only 检查点 (先注入 LoRA):
```python
from sam3d_objects.model.lora import inject_lora, load_lora_state_dict
inject_lora(model, target_modules=['to_qkv', 'to_out'], rank=4, alpha=8)
checkpoint = torch.load('lora_checkpoint.pt', map_location='cpu')
load_lora_state_dict(model, checkpoint['lora_state_dict'])
```

5. 合并 LoRA 权重以便推理:
```python
from sam3d_objects.model.lora import merge_lora_weights
merge_lora_weights(model)  # 替换 LoRA 模块为合并的 nn.Linear
```

---

## 总结
- LoRA 的实现模块化、测试充分，并集成到训练、评估与检查点流程中。
- 实现简洁，不会过度耦合 Trainer 和 LoRA utility；便于扩展与移植。
- 推荐添加更加强健的模块选择方法、严格状态加载选项，以及合并模型的副本接口以增强可靠性与可移植性。

如果需要，我可以：
- 将 `load_lora_state_dict` 增加 `strict=True` 选项，并在测试中覆盖该路径；
- 添加 `setup_lora_for_medical_finetuning()` 的 per-layer 自定义配置；
- 以及在 README 中提供 LoRA 导出、合并与部署示例。

请选择你想进行的下一步，我可以继续实现其中一项改进。