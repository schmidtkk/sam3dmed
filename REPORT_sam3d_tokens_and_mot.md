# SAM3D 源码速览（布局 tokens / 形状 tokens / Mixture-of-Transformer）

> 目的：定位代码中“布局 token (layout tokens)”、"形状 token (shape tokens)" 以及 "混合 Transformer / MoT(MOT)" 实现位置，方便开发者快速跳转与扩展。

---

## 核心问题（总结）
- 形状 token（shape tokens）：作为结构化 latent（SLat）与稀疏 latent（slat、ss）实现存在，核心类为 `Latent` 与 SLat 生成/解码模块。
- 布局/姿态 token（layout / pose tokens）：以 `pose` token 或 pipeline 中的 Pose Decoder/后处理管线体现，`include_pose` 标志可在稀疏模型中追加 pose token。
- Mix-of-Transformer（多模态 Transformer / MOT）：通过 `MOTModulatedTransformerCrossBlock` 与 `MOTMultiHeadSelfAttention` 实现按 latent_name 的多模态 attention，支持按 latent 名称拆分/合并（`latent_share_transformer`）。

---

## 入口 & 快速跳转（在 VS Code 中单击可跳转）
> 说明：下列链接使用相对路径，直接在 VS Code 中单击跳转到指定行。

- 形状 token & 位置信息（Latent）:
  - Latent 定义（`to_input` / `to_output` / pos emb）：
    - [sam3d_objects/model/backbone/tdfy_dit/models/mm_latent.py#L12](sam3d_objects/model/backbone/tdfy_dit/models/mm_latent.py#L12)
  - SLat 生成 / wrapper:
    - SLatFlowModel: [sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py#L77](sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py#L77)
    - SLatFlowModelTdfyWrapper: [sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py#L304](sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_flow.py#L304)
  - SLat Gaussian 解码器:
    - SLatGaussianDecoder: [sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_gs.py#L15](sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/decoder_gs.py#L15)

- 布局 (Pose / layout) token:
  - 稀疏模型开启 pose token 的位置（`include_pose`）:
    - SparseStructureFlowModel（`include_pose` 初始化）: [sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L72](sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L72)
    - `self.include_pose = include_pose` (关键赋值): [sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L115](sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L115)
    - `if self.include_pose:` 用法（模型中增加/解析 pose token 的地方）: [sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L230](sam3d_objects/model/backbone/tdfy_dit/models/sparse_structure_flow.py#L230)
  - Pipeline: Pose 解码器与后处理
    - Pose 解码器 `pose_decoder`（将模型输出映射到 `x_instance_translation/rotation/scale`）: [sam3d_objects/pipeline/inference_utils.py#L224](sam3d_objects/pipeline/inference_utils.py#L224)
    - 布局后处理（优化/ICP/render-fit）: `apply_transform`（layout 后处理）: [sam3d_objects/pipeline/layout_post_optimization_utils.py#L225](sam3d_objects/pipeline/layout_post_optimization_utils.py#L225)

- MoT / 混合 Transformer（MOT）:
  - MOT Transformer block：`MOTModulatedTransformerCrossBlock`（按 latent name 为每个 latent 创建 ModuleDict 的 MSA/MCA/FFN）: [sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py#L174](sam3d_objects/model/backbone/tdfy_dit/modules/transformer/modulated.py#L174)
  - MOT 多模态自注意力：`MOTMultiHeadSelfAttention`（多 latent 的 QKV/输出 ModuleDict，及 `protect_modality_list` 行为）: [sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py#L180](sam3d_objects/model/backbone/tdfy_dit/modules/attention/modules.py#L180)
  - MOT 模型包装器/融合器： `SparseStructureFlowTdfyWrapper`（接收 `latent_mapping` 与 `latent_share_transformer`，提供 `project_input` 与 `project_output`）：
    - MOT wrapper: [sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L173](sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L173)
    - MOT blocks 使用（创建 MOTModulatedTransformerCrossBlock 的位置）: [sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L60](sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L60)
    - 合并/拆分 latent 的工具方法:
      - merge_latent_share_transformer: [sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L256](sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L256)
      - split_latent_share_transformer: [sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L273](sam3d_objects/model/backbone/tdfy_dit/models/mot_sparse_structure_flow.py#L273)

- Embedders / 条件 token:
  - EmbedderFuser（融合多模态 condition tokens & 支持位置索引、压缩、dropout）: [sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py#L10](sam3d_objects/model/backbone/dit/embedder/embedder_fuser.py#L10)
  - 图像 embedder（DINO/CLIP 等）：[sam3d_objects/model/backbone/dit/embedder/dino.py#L10](sam3d_objects/model/backbone/dit/embedder/dino.py#L10)
  - Pointmap embedder（PointPatchEmbed; 内部 CLS / windowed tokens）：[sam3d_objects/model/backbone/dit/embedder/pointmap.py#L11](sam3d_objects/model/backbone/dit/embedder/pointmap.py#L11)

- 稀疏表示 & 布局（SparseTensor、layout）:
  - SparseTensor 定义与 `layout` 机制（如何划分 Batch 内 token）：[sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py#L20](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py#L20)
  - `layout` property（layout 列表为每个 batch 定义 tokens 的 slice）: [sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py#L162](sam3d_objects/model/backbone/tdfy_dit/modules/sparse/basic.py#L162)

- Pipeline 示例（如何在推理中触发一轮 sampling）：
  - 采样 sparse structure: `sample_sparse_structure`（入口，生成 coords、shape_latent）: [sam3d_objects/pipeline/inference_pipeline.py#L642](sam3d_objects/pipeline/inference_pipeline.py#L642)
  - 采样 slat（structured latent）: `sample_slat`（入口，slat_generator 返回 slat）：[sam3d_objects/pipeline/inference_pipeline.py#L721](sam3d_objects/pipeline/inference_pipeline.py#L721)

---

## 如何阅读与扩展（简洁步骤）
1. 查看模型配置（`latent_mapping` 与 `latent_share_transformer`）: 大多数模型由 config/value.json 指明 `latent_mapping`，训练/加载时传入相应 wrapper（示例：mot_sparse_structure_flow wrapper）。建议先打开该 wrapper 的配置以查看 token 命名。
2. 新增 Token（以 shape/new_layout 为例）:
   - 在 `mm_latent.py` 中新增 `Latent` 或调用现有 pos_embedder（`ShapePositionEmbedder` / `LearntPositionEmbedder` 等）。
   - 在 `mot_sparse_structure_flow.py` 的 `latent_mapping` 中添加 name → Latent 的 ModuleDict entry；或在训练 config 中提供该 mapping。
   - 若需共享 Transformer，可在 `latent_share_transformer` 中添加到共享组，wrapper 自动合并/拆分。
   - 若该 token 需要 decoder，请新增对应的 decoder 到 `structured_latent_vae`（或其他 decoder）并在 pipeline 中挂载。
3. 若 token 是条件（非 latent）:
   - 在 `embedder_fuser` 中新增 condition-embedder；并在 pipeline 的 `condition_input_mapping` 加上对应输入 key。
4. 若需控制跨模态 attention 行为，请修改 `MOTMultiHeadSelfAttention.protect_modality_list` 或 `MOTModulatedTransformerCrossBlock` 的 `latent_names` 逻辑。

---

## 参考/备注
- 代码有 “dense token (B, N, D)” 与 “sparse token（`SparseTensor`）” 两种形式，转换通过 `Latent.to_input/to_output` 与 wrapper（`TdfyWrapper`）完成。
- 训练/推理中，pipeline (`inference_pipeline.py`) 负责将 `latent_mapping` 与 `condition embedder` 组合并调用 generator/decoder；对新 token 的修改需要在 pipeline/配置上做配套更新。

---

如需我把上述“如何新增 Token”的步骤自动生成一个最小 patch（比如新增 `latent_mapping['your_token']` 的配置示例与 pipeline patch），我可以继续生成并运行 CI 测试/验证。