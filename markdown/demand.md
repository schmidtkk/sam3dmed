**What do we want to do?**
Fine-tune **SAM3D-Object** to adapt it to the medical imaging domain (CT, MRI, Ultrasound). Our goal: when the user clicks on a certain **2D slice** in a medical image, the model can output the **complete 3D shape** of the corresponding organ or tissue.

**Why use SAM3D-Object?**
The reconstruction of organ shapes in the human body is highly valuable for clinical applications. Since SAM-3D is pretrained on large and diverse datasets, we want to evaluate its **transfer learning capability** in medical imaging.

**Task workflow?**

* **Simplest pipeline**:
  Using the heart as an example.
  A user clicks on one **2D MRI frame** of the heart (for example, a 4-chamber view).
  The model outputs a **2D segmentation**, and based on that, reconstructs the **entire 3D shape of the heart**.

* **More advanced extension**:
  Provide more **2D frames** to enable sparse 3D shape reconstruction, or incorporate modeling of **motion information**.

**Technical implementation hypothesis?**
Primarily rely on **LoRA fine-tuning**.
On the data side, we have various medical images and corresponding 3D shape datasets that can support the fine-tuning task.

**Priority technical choices:**
- Prioritize mesh-based decoders (`SLatMeshDecoder`) for highest surface fidelity and clinical realism.
- Add comprehensive 3D metrics: voxel Dice / IoU, Chamfer Distance and Hausdorff distance (HD95). Prefer using PyTorch3D and `surface-distance`/SciPy for robust implementations.

**Now please generate a fine-tuning plan**
Based on reading the **paper, code, and README**, provide:

* Data preprocessing steps
* Fine-tuning implementation details
* List of needed preprocessing operations (as implied by the README)
* Practical advice based on checking:
  [https://github.com/facebookresearch/sam-3d-objects/issues](https://github.com/facebookresearch/sam-3d-objects/issues)
