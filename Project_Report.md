# Text-Guided Brain Tumor Segmentation using Vision Language Models with FLAIR MRI and Radiology Reports

## 1. Objective
To develop a multimodal deep learning framework that integrates FLAIR MRI images (BraTS 2020) and textual descriptions (Text BraTS 2020) for accurate brain tumor segmentation, using a Vision-Language Model (VLM) based approach to align image and text representations. Additionally, the system includes a sequence-to-sequence generation head to synthesize radiology reports from visual features.

## 2. Dataset Details

### 2.1 Image Dataset
**BraTS 2020 FLAIR MRI dataset**
Data includes 3D MRI volumes (155 slices per patient) and corresponding ground-truth segmentation masks. For this project, we process the data slice-wise (2D) to reduce computational overhead while preserving high-resolution structural details. Slices are filtered to ensure they contain significant brain tissue/tumor context.

### 2.2 Text Dataset
**Text BraTS 2020**
Contains radiology-style descriptions detailing tumor location, edema information, and structural characteristics for each patient. Each unique text report is paired with all corresponding slices from the patient's MRI volume.

## 3. Problem Statement
Design and implement a system that takes MRI images and corresponding text descriptions as input, learns a joint representation by aligning image and text features, performs pixel-wise tumor segmentation, and demonstrates how textual information improves segmentation performance. Furthermore, the system evaluates its representation capability by auto-generating radiology texts from images.

## 4. Methodology

### 4.1 Image Preprocessing Pipeline
1. **Loading**: The notebook pipeline uses the `FLAIR_BRATS2020_split` preprocessed `.npy` slice dataset (image/mask pairs with matched filenames).
2. **Patient-aware pairing**: Patient IDs are parsed from filenames and used to enforce strict patient-level image-text alignment (one report per patient, reused only for that patient's slices).
3. **Resizing & Normalization**: Slices and masks are resized to $224 \times 224$ via OpenCV; image intensities are Min-Max normalized to `[0, 1]`.
4. **Binarization**: Masks are binarized (Tumor vs Background) for primary segmentation training.

### 4.2 Text Preprocessing Pipeline
1. **Tokenization**: Text strings are tokenized using `Bio_ClinicalBERT` (`emilyalsentzer/Bio_ClinicalBERT`), yielding `input_ids` and `attention_masks`.
2. **Padding/Truncation**: Fixed sequence length representation ensures uniform batching during training.

### 4.3 Model Architecture
The network is a **Dual-Task Vision-Language Model (VLM-UNet)** consisting of:
- **Text Encoder**: A pre-trained `ClinicalBERT` extracts a 768-dimensional sequential embedding and a global feature vector from the radiology report.
- **Image Encoder**: A CNN-based U-Net encoder (stacked convolution blocks with pooling) compresses the $224 \times 224$ input into a deep $512$-channel bottleneck representation.
- **Cross-Modal Fusion (Attention Bottleneck)**: A Cross-Attention module sits at the bottleneck. Deep image features act as Queries, attending to the sequential text features (Keys/Values). This allows the vision network to structurally focus on regions described in the text.
- **Segmentation Decoder**: A standard UNet decoder with skip-connections upsamples the fused features back to $224 \times 224$, outputting a pixel-wise probability map.
- **Text Generation Decoder**: A Transformer Decoder takes the mean-pooled visual features as `memory` and autoregressively generates a radiology report text.

### 4.4 Training Strategy
The network uses a multi-task joint optimization approach:
1. **Segmentation Loss**: A combination of Binary Cross Entropy (BCE) and Soft Dice Loss.
2. **Generation Loss**: Cross-Entropy Loss for next-token prediction via teacher forcing.
3. **Alignment Loss**: Cosine alignment loss between global text and image features.
4. **Ablation mode**: A true image-only path (`use_text=False`) bypasses text fusion for controlled with-text vs without-text comparison.

Total Loss = $L_{Seg} + 0.5 \cdot L_{Gen} + 0.1 \cdot L_{Align}$
Optimizer: `AdamW` with learning rate $1 \times 10^{-4}$.

## 5. Required Graphs and Visualizations
*(Note: These will be generated natively within the submitted Colab Notebook during execution.)*

### 5.1 Training Performance
- **Dice Score / IoU / Loss vs Epoch**: Plotted using `matplotlib` to demonstrate convergence and validation performance.

### 5.2 Segmentation Results
- Side-by-side subplot panels showing: `[Input FLAIR] | [Ground Truth Mask] | [Predicted Mask] | [Overlay]`.

### 5.3 Attention / Explainability
- Visualization of the Cross-Attention weights from the fusion bottleneck, resized and overlaid on the original image as a spatial attention map (gradient-based Grad-CAM is additionally implemented in `run_nb.py` Phase 4 analysis).

### 5.4 Ablation Study
- The notebook performs a true architecture-level ablation using `use_text=False` (no text fusion), and compares against the full multimodal (`use_text=True`) model using Dice and IoU bar plots.

### 5.5 Embedding Analysis
- **t-SNE Projection**: A 2D scatter plot generated via `sklearn.manifold.TSNE` mapping the high-dimensional `global_text_feat` and `img_memory` points. Proper alignment shows text and corresponding image embeddings clustering together.

### 5.6 Error Analysis
- Histogram of Dice Scores across the validation set.
- Worst-case qualitative panel (lowest Dice samples) with input, GT, prediction, and overlay views.

## 6. Evaluation Metrics
The model is quantitatively evaluated using:
- **Segmentation**: Dice Coefficient, Intersection over Union (IoU), Precision, Recall, and Hausdorff Distance.
- **Text Generation**: ROUGE-1, ROUGE-L, and sentence BLEU.
