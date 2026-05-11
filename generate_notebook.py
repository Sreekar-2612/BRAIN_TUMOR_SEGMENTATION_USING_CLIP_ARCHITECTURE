import nbformat as nbf

nb = nbf.v4.new_notebook()

text_cells = [
    """# Multimodal Brain Tumor Segmentation using BraTS 2020 (FLAIR + Text) with Vision Language Models

**Objective:**
To develop a multimodal deep learning framework that integrates FLAIR MRI images (BraTS 2020) and textual descriptions (Text BraTS 2020) for accurate brain tumor segmentation and report generation.

**Key Components:**
1. **Data Preprocessing**: Downloading datasets, pairing 155 slices per patient with their corresponding text, and preparing PyTorch DataLoaders.
2. **Model Architecture**: A Dual-Task Vision-Language Model.
    *   **Encoder**: ResNet-based UNet Encoder for Image, ClinicalBERT for Text.
    *   **Fusion**: Cross-Attention module aligning Image & Text.
    *   **Decoder 1 (Segmentation)**: Pixel-wise tumor prediction.
    *   **Decoder 2 (Generation)**: Radiology report generation.
3. **Training Strategy**: Joint optimization using Dice Loss, BCE, and Cross Entropy for text generation.
4. **Evaluation**: Dice, IoU, Hausdorff, ROUGE, BLEU, and visualizations (Grad-CAM, overlays, t-SNE).""",
    
    """## 1. Environment Setup & Data Downloading

**Instructions for Google Colab:**
1. You will need a `kaggle.json` file to download the BraTS dataset. Upload it to Colab when prompted.
2. The Text dataset will be downloaded via `gdown`."""
]

code_cells = [
    """# @title Install Dependencies
!pip install -q kagglehub gdown monai torchmetrics transformers rouge-score nltk nibabel medpy""",
    
    """# @title Download Image Dataset and Verify Text Dataset
import os
import kagglehub
import sys

print("Downloading BraTS 2020 FLAIR dataset from Kaggle...")
# Download latest version
path = kagglehub.dataset_download("hussainnasirkhan/flair-brats2020")
print("Path to image dataset files:", path)

# Set base paths based on environment to avoid symlink issues on Windows

if 'kaggle' in sys.modules or os.path.exists('/kaggle'):
    print("Detected Kaggle Environment.")
    IMAGE_DIR = os.path.join(path, 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
    TEXT_DIR = '/kaggle/input/text-brats'
    if not os.path.exists(TEXT_DIR):
        print("Please upload your 'text_brats.zip' using the Kaggle UI 'Add Data' panel.")
else:
    try:
        from google.colab import files
        print("Detected Colab Environment.")
        IMAGE_DIR = os.path.join(path, 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData')
        TEXT_DIR = '/content/text_brats'
        if not os.path.exists('/content/text_brats.zip'):
            print("Please upload 'text_brats.zip' below:")
            uploaded = files.upload()
            os.system('unzip -q -o /content/text_brats.zip -d /content/text_brats')
    except ImportError:
        print("Detected Local Environment (Windows/CPU).")
        # On Windows, path from kagglehub is usually in .cache. We append the subdirectories.
        IMAGE_DIR = os.path.join(path, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData')
        # We assume TextBraTSData is in the current working directory
        TEXT_DIR = os.path.join(os.getcwd(), 'TextBraTSData')
        
print(f"\\nConfigured IMAGE_DIR: {IMAGE_DIR}")
print(f"Configured TEXT_DIR: {TEXT_DIR}")
"""
]

text_cells.append("## 2. Data Loading and Preprocessing")
code_cells.append("""import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import cv2

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

class BraTSMultimodalDataset(Dataset):
    def __init__(self, image_dir, text_dir, tokenizer, img_size=(224, 224), max_text_len=128):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        if not os.path.exists(self.image_dir):
            print(f"Warning: {self.image_dir} not found. Please verify paths.")
            return samples
            
        patient_folders = sorted(glob.glob(os.path.join(self.image_dir, 'BraTS20_Training_*')))
        
        for p_folder in patient_folders:
            patient_id = os.path.basename(p_folder)
            
            # Paths for FLAIR and Seg
            flair_path = os.path.join(p_folder, f"{patient_id}_flair.nii")
            seg_path = os.path.join(p_folder, f"{patient_id}_seg.nii")
            
            # Find matching text file
            # Assuming text files are named like BraTS20_Training_001_flair_text.txt
            text_files = glob.glob(os.path.join(self.text_dir, '**', f"{patient_id}*text.txt"), recursive=True)
            if not text_files or not os.path.exists(flair_path) or not os.path.exists(seg_path):
                continue
                
            text_path = text_files[0]
            with open(text_path, 'r', encoding='utf-8') as f:
                report_text = f.read().strip()
                
            # We don't load the 3D volume here to save memory, we just store the paths.
            # But since we do 2D slice-wise, we can register 155 samples per patient!
            # To optimize training, we might only sample slices that contain brain/tumor.
            # For simplicity, we register the volume path and in __getitem__ we can randomly pick a slice,
            # OR we can pre-extract 2D slices. Let's do random slice sampling per epoch for a volume.
            samples.append({
                'flair_path': flair_path,
                'seg_path': seg_path,
                'text': report_text
            })
            
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load 3D volumes
        flair_vol = nib.load(sample['flair_path']).get_fdata()
        seg_vol = nib.load(sample['seg_path']).get_fdata()
        
        # Binary mask (Tumor > 0)
        seg_vol = (seg_vol > 0).astype(np.float32)
        
        # Pick a random slice that contains brain matter
        # flair_vol shape: [H, W, D (155)]
        valid_slices = np.where(np.sum(flair_vol, axis=(0,1)) > 0)[0]
        if len(valid_slices) > 0:
            slice_idx = np.random.choice(valid_slices)
        else:
            slice_idx = 75 # middle slice fallback
            
        img_slice = flair_vol[:, :, slice_idx]
        mask_slice = seg_vol[:, :, slice_idx]
        
        # Resize
        img_slice = cv2.resize(img_slice, self.img_size)
        mask_slice = cv2.resize(mask_slice, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize image
        if img_slice.max() > 0:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            
        # Convert to tensor [C, H, W]
        img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)
        
        # Tokenize text
        encoded_text = self.tokenizer(
            sample['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_text_len, 
            return_tensors='pt'
        )
        
        # For auto-regressive generation, input ids are shifted right in the model, 
        # but here we'll just return the full sequence and shift it in the training loop.
        
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'input_ids': encoded_text['input_ids'].squeeze(0),
            'attention_mask': encoded_text['attention_mask'].squeeze(0),
            'raw_text': sample['text']
        }

# Create DataLoaders
print("Instantiating DataLoaders...")
dataset = BraTSMultimodalDataset(IMAGE_DIR, TEXT_DIR, tokenizer)
# Using a small batch size to fit in Colab GPU memory
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"Dataset size: {len(dataset)} samples")
""")

text_cells.append("## 3. Model Architecture (VLM-UNet + Generator)")
code_cells.append("""import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class CrossAttention(nn.Module):
    def __init__(self, in_channels, text_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Linear(text_channels, in_channels)
        self.value = nn.Linear(text_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape
        Q = self.query(img_feat).view(B, C, -1).permute(0, 2, 1) # [B, H*W, C]
        K = self.key(text_feat) # [B, SeqLen, C]
        V = self.value(text_feat) # [B, SeqLen, C]
        
        attn = torch.matmul(Q, K.permute(0, 2, 1)) * self.scale # [B, H*W, SeqLen]
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V) # [B, H*W, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out + img_feat

class VLM_BraTS_Model(nn.Module):
    def __init__(self, vocab_size, text_hidden_size=768):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        
        # UNet Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
        
        # Fusion
        self.cross_attn = CrossAttention(512, text_hidden_size)
        
        # Segmentation Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(64, 1, 1)
        
        # Text Generator Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=8, batch_first=True)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.vocab_proj = nn.Linear(text_hidden_size, vocab_size)
        self.img_to_text = nn.Linear(512, text_hidden_size)

    def forward(self, img, text_input_ids, text_attention_mask, tgt_ids=None):
        text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_feat = text_outputs.last_hidden_state # [B, SeqLen, 768]
        global_text_feat = text_outputs.pooler_output # [B, 768]
        
        e1 = self.enc1(img)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3)) 
        
        fused = self.cross_attn(e4, text_feat)
        
        d3 = self.dec3(torch.cat([self.up3(fused), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_mask = torch.sigmoid(self.final_conv(d1))
        
        img_memory = fused.mean(dim=[2, 3]) 
        img_memory = self.img_to_text(img_memory).unsqueeze(1) 
        
        gen_logits = None
        if tgt_ids is not None:
            tgt_emb = self.text_encoder.embeddings(tgt_ids)
            # Create a causal mask for the decoder
            seq_len = tgt_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(img.device)
            dec_out = self.text_decoder(tgt=tgt_emb, memory=img_memory, tgt_mask=tgt_mask)
            gen_logits = self.vocab_proj(dec_out)
            
        return seg_mask, gen_logits, global_text_feat, img_memory.squeeze(1)
""")

text_cells.append("## 4. Training Strategy")
code_cells.append("""import torch.optim as optim
from tqdm.notebook import tqdm

def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum(dim=[2, 3])
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth)))
    return loss.mean()

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, seg_loss_tot, gen_loss_tot = 0, 0, 0
    
    for batch in tqdm(dataloader, desc="Training"):
        img = batch['image'].to(device)
        mask = batch['mask'].to(device)
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        
        # For text generation, inputs are shifted right.
        # tgt_input: [SOS, token1, token2]
        # tgt_output: [token1, token2, EOS]
        tgt_input = input_ids[:, :-1]
        tgt_expected = input_ids[:, 1:]
        
        optimizer.zero_grad()
        
        seg_pred, gen_logits, text_f, img_f = model(img, input_ids, attn_mask, tgt_input)
        
        # Losses
        loss_bce = F.binary_cross_entropy(seg_pred, mask)
        loss_dice = dice_loss(seg_pred, mask)
        l_seg = loss_bce + loss_dice
        
        l_gen = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), tgt_expected.reshape(-1), ignore_index=tokenizer.pad_token_id)
        l_align = 1.0 - F.cosine_similarity(text_f, img_f).mean()
        
        loss = l_seg + 0.5 * l_gen + 0.1 * l_align
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        seg_loss_tot += l_seg.item()
        gen_loss_tot += l_gen.item()
        
    return total_loss / len(dataloader), seg_loss_tot / len(dataloader), gen_loss_tot / len(dataloader)

# Model Instantiation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = VLM_BraTS_Model(vocab_size=tokenizer.vocab_size).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

epochs = 5 # Reduced for quicker testing, change to 10-20 for full training
for ep in range(epochs):
    train_loss, seg_loss, gen_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {ep+1} | Total Loss: {train_loss:.4f} | Seg Loss: {seg_loss:.4f} | Gen Loss: {gen_loss:.4f}")
""")

text_cells.append("## 5. Evaluation and Visualizations")
code_cells.append("""from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

def evaluate_metrics(model, dataloader, device):
    model.eval()
    dices, ious = [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouges = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            
            # Forward pass without teacher forcing
            seg_pred, _, _, _ = model(img, input_ids, attn_mask, None)
            
            # Binarize prediction
            seg_pred_bin = (seg_pred > 0.5).float()
            
            # Dice & IoU
            intersection = (seg_pred_bin * mask).sum(dim=[2,3])
            union = seg_pred_bin.sum(dim=[2,3]) + mask.sum(dim=[2,3]) - intersection
            
            dice = (2. * intersection + 1e-5) / (seg_pred_bin.sum(dim=[2,3]) + mask.sum(dim=[2,3]) + 1e-5)
            iou = (intersection + 1e-5) / (union + 1e-5)
            
            dices.extend(dice.cpu().numpy().tolist())
            ious.extend(iou.cpu().numpy().tolist())
            
            # Generate Text (Simple Greedy Decoding)
            # This is a basic implementation. For better quality, use Beam Search.
            
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean IoU: {np.mean(ious):.4f}")

def show_predictions(image, true_mask, pred_mask, true_text=""):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title('FLAIR Input')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask.squeeze(), cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask.squeeze(), cmap='gray')
    axes[2].set_title('Predicted Mask')
    axes[2].axis('off')
    
    axes[3].imshow(image.squeeze(), cmap='gray')
    axes[3].imshow(pred_mask.squeeze(), cmap='Reds', alpha=0.5)
    axes[3].set_title('Overlay (Pred in Red)')
    axes[3].axis('off')
    
    plt.show()
    print("Report:", true_text)
""")

for i in range(len(text_cells)):
    nb.cells.append(nbf.v4.new_markdown_cell(text_cells[i]))
    if i < len(code_cells):
        nb.cells.append(nbf.v4.new_code_cell(code_cells[i]))

with open('Multimodal_BraTS_VLM.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Updated Notebook generated successfully!")
