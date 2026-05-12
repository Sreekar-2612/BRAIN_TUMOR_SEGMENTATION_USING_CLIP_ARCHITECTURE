import os
import glob
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import json
import time
import platform
import argparse
import zipfile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, asdict
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    # Training Hyperparameters
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 10
    seed: int = 42
    artifacts_root: str = "artifacts"

    # Data Parameters
    img_size: tuple = (224, 224)
    max_text_len: int = 128
    num_workers: int = 0 # Set to 0 for Windows stability unless configured

    # Checkpointing & Early Stopping
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Phase 4 Analysis
    use_text: bool = True
    experiment_name: str = "multimodal_run"
    error_top_k: int = 10
    save_error_cases: bool = True
    num_visual_samples: int = 5
    output_dir: str = "outputs"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Reproducibility: Seed set to {seed}, CUDNN deterministic=True")

# =============================================================================
# DATA LAYER
# =============================================================================
class BraTSMultimodalDataset(Dataset):
    def __init__(self, image_dir, text_dir, tokenizer, patient_list, mode='train', img_size=(224, 224), max_text_len=128):
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_text_len = max_text_len
        self.patient_list = patient_list
        self.mode = mode
        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        for p_id in self.patient_list:
            p_folder = os.path.join(self.image_dir, p_id)
            flair_path = os.path.join(p_folder, f"{p_id}_flair.nii")
            seg_path = os.path.join(p_folder, f"{p_id}_seg.nii")
            text_files = glob.glob(os.path.join(self.text_dir, '**', f"{p_id}*text.txt"), recursive=True)
            if not text_files or not os.path.exists(flair_path) or not os.path.exists(seg_path):
                continue
            text_path = text_files[0]
            with open(text_path, 'r', encoding='utf-8') as f:
                report_text = f.read().strip()
            samples.append({'patient_id': p_id, 'flair_path': flair_path, 'seg_path': seg_path, 'text': report_text, 'text_path': text_path})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        flair_vol = nib.load(sample['flair_path']).get_fdata()
        seg_vol = nib.load(sample['seg_path']).get_fdata()
        seg_vol = (seg_vol > 0).astype(np.float32)
        valid_slices = np.where(np.sum(flair_vol, axis=(0,1)) > 0)[0]
        if self.mode == 'train':
            slice_idx = np.random.choice(valid_slices) if len(valid_slices) > 0 else 75
        else:
            slice_idx = valid_slices[len(valid_slices)//2] if len(valid_slices) > 0 else 75
        img_slice = flair_vol[:, :, slice_idx]
        mask_slice = seg_vol[:, :, slice_idx]
        img_slice = cv2.resize(img_slice, self.img_size)
        mask_slice = cv2.resize(mask_slice, self.img_size, interpolation=cv2.INTER_NEAREST)
        if img_slice.max() > 0:
            img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)
        encoded_text = self.tokenizer(sample['text'], padding='max_length', truncation=True, max_length=self.max_text_len, return_tensors='pt')
        return {'image': img_tensor, 'mask': mask_tensor, 'input_ids': encoded_text['input_ids'].squeeze(0), 'attention_mask': encoded_text['attention_mask'].squeeze(0), 'raw_text': sample['text']}

def get_dataloaders(config, image_dir, text_dir, tokenizer):
    all_patient_folders = sorted(glob.glob(os.path.join(image_dir, 'BraTS20_Training_*')))
    if len(all_patient_folders) == 0:
        split_images = glob.glob(os.path.join(image_dir, "*.npy"))
        if split_images:
            raise RuntimeError(
                "run_nb.py expects MICCAI BraTS folder layout with per-patient .nii files, "
                f"but found split .npy slices at {image_dir}. "
                "Set BRATS_MICCAI_ROOT to a MICCAI_BraTS2020_TrainingData folder "
                "or run the notebook pipeline for FLAIR_BRATS2020_split."
            )
        raise RuntimeError(
            f"No BraTS20_Training_* folders found under IMAGE_DIR={image_dir}. "
            "Set BRATS_MICCAI_ROOT to a valid MICCAI_BraTS2020_TrainingData path."
        )
    all_patient_ids = [os.path.basename(f) for f in all_patient_folders]
    random.shuffle(all_patient_ids)
    split_idx = int(0.9 * len(all_patient_ids))
    train_ids = all_patient_ids[:split_idx]
    val_ids = all_patient_ids[split_idx:]

    assert len(set(train_ids).intersection(set(val_ids))) == 0, "Train and Val sets must be disjoint"

    train_dataset = BraTSMultimodalDataset(image_dir, text_dir, tokenizer, train_ids, mode='train', img_size=config.img_size, max_text_len=config.max_text_len)
    val_dataset = BraTSMultimodalDataset(image_dir, text_dir, tokenizer, val_ids, mode='val', img_size=config.img_size, max_text_len=config.max_text_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    if len(train_loader) == 0: raise RuntimeError("Training DataLoader is empty. Check data paths.")
    if len(val_loader) == 0: raise RuntimeError("Validation DataLoader is empty. Check data paths.")

    return train_loader, val_loader

# =============================================================================
# MODEL LAYER
# =============================================================================
class CrossAttention(nn.Module):
    def __init__(self, in_channels, text_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Linear(text_channels, in_channels)
        self.value = nn.Linear(text_channels, in_channels)
        self.scale = in_channels ** -0.5

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape
        Q = self.query(img_feat).view(B, C, -1).permute(0, 2, 1)
        K = self.key(text_feat)
        V = self.value(text_feat)
        attn = torch.matmul(Q, K.permute(0, 2, 1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V).permute(0, 2, 1).view(B, C, H, W)
        return out + img_feat, attn.mean(dim=1).view(B, H, W)

class VLM_BraTS_Model(nn.Module):
    def __init__(self, vocab_size, text_hidden_size=768):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.enc1 = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
        self.cross_attn = CrossAttention(512, text_hidden_size)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.final_conv = nn.Conv2d(64, 1, 1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=text_hidden_size, nhead=8, batch_first=True)
        self.text_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.vocab_proj = nn.Linear(text_hidden_size, vocab_size)
        self.img_to_text = nn.Linear(512, text_hidden_size)

    def forward(self, img, text_input_ids, text_attention_mask, tgt_ids=None, use_text=True):
        if use_text:
            text_outputs = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attention_mask)
            text_feat = text_outputs.last_hidden_state
            global_text_feat = text_outputs.pooler_output
        else:
            global_text_feat = torch.zeros((img.size(0), 768), device=img.device)

        e1 = self.enc1(img)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        if use_text:
            fused, attn_map = self.cross_attn(e4, text_feat)
        else:
            fused = e4
            attn_map = torch.zeros((img.size(0), e4.size(2), e4.size(3)), device=img.device)
        d3 = self.dec3(torch.cat([self.up3(fused), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg_mask = torch.sigmoid(self.final_conv(d1))
        img_memory = fused.mean(dim=[2, 3])
        img_memory = self.img_to_text(img_memory).unsqueeze(1)
        gen_logits = None
        if tgt_ids is not None and use_text:
            tgt_emb = self.text_encoder.embeddings(tgt_ids)
            seq_len = tgt_ids.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(img.device)
            dec_out = self.text_decoder(tgt=tgt_emb, memory=img_memory, tgt_mask=tgt_mask)
            gen_logits = self.vocab_proj(dec_out)
        return seg_mask, gen_logits, global_text_feat, img_memory.squeeze(1), attn_map, d1

# =============================================================================
# TRAINING ENGINE
# =============================================================================
def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum(dim=[2, 3])
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + smooth)))
    return loss.mean()

def compute_metrics(pred, target):
    pred_bin = (pred > 0.5).float()
    inter = (pred_bin * target).sum(dim=[2, 3])
    union = pred_bin.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    dice = (2. * inter + 1e-5) / (union + 1e-5)
    iou = (inter + 1e-5) / (pred_bin.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) - inter + 1e-5)
    tp = (pred_bin * target).sum(dim=[2, 3])
    fp = (pred_bin * (1 - target)).sum(dim=[2, 3])
    fn = ((1 - pred_bin) * target).sum(dim=[2, 3])
    precision = (tp + 1e-5) / (tp + fp + 1e-5)
    recall = (tp + 1e-5) / (tp + fn + 1e-5)
    return dice.mean(), iou.mean(), precision.mean(), recall.mean()

def hausdorff_distance(pred, target):
    p_bin = (pred > 0.5).cpu().numpy().astype(np.uint8)
    t_bin = (target > 0.5).cpu().numpy().astype(np.uint8)
    if p_bin.sum() == 0 and t_bin.sum() == 0: return 0.0
    if p_bin.sum() == 0 or t_bin.sum() == 0: return 100.0
    d_to_target = distance_transform_edt(1 - t_bin)
    d_p_g = np.max(d_to_target[p_bin == 1]) if p_bin.sum() > 0 else 100.0
    d_to_pred = distance_transform_edt(1 - p_bin)
    d_g_p = np.max(d_to_pred[t_bin == 1]) if t_bin.sum() > 0 else 100.0
    return max(d_p_g, d_g_p)

def train_one_epoch(model, dataloader, optimizer, device, tokenizer, config):
    model.train()
    metrics = {'total': 0, 'seg': 0, 'gen': 0, 'align': 0}
    valid_batch_count = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        img, mask = batch['image'].to(device), batch['mask'].to(device)
        input_ids, attn_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        tgt_input, tgt_expected = input_ids[:, :-1], input_ids[:, 1:]

        optimizer.zero_grad()
        seg_pred, gen_logits, text_f, img_f, *_ = model(img, input_ids, attn_mask, tgt_input, use_text=config.use_text)

        l_seg = F.binary_cross_entropy(seg_pred, mask) + dice_loss(seg_pred, mask)
        
        if config.use_text:
            l_gen = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), tgt_expected.reshape(-1), ignore_index=tokenizer.pad_token_id)
            l_align = 1.0 - F.cosine_similarity(text_f, img_f).mean()
            loss = l_seg + 0.5 * l_gen + 0.1 * l_align
        else:
            l_gen = torch.tensor(0.0)
            l_align = torch.tensor(0.0)
            loss = l_seg

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        optimizer.step()

        valid_batch_count += 1
        metrics['total'] += loss.item()
        metrics['seg'] += l_seg.item()
        metrics['gen'] += l_gen.item()
        metrics['align'] += l_align.item()

    if valid_batch_count == 0:
        raise RuntimeError("No valid batches processed in this epoch (all loss was NaN/Inf).")

    return {k: v / valid_batch_count for k, v in metrics.items()}, (len(dataloader) - valid_batch_count)

def validate(model, loader, device, config):
    model.eval()
    results = {'dice': [], 'iou': [], 'prec': [], 'rec': [], 'hd': []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            img, mask = batch['image'].to(device), batch['mask'].to(device)
            input_ids, attn_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            seg_pred, *_ = model(img, input_ids, attn_mask, use_text=config.use_text)

            d, i, p, r = compute_metrics(seg_pred, mask)
            results['dice'].append(d.item())
            results['iou'].append(i.item())
            results['prec'].append(p.item())
            results['rec'].append(r.item())

            for b in range(seg_pred.size(0)):
                results['hd'].append(hausdorff_distance(seg_pred[b], mask[b]))

    return {k: np.mean(v) for k, v in results.items()}

def save_checkpoint(model, optimizer, epoch, best_dice, config, is_best=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    path = os.path.join(config.checkpoint_dir, "best.pt" if is_best else "latest.pt")
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_dice': best_dice,
        'config': asdict(config)
    }, path)

def load_checkpoint(path, model, optimizer, config):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}, starting from scratch.")
        return 0, 0.0
    ckpt = torch.load(path, map_location=config.device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    print(f"Resumed from checkpoint: Epoch {ckpt['epoch']}, Best Dice: {ckpt['best_dice']:.4f}")
    return ckpt['epoch'], ckpt['best_dice']

# =============================================================================
# PHASE 4: ANALYSIS & EXPLAINABILITY
# =============================================================================

def get_true_gradcam(model, img, input_ids, attn_mask, use_text):
    """
    True Grad-CAM utilizing gradients from segmentation output wrt last decoder block.
    """
    model.eval()
    img_v = img.clone().requires_grad_(True)
    seg_pred, _, _, _, _, d1 = model(img_v, input_ids, attn_mask, use_text=use_text)
    
    d1.retain_grad()
    loss = seg_pred.sum()
    model.zero_grad()
    loss.backward()
    
    gradients = d1.grad
    activations = d1.detach()
    
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1)
    cam = F.relu(cam)
    
    cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
    cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    return cam

def run_analysis(model, val_loader, device, config):
    print("\n--- Running Phase 4 Analysis ---")
    model.eval()
    
    viz_dir = os.path.join(config.output_dir, "visuals", config.experiment_name)
    err_dir = os.path.join(config.output_dir, "error_analysis", config.experiment_name)
    emb_dir = os.path.join(config.output_dir, "embeddings")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    
    all_dices = []
    worst_cases = []
    text_embeddings = []
    img_embeddings = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Analysis", leave=False)):
            img, mask = batch['image'].to(device), batch['mask'].to(device)
            input_ids, attn_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            
            seg_pred, _, global_text_feat, img_memory, attn_map, d1 = model(img, input_ids, attn_mask, use_text=config.use_text)
            
            if config.use_text:
                text_embeddings.append(global_text_feat.cpu().numpy())
                img_embeddings.append(img_memory.cpu().numpy())
            
            pred_bin = (seg_pred > 0.5).float()
            for b in range(img.size(0)):
                i_b, p_b, m_b = img[b, 0].cpu().numpy(), pred_bin[b, 0].cpu().numpy(), mask[b, 0].cpu().numpy()
                inter = (p_b * m_b).sum()
                union = p_b.sum() + m_b.sum()
                dice = (2. * inter + 1e-5) / (union + 1e-5)
                all_dices.append(dice)
                
                if len(worst_cases) < config.error_top_k:
                    worst_cases.append((dice, i_b, m_b, p_b))
                    worst_cases.sort(key=lambda x: x[0])
                elif dice < worst_cases[-1][0]:
                    worst_cases[-1] = (dice, i_b, m_b, p_b)
                    worst_cases.sort(key=lambda x: x[0])
                
                if batch_idx == 0 and b < config.num_visual_samples:
                    with torch.enable_grad():
                        cam_map = get_true_gradcam(model, img[b:b+1], input_ids[b:b+1], attn_mask[b:b+1], config.use_text)[0]
                        
                    plt.figure(figsize=(20, 4))
                    plt.subplot(1, 6, 1); plt.imshow(i_b, cmap='gray'); plt.title('Input MRI'); plt.axis('off')
                    plt.subplot(1, 6, 2); plt.imshow(m_b, cmap='gray'); plt.title('GT Mask'); plt.axis('off')
                    plt.subplot(1, 6, 3); plt.imshow(p_b, cmap='gray'); plt.title('Predicted Mask'); plt.axis('off')
                    
                    overlay = np.zeros((*i_b.shape, 3))
                    overlay[..., 0] = i_b
                    overlay[..., 1] = np.clip(i_b + m_b * 0.5, 0, 1)
                    overlay[..., 2] = np.clip(i_b + p_b * 0.5, 0, 1)
                    plt.subplot(1, 6, 4); plt.imshow(overlay); plt.title('Overlay'); plt.axis('off')
                    
                    att_b = cv2.resize(attn_map[b].cpu().numpy(), (i_b.shape[1], i_b.shape[0]))
                    plt.subplot(1, 6, 5); plt.imshow(i_b, cmap='gray'); plt.imshow(att_b, cmap='jet', alpha=0.5); plt.title('Attention Map'); plt.axis('off')
                    
                    c_b = cv2.resize(cam_map.cpu().numpy(), (i_b.shape[1], i_b.shape[0]))
                    plt.subplot(1, 6, 6); plt.imshow(i_b, cmap='gray'); plt.imshow(c_b, cmap='jet', alpha=0.5); plt.title('True Grad-CAM'); plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f"sample_panel_{b}.png"))
                    plt.close()

    plt.figure()
    plt.hist(all_dices, bins=20, color='blue', alpha=0.7)
    plt.title("Dice Distribution on Validation Set")
    plt.xlabel("Dice Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(err_dir, "dice_histogram.png"))
    plt.close()

    if config.save_error_cases:
        plt.figure(figsize=(15, 3 * len(worst_cases)))
        for idx, (d, i_b, m_b, p_b) in enumerate(worst_cases):
            plt.subplot(len(worst_cases), 4, idx*4 + 1); plt.imshow(i_b, cmap='gray'); plt.title(f'Rank {idx+1} Input'); plt.axis('off')
            plt.subplot(len(worst_cases), 4, idx*4 + 2); plt.imshow(m_b, cmap='gray'); plt.title('GT Mask'); plt.axis('off')
            plt.subplot(len(worst_cases), 4, idx*4 + 3); plt.imshow(p_b, cmap='gray'); plt.title(f'Pred (Dice: {d:.4f})'); plt.axis('off')
            overlay = np.zeros((*i_b.shape, 3))
            overlay[..., 0] = i_b
            overlay[..., 1] = np.clip(i_b + m_b * 0.5, 0, 1)
            overlay[..., 2] = np.clip(i_b + p_b * 0.5, 0, 1)
            plt.subplot(len(worst_cases), 4, idx*4 + 4); plt.imshow(overlay); plt.title('Overlay'); plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(err_dir, "worst_k_cases.png"))
        plt.close()

    if config.use_text and text_embeddings and img_embeddings:
        all_text = np.concatenate(text_embeddings, axis=0)
        all_img = np.concatenate(img_embeddings, axis=0)
        n_samples = min(500, all_text.shape[0])
        idxs = np.random.choice(all_text.shape[0], n_samples, replace=False)
        features = np.vstack([all_text[idxs], all_img[idxs]])
        
        tsne = TSNE(n_components=2, random_state=config.seed)
        tsne_res = tsne.fit_transform(features)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(tsne_res[:n_samples, 0], tsne_res[:n_samples, 1], c='red', label='Text', alpha=0.5)
        plt.scatter(tsne_res[n_samples:, 0], tsne_res[n_samples:, 1], c='blue', label='Image', alpha=0.5)
        plt.legend()
        plt.title("t-SNE of Image and Text Embeddings")
        plt.savefig(os.path.join(emb_dir, f"tsne_image_text_{config.experiment_name}.png"))
        plt.close()

    print(f"\nArtifacts saved to: {config.output_dir} subdirectories")
    return all_dices

def save_run_summary(config, final_metrics, best_metrics, history):
    os.makedirs(os.path.join(config.output_dir, "metrics"), exist_ok=True)
    history_path = os.path.join(config.output_dir, "metrics", f"{config.experiment_name}_history.json")
    summary = {
        "config": asdict(config),
        "final_metrics": final_metrics,
        "best_metrics": best_metrics,
        "checkpoint_path": os.path.join(config.checkpoint_dir, "best.pt"),
        "history_path": history_path
    }
    with open(os.path.join(config.output_dir, "metrics", f"{config.experiment_name}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================
def _write_run_manifest(config, extra=None):
    root = getattr(config, "artifacts_root", "artifacts")
    mdir = os.path.join(root, "metrics")
    os.makedirs(mdir, exist_ok=True)
    payload = {
        "started_at_unix": getattr(config, "_run_started_at", None),
        "finished_at_unix": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(config),
    }
    if extra:
        payload.update(extra)
    path = os.path.join(mdir, f"run_nb_{config.experiment_name}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Run manifest written to {path}")


def run_pipeline(config):
    config._run_started_at = time.time()
    set_seed(config.seed)
    device = torch.device(config.device)
    print(f"\n{'='*55}")
    print(f"Starting Run: {config.experiment_name} | use_text: {config.use_text}")
    print(f"Running on device: {device}")
    print(f"{'='*55}")

    import kagglehub, sys

    miccai_env = os.environ.get("BRATS_MICCAI_ROOT")
    if miccai_env and os.path.isdir(miccai_env):
        path = os.path.abspath(miccai_env.strip().strip('"').strip("'"))
        print(f"Using local MICCAI BraTS folder (BRATS_MICCAI_ROOT): {path}")
    else:
        path = kagglehub.dataset_download("hussainnasirkhan/flair-brats2020")

    if 'kaggle' in sys.modules or os.path.exists('/kaggle'):
        TEXT_DIR = '/kaggle/input/text-brats'
    else:
        repo = os.path.dirname(os.path.abspath(__file__))
        for cand in (
            os.path.join(os.getcwd(), 'TextBraTSData'),
            os.path.join(repo, 'TextBraTSData'),
            r"C:\Users\theni\OneDrive\Desktop\New folder\BRAIN_TUMOR_SEGMENTATION_USING_CLIP_ARCHITECTURE\TextBraTSData",
        ):
            if os.path.isdir(cand):
                TEXT_DIR = cand
                break
        else:
            TEXT_DIR = os.path.join(os.getcwd(), 'TextBraTSData')

    # Resolve MICCAI image root robustly across kagglehub cache layouts.
    image_candidates = [
        os.path.join(path, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData'),
        os.path.join(path, 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'),
        os.path.join(path, 'versions', '1', 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData'),
    ]
    # If kagglehub returns a ".complete" marker path, probe sibling versions/1 too.
    if path.endswith(".complete"):
        base = os.path.dirname(path)
        image_candidates.append(os.path.join(base, "versions", "1", "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"))

    IMAGE_DIR = None
    for c in image_candidates:
        if os.path.isdir(c):
            IMAGE_DIR = c
            break
    if IMAGE_DIR is None and miccai_env and os.path.isdir(miccai_env):
        IMAGE_DIR = miccai_env

    if IMAGE_DIR is None:
        split_candidates = [
            os.path.join(path, "FLAIR_BRATS2020_split", "train", "images"),
            os.path.join(path, "versions", "1", "FLAIR_BRATS2020_split", "train", "images"),
        ]
        if path.endswith(".complete"):
            base = os.path.dirname(path)
            split_candidates.append(os.path.join(base, "versions", "1", "FLAIR_BRATS2020_split", "train", "images"))
        found_split = next((c for c in split_candidates if os.path.isdir(c)), None)
        if found_split:
            raise RuntimeError(
                "Detected FLAIR_BRATS2020_split dataset, but run_nb.py requires MICCAI BraTS2020 .nii layout. "
                f"Found split path: {found_split}. "
                "Set BRATS_MICCAI_ROOT to MICCAI_BraTS2020_TrainingData or use the notebook pipeline."
            )
        raise RuntimeError(
            "Could not resolve MICCAI_BraTS2020_TrainingData from downloaded path. "
            "Set BRATS_MICCAI_ROOT explicitly."
        )

    # Local-first TextBraTS; fallback to the provided Google Drive zip when missing.
    def _has_text_reports(base_dir):
        return os.path.isdir(base_dir) and bool(
            glob.glob(os.path.join(base_dir, '**', '*.txt'), recursive=True)
        )

    if not _has_text_reports(TEXT_DIR):
        gdrive_file_id = os.environ.get("TEXT_BRATS_GDRIVE_FILE_ID", "17YKI4nwPW8qMKlg9k53dVax7F_1JCk9B")
        dl_base = '/content' if os.path.isdir('/content') else os.getcwd()
        zip_path = os.path.join(dl_base, 'text_brats.zip')
        extract_dir = os.path.join(dl_base, 'TextBraTSData')
        os.makedirs(extract_dir, exist_ok=True)

        # Manual zip fallback before attempting gdown.
        manual_zip = os.environ.get("TEXT_BRATS_ZIP_PATH")
        for cand_zip in (manual_zip, zip_path, os.path.join(dl_base, "TextBraTSData.zip")):
            if cand_zip and os.path.isfile(cand_zip):
                try:
                    with zipfile.ZipFile(cand_zip, 'r') as zf:
                        zf.extractall(extract_dir)
                    if _has_text_reports(extract_dir):
                        TEXT_DIR = extract_dir
                        print(f"Using text dataset extracted from local zip: {cand_zip}")
                        break
                except Exception as e:
                    print(f"Could not extract {cand_zip}: {e}")

    if not _has_text_reports(TEXT_DIR):
        try:
            import gdown
            print(f"Text reports not found at {TEXT_DIR}. Downloading from Google Drive...")
            attempts = [
                {"id": gdrive_file_id},
                {"url": f"https://drive.google.com/uc?id={gdrive_file_id}"},
                {"url": f"https://drive.google.com/file/d/{gdrive_file_id}/view?usp=sharing", "fuzzy": True},
            ]
            errors = []
            for spec in attempts:
                try:
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
                    if "id" in spec:
                        gdown.download(id=spec["id"], output=zip_path, quiet=False, resume=True)
                    else:
                        gdown.download(spec["url"], zip_path, quiet=False, fuzzy=spec.get("fuzzy", False), resume=True)

                    if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            zf.extractall(extract_dir)
                        if _has_text_reports(extract_dir):
                            TEXT_DIR = extract_dir
                            print(f"Using downloaded text dataset: {TEXT_DIR}")
                            break
                except Exception as e:
                    errors.append(str(e))
            if not _has_text_reports(TEXT_DIR):
                print("Google Drive download attempts failed.")
                if errors:
                    print(f"Last gdown error: {errors[-1]}")
        except Exception as e:
            print(f"Text dataset download failed: {e}")

    if not _has_text_reports(TEXT_DIR):
        raise RuntimeError(
            "TextBraTS reports not found. Set TEXT_BRA_TS_DATA to an existing text folder, "
            "or set TEXT_BRATS_ZIP_PATH to a local TextBraTSData.zip, "
            "or ensure Google Drive access for the provided file ID."
        )

    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    train_loader, val_loader = get_dataloaders(config, IMAGE_DIR, TEXT_DIR, tokenizer)

    model = VLM_BraTS_Model(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)

    config.checkpoint_dir = os.path.join("checkpoints", config.experiment_name)
    latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
    start_epoch, best_dice = load_checkpoint(latest_path, model, optimizer, config)

    patience_counter = 0
    
    history = {
        'train_total_loss': [], 'train_seg_loss': [], 'train_gen_loss': [], 'train_align_loss': [],
        'val_dice': [], 'val_iou': [], 'val_prec': [], 'val_rec': [], 'val_hd': []
    }
    best_metrics = {}

    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_metrics, nans = train_one_epoch(model, train_loader, optimizer, device, tokenizer, config)
        val_metrics = validate(model, val_loader, device, config)
        
        history['train_total_loss'].append(train_metrics['total'])
        history['train_seg_loss'].append(train_metrics['seg'])
        history['train_gen_loss'].append(train_metrics['gen'])
        history['train_align_loss'].append(train_metrics['align'])
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_prec'].append(val_metrics['prec'])
        history['val_rec'].append(val_metrics['rec'])
        history['val_hd'].append(val_metrics['hd'])

        if val_metrics['dice'] > best_dice + config.early_stopping_min_delta:
            best_dice = val_metrics['dice']
            best_metrics = val_metrics.copy()
            save_checkpoint(model, optimizer, epoch, best_dice, config, is_best=True)
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, epoch, best_dice, config, is_best=False)

        print(f"Epoch [{epoch:02d}/{config.epochs}] | "
              f"Train Loss: {train_metrics['total']:.4f} (Seg: {train_metrics['seg']:.4f}, Gen: {train_metrics['gen']:.4f}, Align: {train_metrics['align']:.4f}) | "
              f"Val Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | Prec: {val_metrics['prec']:.4f} | Rec: {val_metrics['rec']:.4f} | HD: {val_metrics['hd']:.2f}")

        if nans > 10:
            print("Too many NaNs detected. Stopping training.")
            break

        if patience_counter >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}. Best Dice: {best_dice:.4f}")
            break

    print(f"\n--- Final Evaluation for {config.experiment_name} ---")
    best_model_path = os.path.join(config.checkpoint_dir, "best.pt")
    final_m = {}
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state'])
        final_m = validate(model, val_loader, device, config)
        print(f"Final Metrics -> Dice: {final_m['dice']:.4f}, IoU: {final_m['iou']:.4f}, Prec: {final_m['prec']:.4f}, Rec: {final_m['rec']:.4f}, HD: {final_m['hd']:.2f}")
        # Always use final_m (which comes from best.pt) as the true best_metrics
        best_metrics = final_m.copy()
    
    # Run Phase 4 Analysis
    run_analysis(model, val_loader, device, config)
    save_run_summary(config, final_m, best_metrics, history)
    _write_run_manifest(config, extra={"final_metrics": final_m, "best_metrics": best_metrics})
    return final_m, history

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Multimodal BraTS Ablation Study")
    parser.add_argument('--use_text', action='store_true', help='Run full multimodal pipeline')
    parser.add_argument('--no_text', action='store_true', help='Run image-only baseline')
    parser.add_argument('--run_both', action='store_true', help='Run both sequentially')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override AdamW learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Override random seed')
    parser.add_argument('--artifacts-root', type=str, default=None, help='Root for run manifests (default: artifacts)')
    args = parser.parse_args()

    if not (args.use_text or args.no_text or args.run_both):
        args.use_text = True

    def _apply_overrides(conf: PipelineConfig) -> PipelineConfig:
        if args.epochs is not None:
            conf.epochs = args.epochs
        if args.batch_size is not None:
            conf.batch_size = args.batch_size
        if args.lr is not None:
            conf.lr = args.lr
        if args.seed is not None:
            conf.seed = args.seed
        if args.artifacts_root is not None:
            conf.artifacts_root = args.artifacts_root
        return conf

    results = {}
    histories = {}

    if args.run_both or args.use_text:
        conf = _apply_overrides(PipelineConfig(use_text=True, experiment_name="multimodal_run"))
        res, hist = run_pipeline(conf)
        results['multimodal'] = res
        histories['multimodal'] = hist

    if args.run_both or args.no_text:
        conf = _apply_overrides(PipelineConfig(use_text=False, experiment_name="image_only_baseline"))
        res, hist = run_pipeline(conf)
        results['image_only'] = res
        histories['image_only'] = hist

    if args.run_both:
        print(f"\n{'='*55}")
        print("ABLATION STUDY COMPARISON")
        print(f"{'='*55}")
        keys = ['dice', 'iou', 'prec', 'rec', 'hd']
        m_res = results.get('multimodal', {})
        i_res = results.get('image_only', {})
        for k in keys:
            m_val = f"{m_res.get(k, 0.0):.4f}" if k in m_res else "N/A"
            i_val = f"{i_res.get(k, 0.0):.4f}" if k in i_res else "N/A"
            print(f"{k.upper():<15} | Multi: {m_val:<10} | Image-Only: {i_val:<10}")
        print(f"{'='*55}")
        
        # Plot comparisons
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(histories['multimodal']['val_dice'], label='Multimodal')
        plt.plot(histories['image_only']['val_dice'], label='Image-Only Baseline')
        plt.title('Validation Dice Score')
        plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(histories['multimodal']['val_iou'], label='Multimodal')
        plt.plot(histories['image_only']['val_iou'], label='Image-Only Baseline')
        plt.title('Validation IoU')
        plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend()
        
        os.makedirs("outputs/ablation", exist_ok=True)
        plot_path = os.path.join("outputs", "ablation", "comparison_curves.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
