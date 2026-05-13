import nbformat

with open('fix_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_source = """import os, glob, numpy as np, torch, cv2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("Tokenizer loaded. Vocab size:", tokenizer.vocab_size)

class BraTSMultimodalDataset(Dataset):
    \"\"\"
    Pairs pre-processed 2D FLAIR .npy slices with radiology text reports.
    Images  : IMAGE_DIR/image_N.npy  (H x W x 1 or H x W)
    Masks   : MASK_DIR/image_N.npy   (same naming)
    Texts   : TEXT_DIR/**/BraTS20_Training_*_flair_text.txt  (cycled if fewer than images)
    \"\"\"
    def __init__(self, image_dir, mask_dir, text_dir, tokenizer,
                 img_size=(224, 224), max_text_len=128):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.tokenizer = tokenizer
        self.img_size  = img_size
        self.max_text_len = max_text_len

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
        self.mask_paths  = sorted(glob.glob(os.path.join(mask_dir,  "*.npy")))

        # Collect all text reports
        txt_files = sorted(glob.glob(os.path.join(text_dir, "**", "*.txt"), recursive=True))
        self.texts = []
        for tf in txt_files:
            with open(tf, "r", encoding="utf-8") as f:
                self.texts.append(f.read().strip())

        assert len(self.image_paths) > 0, f"No images found in {image_dir}"
        assert len(self.mask_paths)  > 0, f"No masks found  in {mask_dir}"
        assert len(self.texts)       > 0, f"No text files found in {text_dir}"

        # Align lengths — use min of images & masks; cycle texts
        n = min(len(self.image_paths), len(self.mask_paths))
        self.image_paths = self.image_paths[:n]
        self.mask_paths  = self.mask_paths[:n]
        # Cycle texts to cover all image slices
        self.texts = [self.texts[i % len(self.texts)] for i in range(n)]
        print(f"Dataset: {n} samples, {len(txt_files)} unique text reports (cycled)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_vol  = np.load(self.image_paths[idx]).astype(np.float32)
        mask_vol = np.load(self.mask_paths[idx]).astype(np.float32)

        if mask_vol.ndim == 4:
            tumor_mask = mask_vol.max(axis=-1)
        else:
            tumor_mask = mask_vol
            
        if img_vol.ndim == 4:
            img_vol = img_vol[..., 0] 

        valid_slices = np.where(tumor_mask.sum(axis=(0, 1)) > 0)[0]
        if len(valid_slices) > 0:
            slice_idx = np.random.choice(valid_slices)
        else:
            slice_idx = np.random.randint(0, img_vol.shape[2] if img_vol.ndim >= 3 else 1)

        if img_vol.ndim >= 3:
            img = img_vol[:, :, slice_idx]
            mask = tumor_mask[:, :, slice_idx]
        else:
            img = img_vol
            mask = tumor_mask

        img  = cv2.resize(img,  self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        mn, mx = img.min(), img.max()
        if mx > mn: img = (img - mn) / (mx - mn)
        mask = (mask > 0.5).astype(np.float32)

        img_t  = torch.tensor(img,  dtype=torch.float32).unsqueeze(0)   
        mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)   

        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length", truncation=True,
            max_length=self.max_text_len, return_tensors="pt"
        )
        return {
            "image":          img_t,
            "mask":           mask_t,
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "raw_text":       self.texts[idx],
        }

dataset = BraTSMultimodalDataset(IMAGE_DIR, MASK_DIR, TEXT_DIR, tokenizer)
n_val   = max(1, int(0.1 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

BATCH = 4
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print(f"Batches per epoch: {len(train_loader)}")

sample = dataset[0]
print("image:", sample["image"].shape, "mask:", sample["mask"].shape,
      "input_ids:", sample["input_ids"].shape)
"""

with open('Multimodal_BraTS_VLM.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and 'class BraTSMultimodalDataset(Dataset):' in cell.source:
        cell.source = new_source
        # Clear cell outputs to avoid errors
        cell.outputs = []
        break

with open('Multimodal_BraTS_VLM.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print('Updated notebook successfully!')
