import nbformat
import os

notebook_path = 'Multimodal_BraTS_VLM.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

new_source = """import os
import json
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

OUT_PLOTS = os.path.join("outputs", "plots")
OUT_METRICS = os.path.join("outputs", "metrics")
os.makedirs(OUT_PLOTS, exist_ok=True)
os.makedirs(OUT_METRICS, exist_ok=True)

model.load_state_dict(torch.load("best_vlm_unet.pth", map_location=device, weights_only=False))
model.eval()

# Segmentation metrics (multimodal)
seg_metrics = validate(model, val_loader, device, use_text=True)

# Text metrics
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
rouge1_scores, rougeL_scores, bleu_scores = [], [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating text"):
        img = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        amsk = batch["attention_mask"].to(device)

        gen_ids = model.generate_text(img[:1], ids[:1], amsk[:1], tokenizer)
        pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        ref_text = batch["raw_text"][0]

        sc = scorer.score(ref_text, pred_text)
        rouge1_scores.append(sc["rouge1"].fmeasure)
        rougeL_scores.append(sc["rougeL"].fmeasure)

        ref_tokens = [nltk.word_tokenize(ref_text.lower())]
        pred_tokens = nltk.word_tokenize(pred_text.lower())
        bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=SmoothingFunction().method1))

dices = [seg_metrics["dice"]]
ious = [seg_metrics["iou"]]

print("\\n=== Final Evaluation Results ===")
print(f"Mean Dice      : {seg_metrics['dice']:.4f}")
print(f"Mean IoU       : {seg_metrics['iou']:.4f}")
print(f"Mean Precision : {seg_metrics['precision']:.4f}")
print(f"Mean Recall    : {seg_metrics['recall']:.4f}")
print(f"Mean HD        : {seg_metrics['hd']:.4f}")
print(f"ROUGE-1        : {np.mean(rouge1_scores):.4f}")
print(f"ROUGE-L        : {np.mean(rougeL_scores):.4f}")
print(f"BLEU           : {np.mean(bleu_scores):.4f}")

# Save concise metric summary for downstream report extraction
metric_summary = {
    "dice": float(seg_metrics["dice"]),
    "iou": float(seg_metrics["iou"]),
    "precision": float(seg_metrics["precision"]),
    "recall": float(seg_metrics["recall"]),
    "hausdorff": float(seg_metrics["hd"]),
    "rouge1": float(np.mean(rouge1_scores)),
    "rougeL": float(np.mean(rougeL_scores)),
    "bleu": float(np.mean(bleu_scores)),
}
with open(os.path.join(OUT_METRICS, "notebook_eval_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metric_summary, f, indent=2)

# Training curves: Loss, Dice, IoU
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history["loss"], label="Total Loss")
axes[0].plot(history["seg"], label="Seg Loss")
axes[0].plot(history["gen"], label="Gen Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].set_title("Loss vs Epoch")

axes[1].plot(history["val_dice"], color="green", marker="o")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice"); axes[1].set_title("Dice vs Epoch")

axes[2].plot(history["val_iou"], color="blue", marker="o")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("IoU"); axes[2].set_title("IoU vs Epoch")

plt.tight_layout()
plt.savefig(os.path.join(OUT_PLOTS, "training_curves.png"), dpi=100)
plt.show()"""

found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and 'Evaluating text' in cell.source:
        cell.source = new_source
        # Clear cell outputs to avoid errors
        cell.outputs = []
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('Updated evaluation cell successfully!')
else:
    print('Could not find the evaluation cell.')
