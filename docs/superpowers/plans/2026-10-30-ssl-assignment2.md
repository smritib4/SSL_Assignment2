# Assignment 2 Self-Supervised Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish ITCS 6169/8169 Assignment 2 by training a from-scratch SimCLR comparison on the 2,400-image, 16-class scene dataset, scoring one selected checkpoint on the test set, and submitting a two-page report whose GitHub link reproduces that number.

**Architecture:** One ResNet-18 built with `weights=None` is the shared backbone. SimCLR pretraining optimizes NT-Xent on two views and throws away the projection head. Classification reads the 512-d encoder output. The ladder is a supervised baseline, a linear probe, a fine-tune that changes only the initialization, a fine-tune that also lowers the backbone learning rate, and a joint cross-entropy plus NT-Xent run. The SmallCNN exists only to check this code against Assignment 1's from-scratch result.

**Tech Stack:** Python 3, PyTorch, torchvision, PyYAML, PIL, NumPy. Training machine: Google Colab with a T4 GPU. This laptop has CPU-only PyTorch and is only for the commit, the report, and the submission.

## Global Constraints

- Deadline: October 30, 2026, 11:59 PM EDT. Late work follows the course late-submission policy.
- Dataset: 2,400 training images, 16 classes. Same Drive folder as Assignment 1: `https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M`
- No ImageNet, CLIP, DINO, DINOv2, MAE, SAM, or any other external pretrained weights. `models.resnet18(weights=None)` is the only ResNet constructor.
- No extra images. SimCLR and joint training use the 1,920-image training split only. The 480 validation images stay out of pretraining.
- `train.py` does not read the test set. Hyperparameters are chosen on validation. After training is finished, `evaluate.py --split test` scores the four rows in the handout table: supervised baseline, SSL fine-tune, joint model, and the selected best model. Test accuracy is not used to change a config and retrain.
- Report: one PDF, two pages maximum, submitted on Canvas. The PDF contains a working GitHub link. Source code is not uploaded to Canvas.
- Repository must contain training code, evaluation code, model definitions, the best checkpoint or an external link to it, configs, and a README a grader can follow.
- Do not share code, checkpoints, hyperparameters, or measured accuracies with other students before the deadline.
- Selection rule: highest validation accuracy among finished runs. A gap under about 1 point on 480 images is a tie. Score both checkpoints and report both.
- Accuracies in `reports/REPORT.md` come from `experiments/results/results.csv` and from `evaluate.py`. Do not type a number that those files do not contain.

---

## File map

Already written. Later tasks run these files. They do not reimplement them.

| Path | Responsibility |
|---|---|
| `train.py` | supervised, SimCLR pretrain (`last.pt`), joint (`best.pt`) |
| `evaluate.py` | validation or test accuracy and per-class counts |
| `predict.py` | predictions for a flat image folder |
| `src/ssl_assignment/models.py` | SmallCNN, ResNet-18 from scratch, projection head |
| `src/ssl_assignment/losses.py` | `nt_xent(z1, z2, temperature)` |
| `src/ssl_assignment/data.py` | stratified split, two-view dataset |
| `configs/smallcnn_supervised.yaml` | Assignment 1 from-scratch reference, 60 epochs |
| `configs/resnet18_supervised.yaml` | supervised baseline, 50 epochs |
| `configs/simclr_pretrain.yaml` | 100 epochs, batch 128, crop scale 0.3–1.0 |
| `configs/simclr_linear_probe.yaml` | frozen encoder, 30 epochs |
| `configs/simclr_finetune_matched.yaml` | same optimizer as the baseline |
| `configs/simclr_finetune.yaml` | backbone LR 1e-4, head LR 1e-3 |
| `configs/joint.yaml` | cross-entropy + 0.2 × NT-Xent, 50 epochs |
| `configs/best.yaml` | copy of the winning recipe, name `best` |
| `notebooks/colab_train.ipynb` | Colab driver for the commands below |
| `scripts/make_report_table.py` | validation table and the current leader |
| `reports/REPORT.md` | report source, still containing `XX.X` |

---

### Task 1: Commit and push the training code

**Files:**
- Commit: the paths in the file map, plus `README.md`, `AI_USAGE.md`, `requirements.txt`, `.gitignore`, `scripts/prepare_data.py`, `scripts/make_dummy_data.py`, `configs/smoke.yaml`, `experiments/results/.gitkeep`, `experiments/run_logs/.gitkeep`
- Do not commit: `data/`, `runs/`, `*.pt`, `__pycache__/`, `Assignment2.pdf`

**Interfaces:**
- Consumes: the working tree in `C:\Users\smrit\SSL_Assignment2`
- Produces: `https://github.com/smritib4/SSL_Assignment2` on `main`, which Colab clones

- [x] **Step 1: See what will be committed**

Run from `C:\Users\smrit\SSL_Assignment2`:

```powershell
git status --short
git diff --stat
git log --oneline -5
```

Expected: untracked training files, no commits yet on `main`, remote `origin` is `https://github.com/smritib4/SSL_Assignment2.git`.

- [x] **Step 2: Commit**

```powershell
git add .gitignore AI_USAGE.md README.md requirements.txt train.py evaluate.py predict.py configs scripts src notebooks reports docs experiments
git commit -m "Add from-scratch SimCLR training ladder for Assignment 2."
```

Expected: one commit on `main`. `git status` shows a clean tree.

- [x] **Step 3: Push**

```powershell
git push -u origin HEAD
```

Expected: `main` is on GitHub. Opening the repository URL in a browser shows `train.py` and `configs/`.

---

### Task 2: Open a GPU runtime and stage the dataset

**Files:**
- Create, on the Colab disk only: `data/train/<class>/*.jpg`, `data/test/<class>/*.jpg`
- Modify: nothing in git

**Interfaces:**
- Consumes: the GitHub `main` branch from Task 1
- Produces: a Colab session whose `data/train` contains 2,400 images and whose `torch.cuda.is_available()` is true

- [x] **Step 1: Start Colab on a T4**

In Google Colab: Runtime → Change runtime type → T4 GPU. Open `notebooks/colab_train.ipynb` from the cloned repo, or paste the cells below.

- [x] **Step 2: Refuse to continue without a GPU**

```python
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU. Change the runtime to T4 and rerun.")
```

Expected: `cuda True` and a GPU name.

- [x] **Step 3: Clone and install**

```python
import os
if not os.path.exists("train.py"):
    !git clone https://github.com/smritib4/SSL_Assignment2.git
    os.chdir("/content/SSL_Assignment2")
!pip install -q -r requirements.txt
```

Expected: the shell is in a directory that contains `train.py`.

- [x] **Step 4: Unpack the dataset**

If the Assignment 1 zip is still available, upload it and run:

```bash
python scripts/prepare_data.py --zip /content/dataset.zip
```

Otherwise:

```bash
pip install -q gdown
gdown --folder https://drive.google.com/drive/folders/1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M -O data_download --remaining-ok
python scripts/prepare_data.py --src data_download
```

Expected: the printed train total is `2400`, there are 16 classes, and `data/test` exists. If the total is not 2400, stop. Do not start Task 3.

---

### Task 3: Check the from-scratch reference

**Files:**
- Create: `runs/smallcnn_supervised_seed0/best.pt`, `experiments/run_logs/smallcnn_supervised_seed0/`
- Modify: `experiments/results/results.csv` (this file is created by `train.py`)

**Interfaces:**
- Consumes: `build_backbone("smallcnn", num_classes, dropout)` in `src/ssl_assignment/models.py`
- Produces: one finished row whose `experiment` is `smallcnn_supervised`

- [x] **Step 1: Train**

```bash
python train.py --config configs/smallcnn_supervised.yaml
```

Expected: about 10 minutes on a T4. The log line `trainable params: 4,695,504` matches Assignment 1. The run finishes epoch 60/60 and writes `runs/smallcnn_supervised_seed0/best.pt`.

Done on Colab T4 in 338s. Params `4,695,504`. Best validation accuracy `0.8875` at epoch 54. Saved `runs/smallcnn_supervised_seed0/best.pt`.

- [x] **Step 2: Read the validation accuracy**

```bash
python scripts/make_report_table.py
```

Expected: a `smallcnn_supervised` row with `60/60` epochs and a validation accuracy. Assignment 1 scored this architecture at 89.38% validation. A result in the mid-to-high 80s means the split and the recipe still see the same problem. A result below about 80% means stop and inspect the data layout before spending the rest of the GPU time. Record the printed percentage. It replaces the SmallCNN `XX.X` in the report later.

Recorded from the training log: best validation accuracy 88.75%. The report table prints after the ResNet-18 run in the same cell.

---

### Task 4: Train the supervised baseline

**Files:**
- Create: `runs/resnet18_supervised_seed0/best.pt`

**Interfaces:**
- Consumes: `configs/resnet18_supervised.yaml`
- Produces: the baseline validation accuracy every later comparison uses

- [x] **Step 1: Train**

```bash
python train.py --config configs/resnet18_supervised.yaml
```

Expected: 50/50 epochs, ResNet-18, 11,184,720 trainable parameters, checkpoint `runs/resnet18_supervised_seed0/best.pt`. This is the number SimCLR has to beat under a matched recipe.

Done on Colab T4 in 315s. Params `11,184,720`. Best validation accuracy `0.7958` at epoch 45. Saved `runs/resnet18_supervised_seed0/best.pt`.

- [x] **Step 2: Confirm the row**

```bash
python scripts/make_report_table.py
```

Expected: `resnet18_supervised` shows `50/50` and a validation accuracy. Copy that percentage into the notes you will use for the report. Do not run `evaluate.py --split test` yet.

Recorded: `resnet18_supervised` 50/50, best epoch 45, validation 79.58%. The table also named `smallcnn_supervised` the leader. That line is only a comparison of the two finished runs. SmallCNN stays a validation reference. Do not score it on the test set.

---

### Task 5: Pretrain SimCLR

**Files:**
- Create: `runs/simclr_pretrain_seed0/last.pt`
- This checkpoint's `kind` is `simclr`. It is not a classifier. `evaluate.py` must reject it.

**Interfaces:**
- Consumes: `nt_xent` in `src/ssl_assignment/losses.py` and `SimCLRModel` in `src/ssl_assignment/models.py`
- Produces: `runs/simclr_pretrain_seed0/last.pt`, which Tasks 6, 7, and 8 load through `ssl.init_checkpoint`

- [x] **Step 1: Train**

```bash
python train.py --config configs/simclr_pretrain.yaml
```

Expected: 100/100 epochs, batch 128, log lines of the form `nt-xent <number>`. No validation accuracy. Wall clock is the long step, on the order of 30–40 minutes. The file `runs/simclr_pretrain_seed0/last.pt` exists at the end.

Done on Colab T4 in 1576s. Trainable params `11,505,344 / 11,513,552`. NT-Xent fell from `5.3469` to `1.7372`. Saved `runs/simclr_pretrain_seed0/last.pt`. The matched fine-tune then printed `Loaded encoder from runs/simclr_pretrain_seed0/last.pt (epoch 100)`.

- [x] **Step 2: Confirm the table treats it as pretraining**

```bash
python scripts/make_report_table.py
```

Expected: the `simclr_pretrain` row shows validation `n/a` and a final NT-Xent value. It is not eligible to be the leader.

---

### Task 6: Linear probe

**Files:**
- Create: `runs/simclr_linear_probe_seed0/best.pt`

**Interfaces:**
- Consumes: the encoder inside `runs/simclr_pretrain_seed0/last.pt`
- Produces: a validation accuracy with the encoder frozen (8,208 trainable parameters)

- [ ] **Step 1: Train the head only**

```bash
python train.py --config configs/simclr_linear_probe.yaml
```

Expected: the log says `Loaded encoder from runs/simclr_pretrain_seed0/last.pt` and `Encoder frozen`. Trainable parameters are `8,208`. The run completes 30/30 epochs.

- [ ] **Step 2: Record the probe accuracy**

```bash
python scripts/make_report_table.py
```

Expected: `simclr_linear_probe` has a validation accuracy. This number answers whether the frozen representation is linearly separable. It is a diagnostic. It becomes the selected model only if it actually wins the table, which is unlikely.

---

### Task 7: Matched fine-tune

**Files:**
- Create: `runs/simclr_finetune_matched_seed0/best.pt`

**Interfaces:**
- Consumes: `runs/simclr_pretrain_seed0/last.pt`
- Produces: the single-factor comparison against `resnet18_supervised` (same augmentation, same learning rate 1e-3, same 50 epochs, same split)

- [x] **Step 1: Train**

```bash
python train.py --config configs/simclr_finetune_matched.yaml
```

Expected: encoder loads, encoder is not frozen, 50/50 epochs, checkpoint `runs/simclr_finetune_matched_seed0/best.pt`.

Done on Colab T4 in 318s. Encoder loaded from epoch 100 and was not frozen. Best validation accuracy `0.8313` at epoch 41. Saved `runs/simclr_finetune_matched_seed0/best.pt`.

- [x] **Step 2: Compare with the baseline in the table**

```bash
python scripts/make_report_table.py
```

Expected: both `resnet18_supervised` and `simclr_finetune_matched` show `50/50`. Subtract the two validation accuracies. Keep that difference for the report. A difference under 0.01 is a tie, not a win for SimCLR initialization.

Recorded: matched fine-tune 83.13% minus supervised baseline 79.58% is +3.55 points. That is outside the 1-point noise floor, so the SimCLR initialization won the matched comparison. The report table prints after the joint run.

---

### Task 8: Discriminative fine-tune

**Files:**
- Create: `runs/simclr_finetune_seed0/best.pt`

**Interfaces:**
- Consumes: the same `last.pt` as Task 7
- Produces: a second fine-tune whose backbone learning rate is 1e-4 and whose head learning rate is 1e-3

- [x] **Step 1: Train**

```bash
python train.py --config configs/simclr_finetune.yaml
```

Expected: 50/50 epochs. This run is not a single-factor comparison with the baseline. The report must say the learning rate changed as well as the initialization.

Done on Colab T4 in 323s. Encoder loaded from epoch 100 and was not frozen. Best validation accuracy `0.8396` at epoch 28. Saved `runs/simclr_finetune_seed0/best.pt`.

- [x] **Step 2: Add the row to your notes**

```bash
python scripts/make_report_table.py
```

Expected: `simclr_finetune` has a validation accuracy beside the matched fine-tune.

Recorded: discriminative fine-tune 83.96% (epoch 28) versus matched fine-tune 83.13% (epoch 41). The gap is 0.83 points, inside the 1-point noise floor, so Task 10 scores both and the report states both. The higher validation row is `simclr_finetune`. Joint training started immediately after this save.

---

### Task 9: Joint training

**Files:**
- Create: `runs/joint_seed0/best.pt`

**Interfaces:**
- Consumes: `JointModel.forward`, which returns logits from the encoder output and NT-Xent on the projection
- Produces: the joint validation accuracy. Loss is cross-entropy on both views plus `0.2 * NT-Xent`. Augmentation is the SimCLR view policy, so this row changes the loss and the augmentation together.

- [x] **Step 1: Train**

```bash
python train.py --config configs/joint.yaml
```

Expected: log lines of the form `loss <total> (ce <ce> + 0.2 * nt-xent <ssl>)` and a validation accuracy each epoch. 50/50 epochs. Checkpoint `runs/joint_seed0/best.pt`.

Done on Colab T4 in 834s. Best validation accuracy `0.8146` at epoch 43. Saved `runs/joint_seed0/best.pt`.

- [x] **Step 2: Read the full ladder**

```bash
python scripts/make_report_table.py
```

Expected: seven data rows. Pretraining is `n/a`. The script prints a leader and a checkpoint path. If it also prints that the second-place run is within 1 point, both of those checkpoints are selected in Task 10.

Recorded validation: smallcnn 88.75% (reference only), resnet18_supervised 79.58%, simclr_finetune_matched 83.13%, simclr_finetune 83.96%, joint 81.46%. Pretrain final NT-Xent 1.737. The script named SmallCNN the leader because it is the highest number in the file. SmallCNN is a different architecture and stays out of the test table. Among the ResNet-18 runs, `simclr_finetune` is the leader. The matched fine-tune is within 1 point of it. Joint is 2.5 points behind the leader.

---

### Task 10: Score the handout's four rows on the test set

The report example in Assignment 2, section 6, is a table of **test** accuracy with four rows: Supervised Baseline, SSL + Fine-tuning, Joint SSL + Classification, and Best Model. Validation still decides which fine-tune is the SSL row and which checkpoint is "best." The test numbers are recorded after that choice. They are not a reason to edit a config and train again.

**Files:**
- Modify: `configs/best.yaml` only if the validation leader is not the joint recipe
- Create: `runs/best_seed0/best.pt` only in that same case

**Interfaces:**
- Consumes: the validation table from `scripts/make_report_table.py`
- Produces: four test accuracies, from `evaluate.py` stdout, written down before the report is filled in

- [x] **Step 1: Align `configs/best.yaml` with the winner**

If the leader is `joint`, leave `configs/best.yaml` as it is. It is already that recipe with `experiment.name: best`.

If the leader is another finished classification run, replace the body of `configs/best.yaml` with that run's YAML and set:

```yaml
experiment:
  name: best
```

Keep every other setting identical to the winner, including `seed: 0`. Then rerun that one config:

```bash
python train.py --config configs/best.yaml
```

Expected when a rerun is required: `runs/best_seed0/best.pt`. Expected when joint won: no new training. The checkpoint you will score is `runs/joint_seed0/best.pt`.

The validation leader was `simclr_finetune`, not joint. `configs/best.yaml` was replaced with that recipe and `experiment.name: best`. It was not trained again. Retraining would make a second checkpoint and a second test score. The scored file remains `runs/simclr_finetune_seed0/best.pt`.

- [x] **Step 2: Score the four report rows**

Run each command once. Skip the best-model command when that checkpoint is already one of the other three.

```bash
python evaluate.py --checkpoint runs/resnet18_supervised_seed0/best.pt --split test
python evaluate.py --checkpoint runs/simclr_finetune_matched_seed0/best.pt --split test
python evaluate.py --checkpoint runs/simclr_finetune_seed0/best.pt --split test
python evaluate.py --checkpoint runs/joint_seed0/best.pt --split test
```

The SSL + Fine-tuning cell in the report is the fine-tune with the higher validation accuracy. If those two validation numbers are within 1 point, put both test numbers in the report and say the validation gap was inside the noise floor. The Best Model cell is the validation leader. If a separate `runs/best_seed0/best.pt` exists, score that file too:

```bash
python evaluate.py --checkpoint runs/best_seed0/best.pt --split test
```

Expected for every command: a first line `Scoring the TEST set.`, then `Accuracy: <number>%` and a per-class table. Write down each accuracy and, for the best model only, the three weakest classes.

SmallCNN and the linear probe stay validation numbers. They are not rows in the handout's test table.

Recorded test accuracy: supervised baseline 75.00%, matched fine-tune 76.75%, discriminative fine-tune 77.50%, joint 78.75%. Best model is the discriminative fine-tune because validation selected it (83.96%). Joint's higher test number was not used to change that choice. Weakest test classes of the best model: Industrial 44%, Kitchen 48%, LivingRoom 60%.

---

### Task 11: Download artifacts and fill the report

**Files:**
- Modify: `reports/REPORT.md`, `README.md`
- Create, outside git if the file is large: a Drive or Hugging Face copy of the selected `best.pt`

**Interfaces:**
- Consumes: `experiments/results/results.csv`, the test accuracy from Task 10, and the per-class lines from `evaluate.py`
- Produces: a report with no remaining `XX.X`, and a README checkpoint URL

- [ ] **Step 1: Download the Colab artifacts before the session ends**

```python
!zip -r -q /content/ssl_runs.zip runs experiments/results experiments/run_logs
from google.colab import files
files.download("/content/ssl_runs.zip")
```

Unzip `ssl_runs.zip` into `C:\Users\smrit\SSL_Assignment2` so `runs/` and `experiments/` are present locally. `runs/*.pt` stays gitignored.

- [x] **Step 2: Host the selected checkpoint**

Upload the `best.pt` you scored (joint, or `runs/best_seed0/best.pt` if you reran Task 10) to Google Drive. Put the sharing link in `README.md` under a heading `## Checkpoint`, in this form:

```markdown
## Checkpoint

The scored checkpoint is `<filename>`.

Download: <paste the Drive URL>

python evaluate.py --checkpoint <local path to that file> --split test
```

Confirmed by loading the shared file: kind `classifier`, name `simclr_finetune`, epoch 28, validation accuracy 0.8396, backbone lr 1e-4, head multiplier 10. Link is in `README.md`.

- [ ] **Step 3: Replace every `XX.X` in `reports/REPORT.md`**

Use only numbers from `python scripts/make_report_table.py` and from the `evaluate.py` test output.

| Report cell | Source row or output |
|---|---|
| Best-model test accuracy | Task 10 accuracy for the validation leader |
| Validation accuracy of the selected checkpoint | that run's `Val acc` in the table |
| Supervised baseline test accuracy | Task 10 accuracy for `resnet18_supervised` |
| SSL + fine-tuning test accuracy | Task 10 accuracy for the better fine-tune |
| Joint test accuracy | Task 10 accuracy for `joint` |
| SmallCNN and linear probe, if they fit on the page | validation rows only |

The report table must match Assignment 2 section 6: Method, then Test Accuracy, for Supervised Baseline, SSL + Fine-tuning, Joint SSL + Classification, and Best Model.

In "What the numbers say", answer three things from the runs, not from a guess. Did SSL fine-tuning beat the supervised baseline on the test set, and was the gap larger than 1 point? Did joint training beat SSL fine-tuning? For the baseline and the best model, is training accuracy far above validation accuracy in `experiments/run_logs/<run>/history.jsonl`? That last comparison is the overfitting question. State that the joint run changed both the loss and the augmentation, so a joint win is not a pure loss result. Name the three weakest test classes of the best model. Write what failed as well as what won.

- [ ] **Step 4: Check the report length**

Export `reports/REPORT.md` to PDF. Confirm it is two pages or fewer. Cut the limitations paragraph before cutting the results table if it overflows. The GitHub URL in the report is `https://github.com/smritib4/SSL_Assignment2`.

- [ ] **Step 5: Commit the report, the logs, and the README link**

```powershell
git add reports/REPORT.md reports/REPORT.pdf README.md experiments/results/results.csv experiments/run_logs
git commit -m "Record Assignment 2 validation results and the test accuracy."
git push
```

Expected: GitHub shows `experiments/results/results.csv` and the checkpoint URL in `README.md`. The `.pt` file is not in the commit. `git status` does not list `*.pt`.

---

### Task 12: Submit

**Files:**
- Submit: the two-page PDF only, through Canvas

**Interfaces:**
- Consumes: the PDF from Task 11 and the pushed GitHub `main`
- Produces: a Canvas submission before October 30, 2026, 11:59 PM EDT

- [ ] **Step 1: Open the GitHub link from the PDF in a private window**

Expected: the README checkpoint link loads, `train.py` is on `main`, and `configs/best.yaml` matches the recipe the report describes.

- [ ] **Step 2: Upload the PDF on Canvas**

Expected: the assignment shows the PDF received before 11:59 PM EDT. Source code is not attached separately.

---

## Spec coverage

| Assignment requirement | Task |
|---|---|
| Supervised baseline with architecture and accuracy | Task 4, report Task 11 |
| SSL, then supervised classification or fine-tuning | Tasks 5–8 |
| Best training recipe | Tasks 9–10, `configs/best.yaml` |
| No external pretrained weights | Global Constraints, enforced by `build_backbone` |
| Two-page report: best accuracy, GitHub, baseline, SSL explanation, table, limitations | Task 11 |
| Training code, eval code, models, checkpoint, configs, README | Tasks 1 and 11 |
| Test accuracy for baseline, SSL fine-tune, joint, and best model | Task 10 |
| Interpretation: what worked, what failed, overfitting | Task 11 |

## Self-review

- Spec coverage: the three required experiments, the four-row test table in section 6, the GitHub contents, and the no-pretraining rule each have a task above. Augmentation-only and backbone-only SSL ablations are not separate runs. Section 5 says those questions are optional. The joint-run note in Task 11 is where the augmentation confound gets stated.
- Placeholder scan: report numbers are tied to a named results row or to `evaluate.py` stdout. No step says "implement later".
- Type consistency: the pretrain artifact is always `runs/simclr_pretrain_seed0/last.pt` with `kind: simclr`. Classifiers are `best.pt`. `evaluate.py` scores only `kind: classifier`.
