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
- `train.py` does not read the test set. Test scoring happens once, in `evaluate.py --split test`, after validation selection.
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

- [ ] **Step 1: See what will be committed**

Run from `C:\Users\smrit\SSL_Assignment2`:

```powershell
git status --short
git diff --stat
git log --oneline -5
```

Expected: untracked training files, no commits yet on `main`, remote `origin` is `https://github.com/smritib4/SSL_Assignment2.git`.

- [ ] **Step 2: Commit**

```powershell
git add .gitignore AI_USAGE.md README.md requirements.txt train.py evaluate.py predict.py configs scripts src notebooks reports docs experiments
git commit -m "Add from-scratch SimCLR training ladder for Assignment 2."
```

Expected: one commit on `main`. `git status` shows a clean tree.

- [ ] **Step 3: Push**

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

- [ ] **Step 1: Start Colab on a T4**

In Google Colab: Runtime → Change runtime type → T4 GPU. Open `notebooks/colab_train.ipynb` from the cloned repo, or paste the cells below.

- [ ] **Step 2: Refuse to continue without a GPU**

```python
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No GPU. Change the runtime to T4 and rerun.")
```

Expected: `cuda True` and a GPU name.

- [ ] **Step 3: Clone and install**

```python
import os
if not os.path.exists("train.py"):
    !git clone https://github.com/smritib4/SSL_Assignment2.git
    os.chdir("/content/SSL_Assignment2")
!pip install -q -r requirements.txt
```

Expected: the shell is in a directory that contains `train.py`.

- [ ] **Step 4: Unpack the dataset**

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

- [ ] **Step 1: Train**

```bash
python train.py --config configs/smallcnn_supervised.yaml
```

Expected: about 10 minutes on a T4. The log line `trainable params: 4,695,504` matches Assignment 1. The run finishes epoch 60/60 and writes `runs/smallcnn_supervised_seed0/best.pt`.

- [ ] **Step 2: Read the validation accuracy**

```bash
python scripts/make_report_table.py
```

Expected: a `smallcnn_supervised` row with `60/60` epochs and a validation accuracy. Assignment 1 scored this architecture at 89.38% validation. A result in the mid-to-high 80s means the split and the recipe still see the same problem. A result below about 80% means stop and inspect the data layout before spending the rest of the GPU time. Record the printed percentage. It replaces the SmallCNN `XX.X` in the report later.

---

### Task 4: Train the supervised baseline

**Files:**
- Create: `runs/resnet18_supervised_seed0/best.pt`

**Interfaces:**
- Consumes: `configs/resnet18_supervised.yaml`
- Produces: the baseline validation accuracy every later comparison uses

- [ ] **Step 1: Train**

```bash
python train.py --config configs/resnet18_supervised.yaml
```

Expected: 50/50 epochs, ResNet-18, 11,184,720 trainable parameters, checkpoint `runs/resnet18_supervised_seed0/best.pt`. This is the number SimCLR has to beat under a matched recipe.

- [ ] **Step 2: Confirm the row**

```bash
python scripts/make_report_table.py
```

Expected: `resnet18_supervised` shows `50/50` and a validation accuracy. Copy that percentage into the notes you will use for the report. Do not run `evaluate.py --split test` yet.

---

### Task 5: Pretrain SimCLR

**Files:**
- Create: `runs/simclr_pretrain_seed0/last.pt`
- This checkpoint's `kind` is `simclr`. It is not a classifier. `evaluate.py` must reject it.

**Interfaces:**
- Consumes: `nt_xent` in `src/ssl_assignment/losses.py` and `SimCLRModel` in `src/ssl_assignment/models.py`
- Produces: `runs/simclr_pretrain_seed0/last.pt`, which Tasks 6, 7, and 8 load through `ssl.init_checkpoint`

- [ ] **Step 1: Train**

```bash
python train.py --config configs/simclr_pretrain.yaml
```

Expected: 100/100 epochs, batch 128, log lines of the form `nt-xent <number>`. No validation accuracy. Wall clock is the long step, on the order of 30–40 minutes. The file `runs/simclr_pretrain_seed0/last.pt` exists at the end.

- [ ] **Step 2: Confirm the table treats it as pretraining**

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

- [ ] **Step 1: Train**

```bash
python train.py --config configs/simclr_finetune_matched.yaml
```

Expected: encoder loads, encoder is not frozen, 50/50 epochs, checkpoint `runs/simclr_finetune_matched_seed0/best.pt`.

- [ ] **Step 2: Compare with the baseline in the table**

```bash
python scripts/make_report_table.py
```

Expected: both `resnet18_supervised` and `simclr_finetune_matched` show `50/50`. Subtract the two validation accuracies. Keep that difference for the report. A difference under 0.01 is a tie, not a win for SimCLR initialization.

---

### Task 8: Discriminative fine-tune

**Files:**
- Create: `runs/simclr_finetune_seed0/best.pt`

**Interfaces:**
- Consumes: the same `last.pt` as Task 7
- Produces: a second fine-tune whose backbone learning rate is 1e-4 and whose head learning rate is 1e-3

- [ ] **Step 1: Train**

```bash
python train.py --config configs/simclr_finetune.yaml
```

Expected: 50/50 epochs. This run is not a single-factor comparison with the baseline. The report must say the learning rate changed as well as the initialization.

- [ ] **Step 2: Add the row to your notes**

```bash
python scripts/make_report_table.py
```

Expected: `simclr_finetune` has a validation accuracy beside the matched fine-tune.

---

### Task 9: Joint training

**Files:**
- Create: `runs/joint_seed0/best.pt`

**Interfaces:**
- Consumes: `JointModel.forward`, which returns logits from the encoder output and NT-Xent on the projection
- Produces: the joint validation accuracy. Loss is cross-entropy on both views plus `0.2 * NT-Xent`. Augmentation is the SimCLR view policy, so this row changes the loss and the augmentation together.

- [ ] **Step 1: Train**

```bash
python train.py --config configs/joint.yaml
```

Expected: log lines of the form `loss <total> (ce <ce> + 0.2 * nt-xent <ssl>)` and a validation accuracy each epoch. 50/50 epochs. Checkpoint `runs/joint_seed0/best.pt`.

- [ ] **Step 2: Read the full ladder**

```bash
python scripts/make_report_table.py
```

Expected: seven data rows. Pretraining is `n/a`. The script prints a leader and a checkpoint path. If it also prints that the second-place run is within 1 point, both of those checkpoints are selected in Task 10.

---

### Task 10: Score the test set once

**Files:**
- Modify: `configs/best.yaml` only if the leader is not the joint recipe
- Create: `runs/best_seed0/best.pt` only in that same case

**Interfaces:**
- Consumes: the leader path printed by `scripts/make_report_table.py`
- Produces: the test accuracy that goes in the report, from `evaluate.py` stdout

- [ ] **Step 1: Align `configs/best.yaml` with the winner**

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

- [ ] **Step 2: Evaluate the leader on the test set**

Replace the path with the leader. Run this one time:

```bash
python evaluate.py --checkpoint runs/joint_seed0/best.pt --split test
```

Expected first line: `Scoring the TEST set.` Then `Accuracy: <number>%` and a per-class table. Write down the accuracy and the three weakest classes.

- [ ] **Step 3: Score the runner-up only if the validation gap is under 1 point**

```bash
python evaluate.py --checkpoint <runner-up>/best.pt --split test
```

Expected: a second test accuracy. The report states both and says the validation gap was inside the noise floor. If the gap was 1 point or more, skip this step. Do not score any other checkpoint.

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

- [ ] **Step 2: Host the selected checkpoint**

Upload the `best.pt` you scored (joint, or `runs/best_seed0/best.pt` if you reran Task 10) to Google Drive. Put the sharing link in `README.md` under a heading `## Checkpoint`, in this form:

```markdown
## Checkpoint

The scored checkpoint is `<filename>`.

Download: <paste the Drive URL>

python evaluate.py --checkpoint <local path to that file> --split test
```

- [ ] **Step 3: Replace every `XX.X` in `reports/REPORT.md`**

Use only numbers from `python scripts/make_report_table.py` and from the `evaluate.py` test output.

| Report cell | Source row or output |
|---|---|
| Test accuracy | `Accuracy:` from Task 10 |
| Validation accuracy of the selected checkpoint | that run's `Val acc` in the table |
| SmallCNN validation | `smallcnn_supervised` |
| ResNet-18 supervised baseline | `resnet18_supervised` |
| SimCLR linear probe | `simclr_linear_probe` |
| Matched fine-tune | `simclr_finetune_matched` |
| Smaller-backbone-LR fine-tune | `simclr_finetune` |
| Joint | `joint` |

In "What the numbers say", write which row won and the gap in points. If the gap is under 1 point, write that it matches Assignment 1's 0.62-point same-seed spread and is not evidence of a difference. State explicitly that the joint run changed both the loss and the augmentation. Name the three weakest test classes from the per-class printout.

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
| Test set reported once, after selection | Task 10 |

## Self-review

- Spec coverage: the three required experiments, the report sections, the GitHub contents, and the no-pretraining rule each have a task above.
- Placeholder scan: report numbers are tied to a named results row or to `evaluate.py` stdout. No step says "implement later".
- Type consistency: the pretrain artifact is always `runs/simclr_pretrain_seed0/last.pt` with `kind: simclr`. Classifiers are `best.pt`. `evaluate.py` scores only `kind: classifier`.
