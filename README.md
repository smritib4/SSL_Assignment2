# Self-Supervised Learning on 2,400 Scene Images

ITCS 6169/8169 Computer Vision, Assignment 2.
Author: Smriti Bhemireddy

Question: with the same 2,400 images and 16 scene classes as Assignment 1, do
self-supervised representations beat a supervised model trained from scratch?

External pretrained weights are not allowed. ImageNet ConvNeXt (96.75% test in
Assignment 1) is outside the rules of this assignment, and so is every other
torchvision checkpoint.

## What carries over from Assignment 1

Kept, because it was about the data rather than about ImageNet:

- The dataset, the class folders, and `scripts/prepare_data.py`.
- A stratified 80/20 split at seed 0. That is 1,920 train and 480 validation
  images, 30 validation images per class when each class has 150 training
  images. One percent on that validation set is about five images.
- Colour and a resolution above the starter's 64px grayscale. The from-scratch
  network in Assignment 1 used 128px RGB, and that is the resolution used here.
- The 8-convolution SmallCNN (`configs/smallcnn_supervised.yaml`). It is re-run
  so this repository can be checked against the previous from-scratch number,
  89.38% validation. It is a reference point, not the SSL control.
- AdamW, cosine decay with warmup, gradient clipping, label smoothing on the
  ResNet runs, and weight decay that skips biases and norm parameters.
- Validation-only model selection. `train.py` never reads the test set.

Not kept:

- `pretrained=True`, ConvNeXt, ResNet-50 fine-tunes, and the linear probe of a
  frozen ImageNet encoder. Those were the entire reason Assignment 1 cleared
  90%. The allowed comparison is against a from-scratch network, not against
  96.75%.
- The conclusion that strong augmentation does not help. That result was about
  an encoder that already had ImageNet features. A from-scratch encoder needs
  augmentation, and SimCLR needs a different augmentation again (see below).
- Mixup and cutmix. They blend two images into one, which destroys the
  "two views, one image" pair SimCLR is built on.
- Discriminative learning rates as a default for every run. They are used only
  when the encoder already holds SimCLR features (`configs/simclr_finetune.yaml`).
  The matched fine-tune keeps the baseline learning rate so the init is the
  only change.

## Method

Backbone for the ladder: ResNet-18, `weights=None`, 128px. The same backbone
is used for the supervised baseline, SimCLR, the linear probe, both fine-tunes,
and the joint run. SmallCNN is only the Assignment 1 reference.

SimCLR (Chen et al., 2020). Two random views of an image are a positive pair.
The encoder (512-d) is followed by a 2-layer projection MLP. NT-Xent pulls the
two projections together and pushes them away from the other images in the
batch. After pretraining, the projection is thrown away. The classifier, the
linear probe, and the fine-tune all read the 512-d encoder output. The paper's
own ablation says that layer transfers better than the projection.

The view policy is SimCLR's, with one change: the random crop covers 30% to
100% of the image, not 8% to 100%. These labels are scene categories. A very
small crop often does not contain the scene. Colour jitter (p=0.8), random
grayscale (p=0.2), and Gaussian blur (p=0.5) stay. Random erasing does not:
it deletes the evidence the two views are supposed to agree on. Temperature
is 0.2. Batch size is 128, which divides 1,920 exactly, so `drop_last` does
not throw images away. Pretraining sees only the 1,920 training images. The
validation images are not used as unlabeled data.

Joint training keeps the labels. Each step applies cross-entropy to both views
and adds `0.2 * NT-Xent`. This is the run most likely to help, because every
image is already labelled and Assignment 1 showed those labels are enough for
a from-scratch CNN to reach the high 80s. It is not a pure ablation: it uses
the SimCLR augmentation, so a win is "loss plus augmentation," not the loss
alone.

## Experiments

All of these use seed 0 and the same split, except where a row says otherwise.

| Config | What it answers |
|---|---|
| `configs/smallcnn_supervised.yaml` | Does this code still land near Assignment 1's 89.38% from-scratch result? |
| `configs/resnet18_supervised.yaml` | Supervised baseline. 50 epochs, classification augmentation. |
| `configs/simclr_pretrain.yaml` | 100 epochs of NT-Xent. Saves `last.pt` (an encoder, not a classifier). |
| `configs/simclr_linear_probe.yaml` | Freeze that encoder, train a linear head for 30 epochs. |
| `configs/simclr_finetune_matched.yaml` | Same recipe as the baseline. Only the encoder init changes. |
| `configs/simclr_finetune.yaml` | Same init, but backbone LR 1e-4 and head LR 1e-3. |
| `configs/joint.yaml` | Cross-entropy plus 0.2 * NT-Xent for 50 epochs. |
| `configs/best.yaml` | Starts as a copy of the joint recipe. Change it if validation picks another row. |

`scripts/make_report_table.py` prints the table from `experiments/results/results.csv`
and names the validation leader. Differences under about one point on 480
images were inside the noise floor in Assignment 1. Treat a gap that small as
a tie, score both checkpoints, and report both.

## Train

This laptop has CPU-only PyTorch. The runs above are meant for a Colab GPU.
`notebooks/colab_train.ipynb` clones the repo, downloads the dataset, and runs
the ladder. On a T4 that is on the order of two hours.

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py --zip /path/to/dataset.zip
python train.py --config configs/resnet18_supervised.yaml
python train.py --config configs/simclr_pretrain.yaml
python train.py --config configs/simclr_linear_probe.yaml
python train.py --config configs/simclr_finetune_matched.yaml
python train.py --config configs/simclr_finetune.yaml
python train.py --config configs/joint.yaml
python scripts/make_report_table.py
python evaluate.py --checkpoint runs/<selected>_seed0/best.pt --split test
```

The dataset link is the same Drive folder as Assignment 1:

https://drive.google.com/open?id=1NWC3TMsXSWN2TeoYMCjhf2N1b-WRDh-M

If you still have the zip from Assignment 1, pass that to `prepare_data.py`
and skip the download.

Pipeline check without the real images:

```bash
python scripts/make_dummy_data.py --out data_dummy
python train.py --config configs/smoke.yaml
```

## Layout

```text
configs/                  one YAML per run; best.yaml is the recipe to defend
src/ssl_assignment/       data, transforms, models, NT-Xent, training loops
train.py                  supervised, SimCLR pretrain, and joint training
evaluate.py               validation or test accuracy, plus per-class counts
predict.py                labels for a flat folder of images
scripts/prepare_data.py   unpack the dataset and check the 2,400 count
scripts/make_report_table.py
notebooks/colab_train.ipynb
reports/REPORT.md         two-page report source; fill accuracies after the runs
```

Checkpoints land in `runs/<name>_seed<seed>/`. Pretraining writes `last.pt`.
Everything that classifies writes `best.pt`, chosen by validation accuracy.
Logs without the weights are copied to `experiments/run_logs/`. Large
checkpoints should be linked from this README (Drive or Hugging Face) rather
than committed. `best.pt` stores the config, the class order, and the weights,
so evaluation rebuilds the right preprocessing.

## Rules this code follows

- ResNet-18 is constructed with `weights=None`. SmallCNN is defined in this
  repository. There is no other architecture, and no weight file is downloaded.
- SimCLR pretraining and joint training use the training split only.
- Test images are read only by `evaluate.py --split test` and `predict.py`.
