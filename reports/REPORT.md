# Self-Supervised Representations from 2,400 Images

**Smriti Bhemireddy** | ITCS 6169/8169 Computer Vision, Assignment 2
**GitHub:** https://github.com/smritib4/SSL_Assignment2

Replace every `XX.X` after the Colab runs. Do not submit this file with the
placeholders. `python scripts/make_report_table.py` prints the numbers.

## Best result

| | |
|---|---|
| **Test accuracy** | **XX.X%** |
| Validation accuracy of the selected checkpoint | XX.X% |
| Model | ResNet-18 trained from scratch, 128px RGB. No external weights. |
| Recipe | (supervised baseline / SimCLR fine-tune / joint CE + NT-Xent) |
| Reproduce | `python train.py --config configs/best.yaml` |

Selection was highest validation accuracy among finished runs, on a stratified
80/20 split of the 2,400 images (seed 0, 30 validation images per class). The
test set was scored after that choice. Assignment 1's 96.75% used an
ImageNet ConvNeXt and is not a comparable number under this assignment's rules.
The from-scratch ceiling measured last time was 89.38% validation, with an
8-convolution CNN at 128px. Re-running that network here scored XX.X%
validation, which checks that the split and the training code still see the
same problem.

## Supervised baseline

ResNet-18, random initialization, 128px, AdamW at 1e-3, weight decay 0.01,
cosine decay with 5 warmup epochs, label smoothing 0.1, 50 epochs. Augmentation
is a random resized crop (scale 0.65 to 1.0), horizontal flip, mild colour
jitter, and random erasing. Validation accuracy: **XX.X%**.

The baseline is this ResNet, not the SmallCNN. Every SSL run uses the same
ResNet, the same split, and the same 50-epoch budget where it fine-tunes, so
a difference can be attributed to the objective or the initialization.

## Self-supervised strategy

SimCLR. Two augmented views of one image are a positive pair; the other images
in the batch are negatives. The loss is NT-Xent on a 128-d projection of the
512-d encoder output. The projection is discarded. Classification uses the
encoder output, which is the layer the SimCLR paper found transfers better.

Two adaptations matter on this dataset. The crop keeps at least 30% of the
image, rather than SimCLR's 8%, because a scene label often depends on the
whole frame. Pretraining uses the 1,920 training images only. The 480
validation images are not used as extra unlabeled data, or the validation
accuracy would partly measure memorization.

The linear probe (frozen encoder, new head) scored **XX.X%**. Fine-tuning with
the baseline's exact optimizer scored **XX.X%**. Fine-tuning with a 10x smaller
learning rate on the encoder scored **XX.X%**. Joint training, cross-entropy
on both views plus 0.2 times NT-Xent, scored **XX.X%**. Joint training also
changes the augmentation to the SimCLR view policy, so it is not a pure loss
ablation.

## What the numbers say

| Method | Val accuracy |
|---|---:|
| SmallCNN from scratch (Assignment 1 architecture) | XX.X% |
| ResNet-18 supervised baseline | XX.X% |
| SimCLR linear probe | XX.X% |
| SimCLR fine-tune, matched recipe | XX.X% |
| SimCLR fine-tune, smaller backbone LR | XX.X% |
| Joint CE + NT-Xent | XX.X% |
| Selected model, test | XX.X% |

Write two or three sentences on which row won and whether the gap is larger
than about one point. On 480 images, Assignment 1 measured a same-seed spread
of 0.62 points, so a smaller gap is not evidence.

The result that needs an explanation either way: with 150 labels per class,
pure SimCLR spends 100 epochs ignoring labels the supervised model uses on
every step. A probe well below the baseline means those epochs did not recover
what the labels already provide. A matched fine-tune that only ties the
baseline means the initialization was not worth the pretraining. Joint
training is the case where SSL can still help, as a regularizer, without
throwing the labels away. If it loses, say that the SimCLR augmentation was
part of the run, so the loss is not the only thing that changed.

## Limitations

One seed, one backbone, 128px. ResNet-18's ImageNet stem (stride-2 7x7 and a
max-pool) is a lot of downsampling at this resolution; a stem built for small
images might change the from-scratch baseline. The joint run confounds the
loss with the augmentation. SimCLR's batch of 128 is small compared with the
batches the method was designed for, so the negative set is limited. None of
that is fixed by using an external pretrained model, which this assignment
does not allow.
