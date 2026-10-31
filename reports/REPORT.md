# Self-Supervised Representations from 2,400 Images

**Smriti Bhemireddy** | ITCS 6169/8169 Computer Vision, Assignment 2
**GitHub:** https://github.com/smritib4/SSL_Assignment2

## Best result

| | |
|---|---|
| **Test accuracy** | **77.50%** |
| Validation accuracy of the selected checkpoint | 83.96% |
| Model | ResNet-18 from scratch (`weights=None`), 128px RGB. No external weights. |
| Recipe | SimCLR on the 1,920 training images, then fine-tune with backbone learning rate 1e-4 and head learning rate 1e-3 |
| Checkpoint | `runs/simclr_finetune_seed0/best.pt` |
| Reproduce | `python evaluate.py --checkpoint runs/simclr_finetune_seed0/best.pt --split test` |

Selection was the highest validation accuracy among the ResNet-18 runs, on a stratified 80/20 split of the 2,400 images (seed 0: 1,920 train, 480 validation, 30 validation images per class). The test set was scored after that choice and was not used to swap checkpoints. SmallCNN, the Assignment 1 architecture, scored 88.75% validation here, against 89.38% last time. That is a code check, not a row in the table below.

## Test accuracy

| Method | Test accuracy |
|---|---:|
| Supervised baseline | 75.00% |
| SSL + fine-tuning | 77.50% |
| Joint SSL + classification | 78.75% |
| Best model | 77.50% |

The best-model row is the same checkpoint as SSL + fine-tuning. A second fine-tune that kept the baseline learning rate scored 83.13% validation and 76.75% test. Those validation numbers differ by 0.83 points, inside the 1-point noise floor on 480 images, so both test numbers are reported. The 77.50% row is the fine-tune with the higher validation accuracy.

## Supervised baseline

ResNet-18, random initialization, 128px, AdamW at 1e-3, weight decay 0.01, cosine decay with 5 warmup epochs, label smoothing 0.1, 50 epochs. Augmentation is a random resized crop (scale 0.65 to 1.0), horizontal flip, mild colour jitter, and random erasing. Validation accuracy was 79.58% at epoch 45. Test accuracy was 75.00%.

The baseline is this ResNet, not the SmallCNN. Every SSL run uses the same ResNet and the same split, so a difference comes from the objective or the initialization.

## Self-supervised strategy

SimCLR. Two augmented views of one image are a positive pair; the other images in the batch are negatives. The loss is NT-Xent at temperature 0.2 on a 128-d projection of the 512-d encoder output. The projection is discarded. Classification reads the encoder output.

The crop keeps at least 30% of the image, rather than SimCLR's 8%, because a scene label often depends on the whole frame. Pretraining uses the 1,920 training images only, for 100 epochs at batch 128. Final NT-Xent was 1.74. The 480 validation images were not used as unlabeled data.

Fine-tuning with the baseline's optimizer scored 83.13% validation. Fine-tuning with a 10x smaller encoder learning rate scored 83.96% validation. Joint training, cross-entropy on both views plus 0.2 times NT-Xent, scored 81.46% validation. Joint training also uses the SimCLR view policy, so it changes the loss and the augmentation together.

## What the numbers say

SSL fine-tuning beat the supervised baseline on the test set by 2.50 points (77.50% versus 75.00%). The matched-learning-rate fine-tune also beat it, by 1.75 points (76.75%). Both gaps are larger than one point. The matched run changes only the initialization, so the pretrained encoder is the reason.

Joint training lost the validation comparison, 81.46% versus 83.96%. Its test accuracy, 78.75%, is 1.25 points above the selected model, which is five images out of 400. The checkpoint was already chosen, so that does not replace the best model. The joint run also changed the augmentation, so the test edge is not a pure loss result.

Training accuracy ended at 98.85% for the baseline and 98.39% for the selected fine-tune, while validation stayed at 79.58% and 83.96%. Both runs memorize the training split. The three weakest test classes of the best model are Industrial (11/25, 44%), Kitchen (12/25, 48%), and LivingRoom (15/25, 60%). Flower, Forest, Highway, and Suburb were perfect. The remaining errors are indoor scenes that share furniture and layout.

## Limitations

One seed, one backbone, 128px. ResNet-18's stem downsamples hard at this resolution, and SmallCNN still leads it on validation, so architecture is still part of the gap. The joint run confounds the loss with the augmentation. A batch of 128 is small for NT-Xent. No external pretrained weights were used.
