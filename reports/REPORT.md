# Self-Supervised Representations from 2,400 Images

**Smriti Bhemireddy** | ITCS 6169/8169 Computer Vision, Assignment 2
**GitHub:** https://github.com/smritib4/SSL_Assignment2

## 1. Final Result

| | |
|---|---|
| **Highest test accuracy measured** | **78.75%** (joint training, 315 of 400) |
| Selected model test accuracy | 77.50% (310 of 400), top-5 98.50% |
| Validation accuracy of the selected model | 83.96% |
| Model | ResNet-18, `weights=None`, 11.2M parameters, all trained |
| Input | 128 x 128 RGB. Pretrain 26 min, fine-tune 5 min, one Tesla T4 |
| Reproduce | `python evaluate.py --checkpoint runs/simclr_finetune_seed0/best.pt --split test` |
| Checkpoint | https://drive.google.com/file/d/1Z_jriKslHhRE8CnO6s7U6Ldp7GEvdcph/view?usp=sharing |

The supervised baseline is the same ResNet-18 at 128px, trained with cross-entropy only: AdamW at 1e-3, weight decay 0.01, cosine decay with 5 warmup epochs, label smoothing 0.1, 50 epochs, crop scale 0.65 to 1.0, horizontal flip, mild colour jitter, and random erasing. It scores 75.00% test. The selected model is +2.50 points over that baseline. Assignment 1's 96.75% used an ImageNet ConvNeXt and is not a legal comparison here.

How I validated. Same stratified 80/20 split as Assignment 1, seed 0, so each class contributes 30 of the 480 validation images. The selection rule was fixed first: highest validation accuracy among finished ResNet-18 runs. A gap under about 1 point is a tie, because Assignment 1 measured a same-seed spread of 0.62 points on this validation set. The test set was scored after that choice. Joint training then scored 78.75% test, 1.25 points above the selected model. That is five images. I did not replace the checkpoint.

| Method | Test accuracy |
|---|---:|
| Supervised baseline | 75.00% |
| SSL + fine-tuning | 77.50% |
| Joint SSL + classification | 78.75% |
| Best model | 77.50% |

The best-model row is the SSL fine-tune. A second fine-tune that kept the baseline learning rate scored 76.75% test and 83.13% validation. Those two validation numbers differ by 0.83 points, inside the noise floor, so both test numbers are reported.

## 2. Secret Recipe

SimCLR for 100 epochs on the 1,920 training images, batch 128, temperature 0.2, then 50 epochs of fine-tuning with the backbone at 1e-4 and the new head at 1e-3. AdamW, weight decay 0.01, cosine decay with warmup, label smoothing 0.1. Two views of one image are a positive pair. The other images in the batch are negatives. The loss is NT-Xent on a 128-d projection of the 512-d encoder. The projection is thrown away. The classifier reads the encoder output.

Why this recipe, before the score.

ImageNet weights are illegal, and they were the whole Assignment 1 result. A frozen ResNet-18 head already reached 91%, while a from-scratch CNN reached 89.38%. The features have to be learned on these images. SimCLR does that without labels. I did not use the paper's 8% minimum crop. These labels are scenes, and a crop that small often misses the scene, so the crop keeps at least 30%. The 480 validation images stay out of pretraining. Otherwise the validation score would partly measure memorization.

The labels are too useful to throw away at classification time. Assignment 1's from-scratch network already reached the high 80s, so the plan was to pretrain without labels and then fine-tune with them. A matched fine-tune copies the baseline optimizer exactly. The only change is the encoder's initialization. That run is why the initialization is in the recipe: 83.13% validation against the baseline's 79.58%, and 76.75% test against 75.00%.

The smaller backbone learning rate is in the submitted config for the same reason it was in Assignment 1. After pretraining, the encoder already holds features and the 16-way head is random, so the head gets 10 times the learning rate. That run scored 83.96% validation and is the selected checkpoint. It is only 0.83 points above the matched fine-tune, inside the noise floor, so I do not claim the learning-rate split caused the gain. It is in the recipe because it won the selection rule. The comparison that isolates one change supports the SimCLR initialization.

`configs/best.yaml` is this recipe. It was not trained a second time. The scored weights are `runs/simclr_finetune_seed0/best.pt`.

## 3. Experimental Journey

The first table is validation, which is what selection used.

| # | Experiment | What changed | Val acc (%) |
|---|---|---|---:|
| 00 | SmallCNN from scratch | Assignment 1 architecture, a code check | 88.75 |
| 01 | ResNet-18 supervised | baseline, no SSL | 79.58 |
| 02 | SimCLR pretrain | NT-Xent, labels unused | n/a, loss 1.74 |
| 03 | Fine-tune, same optimizer | initialization only | 83.13 |
| 04 | Fine-tune, backbone 1e-4 | selected recipe | 83.96 |
| 05 | Joint CE + 0.2 NT-Xent | loss and SimCLR augmentation together | 81.46 |

SmallCNN at 88.75% is against 89.38% last time, so the split and the training code still see the same problem. It is a different architecture, so it is not the SSL control and it was not scored on the test set. The control is row 01. Every SSL row uses that same ResNet-18 and the same split.

## 4. Failure Analysis: joint training

What I tried. Each step applies cross-entropy to both views and adds 0.2 times NT-Xent, for 50 epochs. This was the recipe I expected to submit. `configs/best.yaml` started as a copy of it.

Why I expected it to work. With 150 labels per class, pure SimCLR spends 100 epochs ignoring labels the baseline uses on every step. Assignment 1 showed those labels are enough for a from-scratch network to reach the high 80s. Keeping the classification loss, and using NT-Xent only as a regularizer, looked like the way SSL could still help.

What happened. 81.46% validation, 2.5 points behind the selected fine-tune. Test accuracy was 78.75%, higher than 77.50%. The checkpoint stayed where validation put it.

What I learned. The joint run also switched to the SimCLR view policy: a wider crop, blur, and grayscale, and no random erasing. A higher test number is not evidence for the loss alone. Row 03 is the comparison that changes one thing. SSL also did not remove overfitting. Training accuracy ended at 98.85% for the baseline and 98.39% for the selected fine-tune, while validation stayed at 79.58% and 83.96%. The three weakest test classes of the best model are Industrial (11/25, 44%), Kitchen (12/25, 48%), and LivingRoom (15/25, 60%). Flower, Forest, Highway, and Suburb were perfect. The remaining errors are indoor scenes that share a layout.

## 5. Limitations

One seed and one backbone, so I cannot say whether a different architecture would change the SSL gain. SmallCNN still leads this ResNet on validation, and ResNet-18's stem downsamples hard at 128px, so architecture is still part of the gap. A batch of 128 is small for NT-Xent. No external pretrained weights were used.

## 6. AI + Human

Cursor wrote the training loop, the NT-Xent loss, the configs, and the Colab commands. The constraints came from the assignment and from Assignment 1's measured results.

Where I overruled it. It started downloading the dataset one image per request, the same crawl I stopped in Assignment 1. I used one zip, and `scripts/prepare_data.py` checks the 2,400 and 400 counts. I refused a CPU training run after one epoch took about five minutes. When the results table named SmallCNN the leader, I did not score it on the test set. When joint training scored higher on the test set, I did not change the checkpoint.
