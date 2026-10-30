# AI usage

Cursor (Grok 4.7) wrote this repository: the training loop, the SimCLR loss,
the configs, and the Colab notebook. The scientific constraints came from the
assignment and from Assignment 1's measured results, not from the tool.

Choices that were specified rather than left to the tool:

- No ImageNet weights, including the ConvNeXt that scored 96.75% in Assignment 1.
  That number is not a legal target here.
- The SSL comparison uses one from-scratch ResNet-18. The SmallCNN is only a
  check against Assignment 1's 89.38% from-scratch run.
- SimCLR's crop range is 0.3 to 1.0, not the paper's 0.08 to 1.0, because these
  classes are scenes.
- Validation images are excluded from pretraining.
- The matched fine-tune copies the baseline optimizer. A separate config uses
  a smaller backbone learning rate. Those answer different questions and are
  not interchangeable.
- Joint training is reported as a change of loss and augmentation together.

The tool also started from the previous repository's stratified split, cosine
schedule, and the rule that weight decay skips biases and batch-norm scales.
Those transferred because they were about optimization on this dataset, not
about pretrained features.
