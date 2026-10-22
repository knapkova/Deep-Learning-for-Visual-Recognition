# Test results on the same hospital and other hospitals after training on each hospital

The first three tables compare the AUC, accuracy, and loss of the four training scopes for a fixed training hospital and a fixed different test hospital. The final table lists the same-hospital test scores. All scores come from held-out test patients who did not take part in training the corresponding model.

For example, A → C means training only on images from A's training patients, then using that model to test C's test patients. Each score is the average of nine results: three sets of patient lists, with three training seeds for each set.

The four training scopes are: Head only, which updates only the classification head; Head + last 3 blocks, which updates the classification head and the last 3 blocks (`layer4`); Head + last 9 blocks, which updates the classification head and the last 9 blocks (`layer3` + `layer4`); and Whole network, which updates the entire network. “Head + last 3 blocks” and “Head + last 9 blocks” both include training the classification head.

“Distance” describes the difference between the image feature distributions of two hospitals. We use the same fixed ImageNet model to extract features from each hospital's training images, calculate the distance, and then average over the three patient splits. A larger value means a larger difference in these features. Each hospital pair uses the same distance for all four training scopes. Within each training hospital, rows are ordered from the smallest to the largest distance.

Read across a row to compare the four training scopes for the same training and test task. Bold marks only the highest AUC, highest accuracy, or lowest loss in that row. It does not mean that a scope is best across all hospitals and all metrics. These comparisons use test results after training and cannot be used as a way to choose a model before testing.

## Cross-hospital testing: AUC

Fix the training and test hospitals in a row, then compare the AUC of the four training scopes. A higher AUC means the model is better at giving tumour images higher scores. More precisely, randomly take one tumour image and one non-tumour image. AUC measures the proportion of pairs where the former scores higher than the latter, counting a tie as half. It is not the percentage of images classified correctly.

| Train → Test | Distance | Head only | Head + last 3 blocks | Head + last 9 blocks | Whole network |
|---|---:|---:|---:|---:|---:|
| A → D | 0.05984 | 0.8416 | 0.8948 | 0.9024 | **0.9234** |
| A → B | 0.07537 | 0.8439 | 0.8749 | **0.9284** | 0.9135 |
| A → C | 0.08839 | 0.9363 | **0.9680** | 0.9543 | 0.9300 |
| A → E | 0.10608 | 0.9406 | 0.9604 | 0.9537 | **0.9654** |
| B → D | 0.06605 | 0.8034 | 0.8260 | 0.8487 | **0.8749** |
| B → A | 0.07537 | 0.9113 | **0.9281** | 0.9130 | 0.9225 |
| B → C | 0.09193 | 0.5980 | 0.6517 | 0.6958 | **0.7582** |
| B → E | 0.10347 | 0.7899 | 0.8336 | **0.8632** | 0.8434 |
| C → A | 0.08839 | 0.9027 | **0.9044** | 0.8922 | 0.8227 |
| C → D | 0.09094 | 0.6796 | 0.7562 | **0.8013** | 0.6723 |
| C → B | 0.09193 | 0.6152 | 0.7665 | **0.7938** | 0.6835 |
| C → E | 0.09891 | 0.9357 | 0.9400 | **0.9587** | 0.9147 |
| D → A | 0.05984 | 0.9215 | 0.9628 | 0.9631 | **0.9767** |
| D → B | 0.06605 | 0.8857 | **0.9082** | 0.8644 | 0.8792 |
| D → C | 0.09094 | 0.8930 | **0.9005** | 0.8554 | 0.8283 |
| D → E | 0.10475 | 0.9381 | 0.8778 | 0.8174 | **0.9470** |
| E → C | 0.09891 | 0.9308 | **0.9635** | 0.9404 | 0.8949 |
| E → B | 0.10347 | 0.7371 | 0.7463 | **0.8101** | 0.7649 |
| E → D | 0.10475 | 0.8043 | 0.7734 | **0.8487** | 0.8071 |
| E → A | 0.10608 | 0.8054 | 0.9118 | **0.9335** | 0.8674 |

## Cross-hospital testing: accuracy (%)

Again, compare the four training scopes within each row. Here the metric is the proportion of images classified correctly using a fixed threshold, and higher is better. Which class has more images in the test set affects accuracy. Read it together with the AUC above and the errors for each class.

| Train → Test | Distance | Head only | Head + last 3 blocks | Head + last 9 blocks | Whole network |
|---|---:|---:|---:|---:|---:|
| A → D | 0.05984 | 94.38 | 97.31 | 98.46 | **98.67** |
| A → B | 0.07537 | 85.44 | 82.31 | **86.77** | 82.66 |
| A → C | 0.08839 | 80.50 | **83.97** | 76.64 | 72.43 |
| A → E | 0.10608 | 83.72 | 88.46 | 86.28 | **90.63** |
| B → D | 0.06605 | 93.35 | 96.04 | 97.43 | **98.32** |
| B → A | 0.07537 | **83.34** | 83.14 | 77.14 | 79.93 |
| B → C | 0.09193 | **46.22** | 44.94 | 45.55 | 43.44 |
| B → E | 0.10347 | **74.62** | 66.78 | 69.56 | 55.16 |
| C → A | 0.08839 | **85.58** | 83.26 | 83.82 | 82.80 |
| C → D | 0.09094 | 94.81 | 94.86 | 94.16 | **95.95** |
| C → B | 0.09193 | 67.06 | **69.38** | 62.03 | 68.99 |
| C → E | 0.09891 | 68.57 | **78.25** | 76.79 | 74.47 |
| D → A | 0.05984 | 90.34 | 92.60 | 91.90 | **92.88** |
| D → B | 0.06605 | 86.15 | 91.50 | **92.54** | 91.10 |
| D → C | 0.09094 | **80.17** | 73.53 | 69.76 | 67.04 |
| D → E | 0.10475 | **81.86** | 62.09 | 64.00 | 77.32 |
| E → C | 0.09891 | 78.88 | **86.61** | 67.57 | 48.55 |
| E → B | 0.10347 | 82.82 | 83.87 | **86.26** | 78.91 |
| E → D | 0.10475 | 96.27 | 96.51 | 98.32 | **98.77** |
| E → A | 0.10608 | 90.17 | **92.62** | 87.77 | 79.15 |

## Cross-hospital testing: loss

Here we compare how far the predicted probabilities are from the true labels for the four training scopes. Lower loss is better. Within the same row, the lowest loss, highest AUC, and highest accuracy may come from different training scopes. A better score on one metric cannot be described as better scores on all metrics.

| Train → Test | Distance | Head only | Head + last 3 blocks | Head + last 9 blocks | Whole network |
|---|---:|---:|---:|---:|---:|
| A → D | 0.05984 | 0.1534 | 0.0816 | 0.0536 | **0.0483** |
| A → B | 0.07537 | **0.3487** | 0.5605 | 0.3528 | 0.5315 |
| A → C | 0.08839 | **0.5101** | 0.6342 | 0.8941 | 0.9091 |
| A → E | 0.10608 | 0.3664 | 0.4226 | 0.4434 | **0.2833** |
| B → D | 0.06605 | 0.1691 | 0.1116 | 0.0819 | **0.0618** |
| B → A | 0.07537 | **0.5388** | 0.5530 | 0.9469 | 0.7830 |
| B → C | 0.09193 | 1.7526 | 2.1193 | 2.4632 | **1.5830** |
| B → E | 0.10347 | **0.5827** | 0.7050 | 0.7637 | 1.2391 |
| C → A | 0.08839 | **0.3883** | 0.6182 | 0.6207 | 0.8458 |
| C → D | 0.09094 | 0.1535 | 0.1560 | 0.1809 | **0.1323** |
| C → B | 0.09193 | **1.3580** | 2.0289 | 1.7917 | 2.0448 |
| C → E | 0.09891 | **0.7861** | 0.9009 | 0.9765 | 1.3801 |
| D → A | 0.05984 | 0.2812 | 0.3009 | 0.3226 | **0.2737** |
| D → B | 0.06605 | 0.4849 | 0.3907 | **0.3063** | 0.3156 |
| D → C | 0.09094 | **0.5169** | 1.5828 | 2.3646 | 1.7698 |
| D → E | 0.10475 | **0.4138** | 2.7749 | 4.6946 | 0.8107 |
| E → C | 0.09891 | 0.7541 | **0.6023** | 1.9365 | 2.8942 |
| E → B | 0.10347 | **0.9620** | 1.0272 | 1.2392 | 1.6073 |
| E → D | 0.10475 | 0.1212 | 0.1193 | **0.0839** | 0.1236 |
| E → A | 0.10608 | 0.3229 | **0.2530** | 0.6154 | 2.2573 |

## Same-hospital testing (IN, also called ID)

Here we compare the scores of the four training scopes when the training hospital and test hospital are the same. For example, all four rows for A use only images from A's training patients for training, then test on A's held-out test patients. The test patients did not take part in training. These are test scores on new patients from the same hospital, not training-set scores.

When comparing the four rows for one hospital, higher AUC and accuracy are better, and lower loss is better. Each score is still the average of nine results.

| Hospital | Training scope | AUC | Accuracy (%) | Loss |
|---|---|---:|---:|---:|
| A | Head only | 0.9268 | 90.89 | 0.2380 |
| A | Head + last 3 blocks | 0.9309 | 91.76 | 0.3179 |
| A | Head + last 9 blocks | 0.9325 | 90.55 | 0.3779 |
| A | Whole network | 0.9470 | 91.57 | 0.3514 |
| B | Head only | 0.8338 | 81.62 | 0.9014 |
| B | Head + last 3 blocks | 0.8737 | 81.66 | 0.9703 |
| B | Head + last 9 blocks | 0.9166 | 82.74 | 0.8148 |
| B | Whole network | 0.9266 | 84.64 | 0.6174 |
| C | Head only | 0.9485 | 73.91 | 0.7496 |
| C | Head + last 3 blocks | 0.9680 | 76.93 | 0.8909 |
| C | Head + last 9 blocks | 0.9748 | 76.89 | 0.8127 |
| C | Whole network | 0.9766 | 76.08 | 0.9157 |
| D | Head only | 0.8569 | 95.38 | 0.1282 |
| D | Head + last 3 blocks | 0.9104 | 98.20 | 0.0669 |
| D | Head + last 9 blocks | 0.9007 | 98.45 | 0.0646 |
| D | Whole network | 0.9498 | 98.85 | 0.0408 |
| E | Head only | 0.9151 | 71.69 | 0.9628 |
| E | Head + last 3 blocks | 0.8741 | 64.79 | 1.9947 |
| E | Head + last 9 blocks | 0.8132 | 57.15 | 2.9525 |
| E | Whole network | 0.9163 | 66.62 | 1.8080 |

[Back to the results report](Camelyon17_Results_for_Group_EN.md)
