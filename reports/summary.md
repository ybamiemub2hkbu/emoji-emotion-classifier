# Results

| Model | Training frames | Random frame split | Unseen clips | Other designs, trained on frames (95% CI) | Unseen design, +4 designs in training (95% CI) |
|---|---|---|---|---|---|
| Classic CNN | 99.5% | 100.0% | 76.1% | 59.0% (49.8%–67.3%) | 87.9% (84.2%–91.7%) |
| Compact CNN + augmentation | 98.0% | 95.0% | 77.3% | 63.5% (51.5%–74.5%) | 76.2% (66.2%–86.2%) |
| MobileNetV2 (frozen, ImageNet) | 100.0% | 98.0% | 84.8% | 71.8% (61.3%–82.0%) | 86.7% (80.4%–92.9%) |

## Unseen clips, by held-out pair

| Held-out clips | Classic CNN | Compact CNN | MobileNetV2 |
|---|---|---|---|
| happy1+sad1 | 79.0% | 81.0% | 83.0% |
| happy1+sad2 | 75.5% | 78.0% | 83.0% |
| happy2+sad1 | 50.0% | 50.0% | 75.0% |
| happy2+sad2 | 100.0% | 100.0% | 98.0% |

## Other emoji designs, trained on the frames only

| Design | Classic CNN | Compact CNN | MobileNetV2 |
|---|---|---|---|
| Fluent 3D | 51.2% | 47.5% | 58.8% |
| Noto 3D | 57.5% | 67.5% | 90.0% |
| Noto | 78.8% | 68.8% | 86.2% |
| Twemoji | 76.2% | 71.2% | 72.5% |
| OpenMoji | 31.2% | 62.5% | 51.2% |

## Each design held out, trained on the frames + the other four designs

| Design | Classic CNN | Compact CNN | MobileNetV2 |
|---|---|---|---|
| Fluent 3D | 77.1% | 64.6% | 81.2% |
| Noto 3D | 100.0% | 77.1% | 95.8% |
| Noto | 97.9% | 81.2% | 97.9% |
| Twemoji | 93.8% | 93.8% | 91.7% |
| OpenMoji | 70.8% | 64.6% | 66.7% |

## Training cost (all 80 frames)

| Model | Epochs | Seconds |
|---|---|---|
| Classic CNN | 8.2 | 20 |
| Compact CNN + augmentation | 100.0 | 51 |
| MobileNetV2 (frozen, ImageNet) | 20.0 | 39 |

## Final holdout (EmojiOne 2.2, never used in any experiment)

Demo model (MobileNetV2 trained on all 160 labeled images): 15/16 correct (93.8%).
Wrong: grinning face.

Runs: 135 (seeds per configuration: leave_clips_out 5, leave_design_out 3, other_designs 5, random_split 5).
