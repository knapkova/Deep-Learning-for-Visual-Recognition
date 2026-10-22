# Camelyon17 experiment results

We completed **180 training runs**: 3 patient splits × 5 training hospitals × 4 training ranges × 3 model seeds. Each model was tested on all five hospitals, giving **900 hospital test results**.

## Start here

- [Read the results report](reports/Camelyon17_Results_for_Group_EN.md): the questions, figures, comparisons and conclusions in plain English.
- [How the experiments worked](reports/Experiment_Setup_and_Questions_EN.md): data splits, patients, training, model selection and limitations.
- [All hospital results](reports/All_Hospital_Results_EN.md): the scores behind the comparisons.
- [Code and running instructions](code/README.md).

The reports and figures are the same as the latest reviewed English report. The saved training and test results have not been recalculated or changed for this upload.

## Where to find the evidence

| What you want to check | File or folder |
|---|---|
| All 900 hospital test results | [evaluations.csv](reports/data/evaluations.csv) |
| Best epoch and stopping point for every run | [training_audit.csv](reports/data/training_audit.csv) |
| Patient and image counts | [patient_and_image_counts.csv](reports/data/patient_and_image_counts.csv) |
| The patients assigned to train, validation and test | [data/patient_splits](data/patient_splits/) |
| Distance values for each patient split | [results/distance](results/distance/) |
| Mean scores, variation across repeats and distance matrices | [results/summary](results/summary/) |
| Training loss at every epoch and saved test metrics | [split 42](results/split_42/runs/), [split 43](results/split_43/runs/), [split 44](results/split_44/runs/) |
| Which trained models are available | [MODEL_INDEX.csv](MODEL_INDEX.csv) |

A run lives at `results/split_42/runs/source_0/layer4/seed_42/`, for example.
It contains `history.csv` (one row per epoch), `metrics.json` (saved model and test scores), `run.json` (settings) and `completion.json` (original file checksums).

Hospital IDs **0, 1, 2, 3, 4** mean **A, B, C, D, E**.
`ID` means the test hospital is the same hospital used for training; `OOD` means a different hospital.
In the report these are called **IN** and **OUT**. The IN test patients were still held out from training.

| Code name | What was trained |
|---|---|
| `head` | Classification head only |
| `layer4` | Classification head and the last 3 residual blocks |
| `layer3_layer4` | Classification head and the last 9 residual blocks |
| `full` | Whole network |

The proposal said the last one or two blocks. The implemented experiment used the last one or two **stages**, containing 3 or 9 blocks. The report explains this difference.

## Models and detailed data

The **180 trained model files total about 17 GB**. They are stored separately and are not in this branch. To request a model, choose its patient split, source hospital, training range and model seed from [MODEL_INDEX.csv](MODEL_INDEX.csv). The index includes its size and original SHA256 checksum.

The per-image prediction files and full image lists are also stored separately. The smaller files here let you read all reported scores and inspect each training curve. Recomputing every test metric from individual predictions requires those additional files. The original completion records still name model and prediction files; they describe the full experiment archive, not files included in this GitHub folder.

The original image data can be downloaded from the [pinned Camelyon17-WILDS export](https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/tree/d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f). The experiment used 455,954 exported images from 43 patients. Our patient splits are custom splits within each hospital, so these scores are not official WILDS leaderboard scores.

The three patient splits reuse the same patients. They test how much the results change when we divide those patients differently; they do not add new patients. The source hospital's validation loss selected the checkpoint. The target hospital's test scores were used only for the final comparisons.

## Integrity

`PUBLICATION_MANIFEST.json` lists the sizes and SHA256 values of the files in this folder. Reports, figures and saved result files were copied unchanged from the reviewed local experiment. The code README records the small packaging changes needed to run the published subset.
