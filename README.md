# LitGen

In collaboration with the Clinical Genomic Resource (ClinGen), the flagship NIH program for clinical curation---we propose the first machine learning system, LitGen, that can retrieve papers for a particular variant and filter them by specific evidence types used by curators to assess for pathogenicity. LitGen uses semi-supervised deep learning to predict the type of evidence provided by each paper. It is trained on papers annotated by ClinGen curators and systematically evaluated on new test data collected by ClinGen. LitGen further leverages rich human explanations and unlabeled data to gain 7.9%-12.6% relative performance improvement over models learned only on the annotated papers.

You can find our paper here: [LitGen: Genetic Literature Recommendation Guided by Human Explanations
](https://arxiv.org/abs/1909.10699)

## Dataset

All our data are public and contain no private information. 

Our raw data, extracted directly from the ClinGen Variant Curation Interface (VCI) is 
in `./corpus/` folder. The date `ML Data (as of 3_17_19).csv` indicates date of extraction.
We use the papers contained in earlier date as training dataset, and new papers added to the later date
as our evaluation (test).

We are currently talking to ClinGen VCI team to release a public API, but if not, we will consider
releasing data annually or semi-annually.

You can produce the test dataset like this:

```python
from data.clingen_raw_to_training import DatasetExtractor

old = DatasetExtractor("../corpus/ML Data (as of 3_17_19).csv")
new = DatasetExtractor("../corpus/ML Data (as of 5_1_19).csv")
de = new - old

print(len(de.major_5_pmid_to_panel))

data, _ = de.generate_pmid_panel_set(log=True)

de.write_data_to_csv(data, "../models/data/vci_358_abs_tit_key_may_7_2019_true_test.csv")
```  

The `DatasetExtractor` will actually download paper information using `Entrez`.

We also uploaded our partition to `./models/data/`, you can find it under:

```bash
vci_1543_abs_tit_key_apr_1_2019_train.csv
vci_1543_abs_tit_key_apr_1_2019_valid.csv
```

And our out-of-distribution hold-out test set is:

```bash
vci_358_abs_tit_key_may_7_2019_true_test.csv
```

## Weak Supervision Labels

We conducted various noisy-labeling techniques in our paper. We mostly focus on
leveraging information from human explanations to generate weak labels. 
We provided these weak labels in `./models/data/` as well!

| Strategy                              | Avg Accu | EM   | Weighted F1 |
| ------------------------------------- | -------- | ---- | ----------- |
| BiLSTM                                | 82.6     | 45.2 | 62.7        |
| BiLSTM + Naive Exp                    | 83.8     | 48.7 | 66.5        |
| BiLSTM + Naive Unlabeled              | 83.9     | 50.1 | 65.7        |
| BiLSTM + Naive Exp + Naive Unlabeled  | 82.9     | 48.4 | 66.4        |
| BiLSTM + Exp-guided Snorkel           | 84.0     | 50.1 | 66.8        |
| LitGen: BiLSTM + Exp-guided Unlabeled | 85.0     | 51.6 | 68.1        |

BiLSTM training command:

```bash
python3.6 train.py --exp_path ./saved/bilstm/  --epochs 5
```

BiLSTM + Naive Exp training command:

(We also refer to this as "MTL" -- multi-task learning. We experimented with a few architectures
and settled with seperate prediction head).

```bash
python3.6 train.py --exp_path ./saved/multi_task_exp_train_on_vci_no_share_decoder_aug5_2019_run3/ --epochs 5 --multi_task_data_path ./data/explanations_5panels_shuffled.csv
```

BiLSTM + Naive Unlabeled training command:

Here we use a Random Forest to generate noisy labels for unlabeled dataset.

```bash
python3.6 train.py --exp_path ./saved/weak_litvar3827_rf_cotrain_on_vci_aug5/  --weak_data_path ./data/litvar_3827_abs_tit_key_may_31_2019_random_forest_cotrain_on_vci.csv --weak_vocab --weak_loss_scale 0.3 --epochs 5

```

BiLSTM + Naive Exp + Naive Unlabeled training command:

```bash
python3.6 train.py --exp_path ./saved/multi_task_exp_weak_litvar3827_rf_cotrain_on_vci_aug5/  --weak_data_path ./data/litvar_3827_abs_tit_key_may_31_2019_random_forest_cotrain_on_vci.csv --weak_vocab --weak_loss_scale 0.3 --epochs 5 --mtl_loss_scale 0.3 --multi_task_data_path ./data/explanations_5panels_shuffled.csv --mtl_first
```

BiLSTM + Exp-guided Snorkel

(Snorkel-MeTal Using Lasso features based labeling functions)

```bash
python3.6 train.py --exp_path ./saved/weak_litvar3827_metal_on_explanations_may31_2019/  --weak_data_path ./data/litvar_3827_abs_tit_key_may_31_2019_metal_on_explanations.csv --weak_vocab --weak_loss_scale 0.3 --epochs 5
```

LitGen: BiLSTM + Exp-guided Unlabeled

(Random Forest using Lasso features)

```bash
python3.6 train.py --exp_path ./saved/weak_litvar3827_rf_tuned_on_vci_aug5_2019/  --weak_data_path ./data/litvar_3827_abs_tit_key_may_31_2019_random_forest_tuned_on_vci.csv --weak_vocab --weak_loss_scale 0.3 --epochs 5
```

## Lasso Features

We use `sklearn` Lasso model and directly predict labels using explanations. We do not provide our Lasso model, but we do provide the keyword with non-zero coefficients
as well as their coefficients.

You can find it in this file: `explanation_L1_predict_panels_v3.csv`.

## Contact Info

You can reach out to the first aruthor anie@stanford.edu for inquiry on the dataset or model!
