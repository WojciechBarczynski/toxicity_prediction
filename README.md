# Comparison of methods for toxicity prediction from SMILEs

Authors:
- Adam Mytnik
- Igor Urbanik
- Jakub Pryc
- Krzysztof Pęczek
- Szymon Rusiecki
- Wojciech Barczyński

# Problem description

The task of toxicity prediction involves determining the potential toxic effects of chemical compounds based on their molecular structure. This is a critical problem in fields such as drug discovery, environmental science, and chemical safety. In this project, we explore various computational methods to predict toxicity directly from SMILES (Simplified Molecular Input Line Entry System) representations of molecules. We aim to identify approaches that achieve the best results on tested datasets by comparing traditional baselines with advanced Graph Machine Learning models.

# Dataset selection

In this project, we primarily evaluate our models on the `peptides_func` dataset. While the `tox_21` dataset was initially considered, it proved to be significantly more challenging due to its complexity and class imbalance. The `peptides_func` dataset offers a more manageable starting point, allowing us to effectively compare the performance of different methods and gain meaningful insights into their strengths and weaknesses.

# Baseline - classical algorithms

In our baseline approach, we extract molecular fingerprints from SMILES representations using the `scikit-fingerprints` library. These fingerprints serve as feature vectors that capture the structural properties of the molecules. We then treat the toxicity prediction task as a classical classification problem, applying traditional machine learning algorithms, such as Gradient Boosting (LightGBM and XGBoost), kNN and SVM, to predict toxicity labels based on these features.

Code for the testing of all baseline algorithms can be found in `baseline.ipynb` notebook.

## Baseline results summary
The baseline methods demonstrated varying levels of performance in toxicity prediction on the `peptides_func` dataset:

- **KNN** achieved an overall accuracy of 77%, with a higher recall for non-toxic compounds (class 0) but struggled with toxic compounds (class 1).
- **XGBoost** performed slightly better, achieving an accuracy of 80%, with improved precision and recall for both classes compared to KNN.
- **LightGBM** achieved an overall accuracy of 81% and slightly better performance across precision, recall, and f1-score.
- **SVM** achieved an overall accuracy of 81%. It demonstrated high precision (81%) and recall (95%) for non-toxic compounds (class 0), but its performance for toxic compounds (class 1) was less effective, with a recall of 50% and an f1-score of 62%.
- **Random Forest** achieved an overall accuracy of 79%. It demonstrated high precision (82%) and recall (91%) for non-toxic compounds (class 0), but its performance for toxic compounds (class 1) was less effective, with a recall of 53% and an f1-score of 61%.

These results highlight the potential of classical machine learning algorithms for toxicity prediction, while also indicating room for improvement, particularly in identifying toxic compounds.

![Baseline Results](assets/baseline_results.png)

# Graph ML models

## GIN

The Graph Isomorphism Network (GIN) model achieved the following results on the `peptides_func` dataset:

- **Class 0 (Non-toxic)**: Precision 0.76, Recall 0.98, F1-score 0.85 (2153 samples)
- **Class 1 (Toxic)**: Precision 0.84, Recall 0.29, F1-score 0.43 (954 samples)
- **Overall**: Accuracy 76%, Macro F1-score 0.64, Weighted F1-score 0.72

GIN excels in identifying non-toxic compounds but struggles with toxic ones, achieving only 29% recall for class 1. This highlights the need for better handling of class imbalance.

The code for GIN can be found in `gin.ipynb` notebook.

## GCN+

TODO Krzysiu

## GRIT

TODO Kuba

# Failed attepts

## Issues with Top Leaderboard Results

During our exploration, we attempted to replicate some of the top-performing methods for toxicity prediction listed on "Papers with Code". Unfortunately, several of these methods presented significant challenges:

- **Unreplicable Results**: Some methods lacked sufficient implementation details or had discrepancies between the reported results and the available code, making it impossible to reproduce their performance.
- **Code Failures**: In certain cases, the provided codebases were incomplete, outdated, or incompatible with current software dependencies, leading to runtime errors or failures during execution.

These issues highlight the importance of reproducibility and transparency in machine learning research, as they directly impact the ability to validate and build upon existing work.

## Neural Walker

TODO Szymon

## GRED
We were unable to reproduce author's results as version of dependencies provided by them were incompatible with each other, and even after long trials to resolve these conflicts, peptides preprocessing step provided by authors requiered computional resources beyond our grasp.

## the rest stuff
