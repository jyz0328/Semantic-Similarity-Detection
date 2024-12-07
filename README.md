# Semantic Similarity Detection

This project measures semantic similarity between sentence pairs using traditional models (Bag-of-Words, TF-IDF) and an advanced contextual model (Sentence-BERT, SBERT). It includes datasets, code, and outputs for easy replication and analysis.
## Author
-Jingyi Zhang jyz0328@bu.edu
-Ruoxi Jin jrx99@bu.edu
## Project Structure
- **`source_dataset/`**: Contains the original STS Benchmark dataset files. We utilize the Semantic Textual Similarity Benchmark (STSb) dataset as source dataset from this link: https://huggingface.co/datasets/sentence-transformers/stsb/viewer/default/train. The dataset contains three variables:Sentence 1,Sentence 2 and Score.Sentence 1 and Sentence2 are sentence pairs drawn from sources such as news headlines, video captions, and natural lan-guage inference data. Score is a human-annotated similarity score, normalized between 0 and 1. We treat this score as the true score in this project.
  - `stsb_train.csv`: Training set.
  - `stsb_validation.csv`: Validation set.
  - `stsb_test.csv`: Test set.
- **`code/`**: Contains all the code files for training, evaluation, and visualization.
  - `baseline.py`: Implements and evaluates Bag-of-Words and TF-IDF models(baseline models).
  - `SBERT.py`: Implements and evaluates the Sentence-BERT model.
  - `figure_analysis.py`: Generates visualizations and compares metrics between models.
- **`output_dataset/`**: Contains evaluation results for validation and test sets. Each set include sentence pair, original true score, model-predicted score, model predicted-to-true score ratios and  evaluation metrics(F1-Score, Accuracy, Spearman, and Pearson Correlations).
  - `baseline_stsb_test.csv`: Test results for Bag-of-Words and TF-IDF models.
  - `baseline_stsb_validation.csv`: Validation results for Bag-of-Words and TF-IDF models.
  - `sbert_stsb_test.csv`: Test results for the SBERT model.
  - `sbert_stsb_validation.csv`: Validation results for the SBERT model.
- **`figures/`**: Stores generated plots and charts.
  - `bow_test_histogram.png` `tfidf_test_histogram.png` `sbert_test_histogram.png`: Histogram showing the distribution of BoW/TF-IDF/SBERT model predicted-to-true score ratios on the test set.
  - `bow_test_line_chart.png` `tfidf_test_line_chart.png` `sbert_test_line_chart.png`: Line chart illustrating the frequency of predicted-to-true score ratios for the BoW/TF-IDF/SBERT model on the test set.
  - `bow_validation_histogram.png` `tfidf_validation_histogram.png` `sbert_validation_histogram.png`: Histogram showing the distribution of BoW/TF-IDF/SBERT model predicted-to-true score ratios on the validation set.
  - `bow_validation_line_chart.png` `tfidf_validation_line_chart.png` `sbert_validation_line_chart.png`: Line chart illustrating the frequency of predicted-to-true score ratios for the BoW/TF-IDF/SBERT model on the validation set.
  - `validation_comparison_chart.png``test_comparison_chart.png``combined_metrics_grid.png`:Compare F1-Score, Accuracy, Spearman, and Pearson metrics for Validation and Test sets across BoW, TF-IDF, and SBERT models, presented in grid and histogram formats.
  - `terminal1.png``terminal2.png`:Screenshots of terminal output from `baseline.py`, displaying baseline (BoW/TF-IDF) training results, including predicted-to-true ratio distributions and key evaluation metrics for validation and test sets.
  - `terminal3.jpg``terminal4.png`:Screenshots of terminal output from `SBERT.py`, displaying SBERT training results, including metrics such as loss, gradient, and learning rate changes during training; predicted-to-true ratio distributions; and key evaluation metrics for validation and test sets.
- **`report/`**: Includes project milestone reports.
  - `milestone3.pdf`: Milestone 3 report.
  - `milestone4.pdf`: Final milestone report.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jyz0328/Semantic-Similarity-Detection.git
   cd Semantic-Similarity-Detection
   cd code 
2. **Run Baseline Models:**

   ```bash
   python baseline.py 
3. **Run SBERT Model:**

   ```bash
   python SBERT.py 
4. **Generate Visualizations:**

   ```bash
   python figure_analysis.py 
