# Semantic Similarity Detection

This project measures semantic similarity between sentence pairs using traditional models (Bag-of-Words, TF-IDF) and an advanced contextual model (Sentence-BERT, SBERT). It includes datasets, code, and outputs for easy replication and analysis.
## Author
-Jingyi Zhang jyz0328@bu.edu
-Ruoxi Jin jrx99@bu.edu
## Project Structure
- **`source_dataset/`**: Contains the original STS Benchmark dataset files. We utilize the Semantic Textual Similarity Benchmark (STSb) dataset as source dataset from this link: https://huggingface.co/datasets/sentence-transformers/stsb/viewer/default/train.
  - `stsb_train.csv`: Training set.
  - `stsb_validation.csv`: Validation set.
  - `stsb_test.csv`: Test set.
- **`code/`**: Contains all the code files for training, evaluation, and visualization.
  - `baseline.py`: Implements and evaluates Bag-of-Words and TF-IDF models(baseline models).
  - `SBERT.py`: Implements and evaluates the Sentence-BERT model.
  - `figure_analysis.py`: Generates visualizations and compares metrics between models.
- **`figures/`**: Stores generated plots and charts.
  - `bow_test_histogram.png` `tfidf_test_histogram.png` `sbert_test_histogram.png`: Histogram showing the distribution of BoW/TF-IDF/SBERT model predicted-to-true score ratios on the test set.
  - `bow_test_line_chart.png` `tfidf_test_line_chart.png` `sbert_test_line_chart.png`: Line chart illustrating the frequency of predicted-to-true score ratios for the BoW/TF-IDF/SBERT model on the test set.
  - `bow_validation_histogram.png` `tfidf_validation_histogram.png` `sbert_validation_histogram.png`: Histogram showing the distribution of BoW/TF-IDF/SBERT model predicted-to-true score ratios on the validation set.
  - `bow_validation_line_chart.png` `tfidf_validation_line_chart.png` `sbert_validation_line_chart.png`: Line chart illustrating the frequency of predicted-to-true score ratios for the BoW/TF-IDF/SBERT model on the validation set.
- **`output_dataset/`**: Contains evaluation results for validation and test sets.
  - `baseline_stsb_test.csv`: Test results for Bag-of-Words and TF-IDF models.
  - `baseline_stsb_validation.csv`: Validation results for Bag-of-Words and TF-IDF models.
  - `sbert_stsb_test.csv`: Test results for the SBERT model.
  - `sbert_stsb_validation.csv`: Validation results for the SBERT model.
- **`report/`**: Includes project milestone reports.
  - `milestone3.pdf`: Milestone 3 report.
  - `milestone4.pdf`: Final milestone report.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jyz0328/Semantic-Similarity-Detection.git
   cd Semantic-Similarity-Detection
   cd code 
2. **Run Baseline Models:**:

   ```bash
   python baseline.py 
3. **Run SBERT Model:**:

   ```bash
   python SBERT.py 
4. **Generate Visualizations:**:

   ```bash
   python figure_analysis.py 
