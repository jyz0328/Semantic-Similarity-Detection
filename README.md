# Semantic Similarity Detection

This project measures semantic similarity between sentence pairs using traditional models (Bag-of-Words, TF-IDF) and an advanced contextual model (Sentence-BERT, SBERT). It includes datasets, code, and outputs for easy replication and analysis.

## Project Structure

- **`code/`**: Contains all the code files for training, evaluation, and visualization.
  - `SBERT.py`: Implements and evaluates the Sentence-BERT model.
  - `baseline.py`: Implements Bag-of-Words and TF-IDF models for baseline comparison.
  - `figure_analysis.py`: Generates visualizations and compares metrics between models.
- **`figures/`**: Stores generated plots and charts.
- **`output_dataset/`**: Contains evaluation results for validation and test sets.
  - `baseline_stsb_test.csv`: Test results for Bag-of-Words and TF-IDF models.
  - `baseline_stsb_validation.csv`: Validation results for Bag-of-Words and TF-IDF models.
  - `sbert_stsb_test.csv`: Test results for the SBERT model.
  - `sbert_stsb_validation.csv`: Validation results for the SBERT model.
- **`report/`**: Includes project milestone reports.
  - `milestone3.pdf`: Milestone 3 report.
  - `milestone4.pdf`: Final milestone report.
- **`source_dataset/`**: Contains the original STS Benchmark dataset files.
  - `stsb_train.csv`: Training set.
  - `stsb_validation.csv`: Validation set.
  - `stsb_test.csv`: Test set.

## How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/jyz0328/Semantic-Similarity-Detection.git
   cd Semantic-Similarity-Detection
   cd code ```
2. **Run Baseline Models:**:

   ```bash
   python baseline.py ```
3. **Run SBERT Model:**:

   ```bash
   python SBERT.py ```
4. **Generate Visualizations:**:

   ```bash
   python figure_analysis.py ```
