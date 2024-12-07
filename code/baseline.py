import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集 load the dataset
def load_datasets():
    train_df = pd.read_csv('../source_dataset/stsb_train.csv')
    validation_df = pd.read_csv('../source_dataset/stsb_validation.csv')
    test_df = pd.read_csv('../source_dataset/stsb_test.csv')
    return train_df, validation_df, test_df

# 提取句子对和真实分数 Extract sentence pairs and true scores
def get_sentence_pairs(df):
    return df[['sentence1', 'sentence2']].values.tolist(), df['score'].tolist()

# 计算余弦相似度 Calculate cosine similarity
def compute_similarity(pairs, sentence_vectors):
    predicted_scores = []
    for i, (sent1, sent2) in enumerate(pairs):
        vec1 = sentence_vectors[i * 2]
        vec2 = sentence_vectors[i * 2 + 1]
        cos_sim = cosine_similarity([vec1], [vec2])[0][0]
        predicted_scores.append(cos_sim)
    return predicted_scores

# 评估模型性能 Evaluate model performance
def evaluate_model(predicted_scores, true_scores):
    pearson_corr, _ = pearsonr(predicted_scores, true_scores)
    spearman_corr, _ = spearmanr(predicted_scores, true_scores)
    return pearson_corr, spearman_corr

# 计算分类指标 Compute classification metrics
def evaluate_classification_metrics(predicted_scores, true_scores, threshold=0.7):
    binary_labels = [1 if score >= threshold else 0 for score in true_scores]
    predicted_binary_labels = [1 if score >= threshold else 0 for score in predicted_scores]
    accuracy = accuracy_score(binary_labels, predicted_binary_labels)
    f1 = f1_score(binary_labels, predicted_binary_labels)
    return accuracy, f1

# 保存结果到文件 Save results to file
def save_combined_results(df, bow_scores, tfidf_scores, true_scores, output_csv):
    """
    将 BoW 和 TF-IDF 的结果存入同一个 CSV 文件 Save BoW and TF-IDF results into a CSV file
    """
    combined_df = df.copy()
    combined_df['BoW score'] = bow_scores
    combined_df['TF-IDF score'] = tfidf_scores

    # 计算并保存评估指标 Calculate and save evaluation metrics
    bow_pearson, bow_spearman = evaluate_model(bow_scores, true_scores)
    tfidf_pearson, tfidf_spearman = evaluate_model(tfidf_scores, true_scores)

    bow_accuracy, bow_f1 = evaluate_classification_metrics(bow_scores, true_scores)
    tfidf_accuracy, tfidf_f1 = evaluate_classification_metrics(tfidf_scores, true_scores)

    # 添加到第一行 Add to the first row
    combined_df.at[0, 'BoW Pearson Correlation'] = bow_pearson
    combined_df.at[0, 'BoW Spearman Rank Correlation'] = bow_spearman
    combined_df.at[0, 'BoW Accuracy'] = bow_accuracy
    combined_df.at[0, 'BoW F1-Score'] = bow_f1

    combined_df.at[0, 'TF-IDF Pearson Correlation'] = tfidf_pearson
    combined_df.at[0, 'TF-IDF Spearman Rank Correlation'] = tfidf_spearman
    combined_df.at[0, 'TF-IDF Accuracy'] = tfidf_accuracy
    combined_df.at[0, 'TF-IDF F1-Score'] = tfidf_f1

    # 计算每行的 BoW/True 和 TF-IDF/True 比例 Calculate the BoW/True and TF-IDF/True ratios for each row
    combined_df['BoW/True'] = combined_df['BoW score'] / combined_df['score']
    combined_df['TF-IDF/True'] = combined_df['TF-IDF score'] / combined_df['score']
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.fillna(0, inplace=True)

    # 保存处理后的数据到 CSV 文件 Save the processed data to a CSV file
    combined_df.to_csv(output_csv, index=False)

# 绘制直方图 Plot histogram
def plot_histogram(predicted_scores, true_scores, model_name, output_image, color, dataset_name):
    # 转换为 NumPy 数组 Convert to NumPy arrays
    predicted_scores = np.array(predicted_scores)
    true_scores = np.array(true_scores)

    # 过滤掉 True score 或 Predicted score 为 0 的情况 Filter out cases where True score or Predicted score is 0
    valid_indices = (true_scores != 0) & (predicted_scores != 0)
    filtered_predicted_scores = predicted_scores[valid_indices]
    filtered_true_scores = true_scores[valid_indices]

    # 计算 Predicted/True Ratio Calculate Predicted/True Ratio
    predicted_true_ratio = filtered_predicted_scores / filtered_true_scores

    # 绘制直方图 Plot histogram
    bins = np.linspace(0, 5, 50)
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_true_ratio, bins=bins, color=color, alpha=0.7, label=f'{model_name} Predicted/True Ratio')
    plt.title(f'Distribution of {model_name} Predicted/True Ratios ({dataset_name})', fontsize=24)
    plt.xlabel(f'{model_name} Predicted-to-True Score Ratio', fontsize=24)
    plt.ylabel('Counts(Times)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()


def modify_and_save_csv(input_csv, output_csv):
    """
    修改指定 CSV 文件，更新列名和顺序，并保存为新的文件 Modify the specified CSV file, update column names and order, and save it as a new file
    """
    # 加载 CSV 文件 Load the CSV file
    df = pd.read_csv(input_csv)

    # 修改列名 Rename the columns
    df.rename(columns={
        'score': 'True score',
        'BoW Pearson Correlation': 'BoW Pearson',
        'BoW Spearman Rank Correlation': 'BoW Spearman',
        'TF-IDF Pearson Correlation': 'TF-IDF Pearson',
        'TF-IDF Spearman Rank Correlation': 'TF-IDF Spearman',
        'BoW F1-Score': 'BoW F1',
        'TF-IDF F1-Score': 'TF-IDF F1'
    }, inplace=True)

    # 调整列的顺序 Rearrange the column order
    new_column_order = [
        'sentence1', 'sentence2', 'True score',
        'BoW score', 'BoW/True', 'BoW F1', 'BoW Accuracy', 'BoW Spearman', 'BoW Pearson',
        'TF-IDF score', 'TF-IDF/True', 'TF-IDF F1', 'TF-IDF Accuracy', 'TF-IDF Spearman', 'TF-IDF Pearson'
    ]
    df = df[new_column_order]

    # 保存为新的 CSV 文件 Save as a new CSV file
    df.to_csv(output_csv, index=False)

def plot_line_chart(predicted_scores, true_scores, model_name, output_image,color, dataset_name):
    # 转换为 NumPy 数组 Convert to NumPy array
    predicted_scores = np.array(predicted_scores)
    true_scores = np.array(true_scores)

    # 过滤掉 True score 或 Predicted score 为 0 的情况 Filter out cases where True score or Predicted score is 0
    valid_indices = (true_scores != 0) & (predicted_scores != 0)
    filtered_predicted_scores = predicted_scores[valid_indices]
    filtered_true_scores = true_scores[valid_indices]

    # 检查过滤后的数据是否为空 Check if the filtered data is empty
    if len(filtered_predicted_scores) == 0:
        print(f"No valid data for {model_name} in {dataset_name}. Skipping line chart.")
        return

    # 计算 Predicted / True Ratios Calculate Predicted / True Ratios
    predicted_true_ratio = filtered_predicted_scores / filtered_true_scores

    # 定义直方图区间（bins）Define histogram bins
    bins = np.linspace(0, 5, 50)
    freq, edges = np.histogram(predicted_true_ratio, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2  # 计算每个区间的中心点

    # 计算频率为 ratio count / total count, Calculate frequency as ratio count / total count
    normalized_freq = freq / freq.sum()
    # 找到频率最高的 bin, Find the bin with the highest frequency
    max_freq_index = np.argmax(normalized_freq)
    max_freq_value = normalized_freq[max_freq_index]
    max_bin_range = (edges[max_freq_index], edges[max_freq_index + 1])

    # 找到频率最高的 3 个 bin, Find the top 3 bins with the highest frequencies
    top_3_indices = np.argsort(normalized_freq)[-3:][::-1]
    top_3_freqs = normalized_freq[top_3_indices]
    top_3_bins = [(edges[i], edges[i + 1]) for i in top_3_indices]

    # 打印最高的3个频率的信息到 terminal, Print the top 3 highest frequency and corresponding bins to the terminal
    print(f"\n{model_name} - {dataset_name} - 3 Most Frequent Predicted-to-True Ratio Ranges")
    for i, (freq_val, bin_range) in enumerate(zip(top_3_freqs, top_3_bins)):
        print(f"{i+1} - Bin Range: [{bin_range[0]:.2f}, {bin_range[1]:.2f}] - Frequency: {freq_val:.4f}")
    # 绘制折线图 Plot a line chart
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers,normalized_freq, color=color,label=f'{model_name} Ratio Frequency', marker='o')
    plt.title(f'{model_name} Predicted/True Ratios Frequency ({dataset_name})', fontsize=24)
    plt.xlabel(f'{model_name} Predicted-to-True Score Ratio', fontsize=24)
    plt.ylabel('Frequency (Count / All Count)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    # 添加一个方框标注最高频率点和对应bin , Add a box annotation to highlight the highest frequency point and its corresponding bin
    plt.text(
        bin_centers[max_freq_index], 
        max_freq_value + 0.02, 
        f'Max Freq Bin Range: [{max_bin_range[0]:.2f}, {max_bin_range[1]:.2f}]\nMax Freq: {max_freq_value:.4f}', 
        fontsize=16, 
        ha='center', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )
    plt.legend()
    plt.tight_layout()
    # 保存折线图 Save the line chart
    try:
        plt.savefig(output_image)
    except Exception as e:
        print(f"Failed to save line chart: {e}")
    
    plt.show()


# 主程序 main
if __name__ == "__main__":
    # 加载数据集 Load the dataset
    train_df, validation_df, test_df = load_datasets()

    # 提取句子对和真实分数 Extract sentence pairs and true scores
    train_pairs, train_scores = get_sentence_pairs(train_df)
    val_pairs, val_scores = get_sentence_pairs(validation_df)
    test_pairs, test_scores = get_sentence_pairs(test_df)

    # 提取所有句子用于 BoW 和 TF-IDF 向量化 Extract all sentences for BoW and TF-IDF vectorization
    all_sentences = [sentence for pair in train_pairs + val_pairs + test_pairs for sentence in pair]

    # Bag-of-Words 模型 Bag-of-Words model: Convert sentences into Bag-of-Words vectors
    vectorizer = CountVectorizer()
    sentence_vectors_bow = vectorizer.fit_transform(all_sentences).toarray()

    # TF-IDF 模型  TF-IDF model: Convert sentences into TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    sentence_vectors_tfidf = tfidf_vectorizer.fit_transform(all_sentences).toarray()

    # 在开发集上评估模型 Evaluate models on the validation set
    val_predicted_scores_bow = compute_similarity(val_pairs, sentence_vectors_bow)
    val_predicted_scores_tfidf = compute_similarity(val_pairs, sentence_vectors_tfidf)
    save_combined_results(validation_df, val_predicted_scores_bow, val_predicted_scores_tfidf, val_scores, '../output_dataset/baseline_stsb_validation.csv')
    print("=== Validation Performance (BoW) ===")
    print(f"Pearson Correlation: {evaluate_model(val_predicted_scores_bow, val_scores)[0]:.4f}")
    print(f"Spearman Rank Correlation: {evaluate_model(val_predicted_scores_bow, val_scores)[1]:.4f}")
    print(f"Accuracy: {evaluate_classification_metrics(val_predicted_scores_bow, val_scores)[0]:.4f}")
    print(f"F1-Score: {evaluate_classification_metrics(val_predicted_scores_bow, val_scores)[1]:.4f}")
    print("=== Validation Performance (TF-IDF) ===")
    print(f"Pearson Correlation: {evaluate_model(val_predicted_scores_tfidf, val_scores)[0]:.4f}")
    print(f"Spearman Rank Correlation: {evaluate_model(val_predicted_scores_tfidf, val_scores)[1]:.4f}")
    print(f"Accuracy: {evaluate_classification_metrics(val_predicted_scores_tfidf, val_scores)[0]:.4f}")
    print(f"F1-Score: {evaluate_classification_metrics(val_predicted_scores_tfidf, val_scores)[1]:.4f}")

    # 在测试集上评估模型 Evaluate models on the test set
    test_predicted_scores_bow = compute_similarity(test_pairs, sentence_vectors_bow)
    test_predicted_scores_tfidf = compute_similarity(test_pairs, sentence_vectors_tfidf)
    save_combined_results(test_df, test_predicted_scores_bow, test_predicted_scores_tfidf, test_scores, '../output_dataset/baseline_stsb_test.csv')
    print("=== Test Performance (BoW) ===")
    print(f"Pearson Correlation: {evaluate_model(test_predicted_scores_bow, test_scores)[0]:.4f}")
    print(f"Spearman Rank Correlation: {evaluate_model(test_predicted_scores_bow, test_scores)[1]:.4f}")
    print(f"Accuracy: {evaluate_classification_metrics(test_predicted_scores_bow, test_scores)[0]:.4f}")
    print(f"F1-Score: {evaluate_classification_metrics(test_predicted_scores_bow, test_scores)[1]:.4f}")
    print("=== Test Performance (TF-IDF) ===")
    print(f"Pearson Correlation: {evaluate_model(test_predicted_scores_tfidf, test_scores)[0]:.4f}")
    print(f"Spearman Rank Correlation: {evaluate_model(test_predicted_scores_tfidf, test_scores)[1]:.4f}")
    print(f"Accuracy: {evaluate_classification_metrics(test_predicted_scores_tfidf, test_scores)[0]:.4f}")
    print(f"F1-Score: {evaluate_classification_metrics(test_predicted_scores_tfidf, test_scores)[1]:.4f}")
    # 调整生成的csv, modify csv to make output dataset easier to understand
    modify_and_save_csv('../output_dataset/baseline_stsb_validation.csv', '../output_dataset/baseline_stsb_validation.csv')
    modify_and_save_csv('../output_dataset/baseline_stsb_test.csv', '../output_dataset/baseline_stsb_test.csv')
    
    # 绘制开发集和测试集的直方图 Plot histograms for validation and test sets (counts as y axis, predict to true score as x axis)
    plot_histogram(val_predicted_scores_bow, val_scores, 'BoW', '../figures/bow_validation_histogram.png', 'red', 'Validation')
    plot_histogram(val_predicted_scores_tfidf, val_scores, 'TF-IDF', '../figures/tfidf_validation_histogram.png', 'brown', 'Validation')
    plot_histogram(test_predicted_scores_bow, test_scores, 'BoW', '../figures/bow_test_histogram.png', 'blue', 'Test')
    plot_histogram(test_predicted_scores_tfidf, test_scores, 'TF-IDF', '../figures/tfidf_test_histogram.png', 'green', 'Test')

    # 绘制开发集和测试集的折线图 Plot line charts for validation and test sets (frequency=counts/all counts as y axis, predict to true score as x axis)
    plot_line_chart(val_predicted_scores_bow, val_scores,'BoW', '../figures/bow_validation_line_chart.png','red', 'Validation')
    plot_line_chart(val_predicted_scores_tfidf, val_scores, 'TF-IDF', '../figures/tfidf_validation_line_chart.png','brown', 'Validation')
    plot_line_chart(test_predicted_scores_bow, test_scores, 'BoW', '../figures/bow_test_line_chart.png','blue', 'Test')
    plot_line_chart(test_predicted_scores_tfidf, test_scores, 'TF-IDF', '../figures/tfidf_test_line_chart.png','green', 'Test')