import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
EPOCHS = 6
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
SIMILARITY_THRESHOLD = 0.7

# 加载数据集
def load_datasets():
    train_df = pd.read_csv('../source_dataset/stsb_train.csv')
    validation_df = pd.read_csv('../source_dataset/stsb_validation.csv')
    test_df = pd.read_csv('../source_dataset/stsb_test.csv')
    return train_df, validation_df, test_df

# 提取句子对和相似度分数
def get_sentence_pairs(df):
    return df[['sentence1', 'sentence2']].values.tolist(), df['score'].tolist()

# 创建训练样本和数据加载器
def create_dataloader(train_pairs, train_scores):
    train_examples = [InputExample(texts=[sent1, sent2], label=score) for (sent1, sent2), score in zip(train_pairs, train_scores)]
    return DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# 训练模型
def train_model(model, train_dataloader):
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        optimizer_params={'lr': LEARNING_RATE}
    )

# 计算句子对的嵌入和相似度
def compute_sbert_similarity(model, pairs):
    predicted_scores = []
    for sent1, sent2 in pairs:
        embedding1 = model.encode(sent1)
        embedding2 = model.encode(sent2)
        cos_sim = cosine_similarity([embedding1], [embedding2])[0][0]
        predicted_scores.append(cos_sim)
    return predicted_scores

# 评估模型性能
def evaluate_model(predicted_scores, true_scores):
    pearson_corr, _ = pearsonr(predicted_scores, true_scores)
    spearman_corr, _ = spearmanr(predicted_scores, true_scores)
    return pearson_corr, spearman_corr

# 计算分类指标
def evaluate_classification_metrics(predicted_scores, true_scores):
    binary_labels = [1 if score >= SIMILARITY_THRESHOLD else 0 for score in true_scores]
    predicted_binary_labels = [1 if score >= SIMILARITY_THRESHOLD else 0 for score in predicted_scores]
    accuracy = accuracy_score(binary_labels, predicted_binary_labels)
    f1 = f1_score(binary_labels, predicted_binary_labels)
    return accuracy, f1

# 保存结果到文件
def save_results(df, predicted_scores, true_scores, output_csv, output_image, dataset_name, color):
    processed_df = df.copy()
    processed_df['SBERT score'] = predicted_scores
    processed_df['SBERT Pearson Correlation'] = None
    processed_df['SBERT Spearman Rank Correlation'] = None
    processed_df['Accuracy'] = None
    processed_df['F1-Score'] = None
    
    # 评估指标
    pearson, spearman = evaluate_model(predicted_scores, true_scores)
    accuracy, f1 = evaluate_classification_metrics(predicted_scores, true_scores)
    
    # 保存评估指标
    processed_df.at[0, 'SBERT Pearson Correlation'] = pearson
    processed_df.at[0, 'SBERT Spearman Rank Correlation'] = spearman
    processed_df.at[0, 'Accuracy'] = accuracy
    processed_df.at[0, 'F1-Score'] = f1
    
    # 控制台输出带说明的指标
    print(f"=== {dataset_name} Performance (SBERT) ===")
    print(f"Pearson Correlation: {pearson:.4f}")
    print(f"Spearman Rank Correlation: {spearman:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("==========================")
    
    # 计算每行的 SBERT/True 比例
    processed_df['SBERT/True'] = processed_df['SBERT score'] / processed_df['score']
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    processed_df = processed_df.infer_objects(copy=False)  # 添加明确的类型推断以避免 FutureWarning
    processed_df.fillna(0, inplace=True)
    processed_df.to_csv(output_csv, index=False)
    # **仅绘制直方图时过滤掉 True score 或 Predicted score 为 0 的情况**
    filtered_scores = processed_df[(processed_df['score'] != 0) & (processed_df['SBERT score'] != 0)]
    
    
    # 绘制直方图
    bins = np.linspace(0, 5, 50)
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_scores['SBERT/True'], bins=bins, color=color, alpha=0.7, label=f'SBERT/True Ratio ({dataset_name})')
    #plt.hist(processed_df['SBERT/True'], bins=bins, color=color, alpha=0.7, label=f'SBERT/True Ratio ({dataset_name})')
    plt.title(f'Distribution of SBERT/True Ratios - {dataset_name}', fontsize=24)
    plt.xlabel('SBERT Predicted-to-True Score Ratio', fontsize=24)
    plt.ylabel('Counts(Times)', fontsize=24)
    # 设置刻度字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image)
    plt.show()
def modify_and_save_csv(input_csv, output_csv):
    """
    修改指定 CSV 文件，更新列名和顺序，并保存为新的文件
    """
    # 加载 CSV 文件
    df = pd.read_csv(input_csv)

    # 修改列名
    df.rename(columns={
        'score': 'True score',
        'SBERT Pearson Correlation': 'SBERT Pearson',
        'SBERT Spearman Rank Correlation': 'SBERT Spearman',
        'F1-Score': 'SBERT F1',
        'Accuracy': 'SBERT Accuracy'
    }, inplace=True)

    # 调整列的顺序
    new_column_order = [
        'sentence1', 'sentence2', 'True score',
        'SBERT score', 'SBERT/True', 'SBERT F1', 'SBERT Accuracy', 'SBERT Spearman', 'SBERT Pearson'
    ]
    df = df[new_column_order]

    # 保存为新的 CSV 文件
    df.to_csv(output_csv, index=False)
    #print(f"Modified CSV saved to: {output_csv}")
# 主程序
if __name__ == "__main__":
    # 加载数据集
    train_df, validation_df, test_df = load_datasets()

    # 提取句子对和相似度分数
    train_pairs, train_scores = get_sentence_pairs(train_df)
    val_pairs, val_scores = get_sentence_pairs(validation_df)
    test_pairs, test_scores = get_sentence_pairs(test_df)

    # 创建 Sentence-BERT 模型
    model = SentenceTransformer(MODEL_NAME)

    # 训练模型
    train_dataloader = create_dataloader(train_pairs, train_scores)
    train_model(model, train_dataloader)

    # 在开发集上评估模型
    val_predicted_scores_sbert = compute_sbert_similarity(model, val_pairs)
    save_results(validation_df, val_predicted_scores_sbert, val_scores, 
                 '../output_dataset/sbert_stsb_validation.csv', '../figures/sbert_validation_histogram.png', 'Validation', 'yellow')

    # 在测试集上评估模型
    test_predicted_scores_sbert = compute_sbert_similarity(model, test_pairs)
    save_results(test_df, test_predicted_scores_sbert, test_scores, 
                 '../output_dataset/sbert_stsb_test.csv', '../figures/sbert_test_histogram.png', 'Test', 'purple')
    
    modify_and_save_csv('../output_dataset/sbert_stsb_validation.csv', '../output_dataset/sbert_stsb_validation.csv')
    modify_and_save_csv('../output_dataset/sbert_stsb_test.csv', '../output_dataset/sbert_stsb_test.csv')