import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#在这个python code也做到（注意这个和前面发你那个不是同一个文件了 因为前面那个训练模型要很长时间 所以只是绘制我就单独放着了
#就是（1）sbert方面计算fequency=count/total count（前面做的去0处理这里还是需要的） (2)绘制折线图 
# (3) #希望把输入模型最高的3个频率跟频率对应也打印在terminal   （4）# 图上添加一个方框标注最高频率点+对应bin
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_frequency_line_chart(csv_file, score_column, model_name, dataset_name, output_image, color):
    """
    绘制频率折线图，并标记最高频率及对应的 bin 范围。
    :param csv_file: 输入的 CSV 文件路径
    :param score_column: 要分析的分数列名
    :param model_name: 模型名称
    :param dataset_name: 数据集名称（Validation 或 Test）
    :param output_image: 保存折线图的文件路径
    :param color: 折线图颜色
    """
    # 加载数据
    df = pd.read_csv(csv_file)
    
    # 提取分数列
    scores = df[score_column].values
    true_scores = df['True score'].values  # 假设 "True score" 是列名

    # 过滤掉 True score 或 Predicted score 为 0 的情况
    valid_indices = (scores != 0) & (true_scores != 0)
    filtered_scores = scores[valid_indices]
    filtered_true_scores = true_scores[valid_indices]

    # 检查过滤后的数据是否为空
    if len(filtered_scores) == 0:
        print(f"No valid data for {model_name} in {dataset_name}. Skipping line chart.")
        return

    # 计算 Predicted / True Ratios
    predicted_true_ratio = filtered_scores / filtered_true_scores

    # 定义直方图区间（bins）
    bins = np.linspace(0, 5, 50)
    freq, edges = np.histogram(predicted_true_ratio, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2  # 计算每个区间的中心点

    # 计算频率为 ratio count / total count
    normalized_freq = freq / freq.sum()

    # 找到最高频率及对应的 bin 范围
    max_freq_index = np.argmax(normalized_freq)
    max_freq_value = normalized_freq[max_freq_index]
    max_bin_range = (edges[max_freq_index], edges[max_freq_index + 1])

    # 控制台打印最高的 3 个频率及对应的 bin 范围
    top_3_indices = np.argsort(normalized_freq)[-3:][::-1]
    print(f"=== {model_name} {dataset_name} -3 Mosy Frequent Predicted-to-True Ratio Ranges===")
    for i, idx in enumerate(top_3_indices):
        print(f"{i+1} Max Freq Bin Range: [{edges[idx]:.2f}, {edges[idx+1]:.2f}] - Frequency: {normalized_freq[idx]:.4f}")
    print("===================================")

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, normalized_freq, color=color, label=f'{model_name} Ratio Frequency', marker='o')
    plt.title(f'{model_name} Predicted/True Ratios Frequency ({dataset_name})', fontsize=24)
    plt.xlabel(f'{model_name} Predicted-to-True Score Ratio', fontsize=24)
    plt.ylabel('Frequency (Count / All Count)', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)

    # 图上添加一个方框标注最高频率点
    plt.text(
        bin_centers[max_freq_index], 
        max_freq_value + 0.05, 
        f'Max Freq Bin Range: [{max_bin_range[0]:.2f}, {max_bin_range[1]:.2f}]\nMax Freq: {max_freq_value:.4f}', 
        fontsize=16, 
        ha='center', 
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
    )

    plt.legend()
    plt.tight_layout()

    # 保存图表
    try:
        plt.savefig(output_image)
        #print(f"Line chart saved to {output_image}")
    except Exception as e:
        print(f"Failed to save line chart: {e}")
    
    plt.show()

#能不能除了这个chart也把这些指标做成网格图的方式显示
def plot_comparison_chart(data, metrics, models, output_file, title):
    """
    绘制对比图
    :param data: 包含数据的字典，键是模型名，值是对应的指标列表
    :param metrics: 指标名列表，例如 ['F1-Score', 'Accuracy', 'Spearman', 'Pearson']
    :param models: 模型名列表，例如 ['BoW', 'TF-IDF', 'SBERT']
    :param output_file: 保存图像的文件名
    :param title: 图表标题
    """
    x = np.arange(len(metrics))  # x轴位置
    width = 0.2  # 柱子宽度

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        ax.bar(x + i * width, data[model], width, label=model)

    # 添加标签和标题
    ax.set_xlabel('Metrics', fontsize=24, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics, fontsize=20)
    #ax.legend(fontsize=20)
    ax.legend(fontsize=20, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # 保存图表并展示
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def plot_combined_metrics_grid(validation_data, test_data, metrics, models, output_file, title):
    """
    绘制合并网格图，左侧为 Validation 数据，右侧为 Test 数据
    :param validation_data: Validation 数据字典
    :param test_data: Test 数据字典
    :param metrics: 指标名列表，例如 ['F1-Score', 'Accuracy', 'Spearman', 'Pearson']
    :param models: 模型名列表，例如 ['BoW', 'TF-IDF', 'SBERT']
    :param output_file: 保存图像的文件名
    :param title: 图表标题
    """
    # 创建子图
    #fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

    # 创建 Validation 数据的网格
    validation_grid = [
        [validation_data[model][metrics.index(metric)] for model in models]
        for metric in metrics
    ]
    validation_df = pd.DataFrame(validation_grid, index=metrics, columns=models)

    # 绘制 Validation 数据网格
    cax1 = axes[0].matshow(validation_df, cmap='coolwarm')
    axes[0].set_title('Validation Metrics', fontsize=16, pad=15)
    axes[0].set_xticks(np.arange(len(models)))
    axes[0].set_xticklabels(models, fontsize=12)
    axes[0].set_yticks(np.arange(len(metrics)))
    axes[0].set_yticklabels(metrics, fontsize=12)

    # 添加 Validation 数据网格中的数值
    for (i, j), val in np.ndenumerate(validation_df.values):
        axes[0].text(j, i, f"{val:.4f}", ha='center', va='center', color='black', fontsize=10)

    # 创建 Test 数据的网格
    test_grid = [
        [test_data[model][metrics.index(metric)] for model in models]
        for metric in metrics
    ]
    test_df = pd.DataFrame(test_grid, index=metrics, columns=models)

    # 绘制 Test 数据网格
    cax2 = axes[1].matshow(test_df, cmap='coolwarm')
    axes[1].set_title('Test Metrics', fontsize=16, pad=15)
    axes[1].set_xticks(np.arange(len(models)))
    axes[1].set_xticklabels(models, fontsize=12)
    axes[1].set_yticks(np.arange(len(metrics)))
    axes[1].set_yticklabels([])  # 不显示右侧网格的 Y 轴标签

    # 添加 Test 数据网格中的数值
    for (i, j), val in np.ndenumerate(test_df.values):
        axes[1].text(j, i, f"{val:.4f}", ha='center', va='center', color='black', fontsize=10)

    # 添加颜色条
    fig.colorbar(cax1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(cax2, ax=axes[1], fraction=0.046, pad=0.04)

    # 设置全局标题
    plt.suptitle(title, fontsize=20, fontweight='bold', y=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.9]) 
    plt.subplots_adjust(top=0.85)  # 调整子图与全局标题间距

    # 保存图表
    plt.savefig(output_file)
    plt.show()
if __name__ == "__main__":
    # 读取 CSV 文件
    validation_results = pd.read_csv('../output_dataset/baseline_stsb_validation.csv')
    test_results = pd.read_csv('../output_dataset/baseline_stsb_test.csv')
    sbert_validation_results = pd.read_csv('../output_dataset/sbert_stsb_validation.csv')
    sbert_test_results = pd.read_csv('../output_dataset/sbert_stsb_test.csv')

    # 提取指标
    metrics = ['F1-Score', 'Accuracy', 'Spearman', 'Pearson']
    models = ['BoW', 'TF-IDF', 'SBERT']

    # 从 Validation 结果中提取指标
    validation_data = {
        'BoW': [
            validation_results.loc[0, 'BoW F1'],
            validation_results.loc[0, 'BoW Accuracy'],
            validation_results.loc[0, 'BoW Spearman'],
            validation_results.loc[0, 'BoW Pearson']
        ],
        'TF-IDF': [
            validation_results.loc[0, 'TF-IDF F1'],
            validation_results.loc[0, 'TF-IDF Accuracy'],
            validation_results.loc[0, 'TF-IDF Spearman'],
            validation_results.loc[0, 'TF-IDF Pearson']
        ],
        'SBERT': [
            sbert_validation_results.loc[0, 'SBERT F1'],
            sbert_validation_results.loc[0, 'SBERT Accuracy'],
            sbert_validation_results.loc[0, 'SBERT Spearman'],
            sbert_validation_results.loc[0, 'SBERT Pearson']
        ]
    }

    # 从 Test 结果中提取指标
    test_data = {
        'BoW': [
            test_results.loc[0, 'BoW F1'],
            test_results.loc[0, 'BoW Accuracy'],
            test_results.loc[0, 'BoW Spearman'],
            test_results.loc[0, 'BoW Pearson']
        ],
        'TF-IDF': [
            test_results.loc[0, 'TF-IDF F1'],
            test_results.loc[0, 'TF-IDF Accuracy'],
            test_results.loc[0, 'TF-IDF Spearman'],
            test_results.loc[0, 'TF-IDF Pearson']
        ],
        'SBERT': [
            sbert_test_results.loc[0, 'SBERT F1'],
            sbert_test_results.loc[0, 'SBERT Accuracy'],
            sbert_test_results.loc[0, 'SBERT Spearman'],
            sbert_test_results.loc[0, 'SBERT Pearson']
        ]
    }

    # 绘制 Validation 集对比图
    plot_comparison_chart(
        validation_data,
        metrics,
        models,
        '../figures/validation_comparison_chart.png',
        'Validation Set Comparison'
    )

    # 绘制 Test 集对比图
    plot_comparison_chart(
        test_data,
        metrics,
        models,
        '../figures/test_comparison_chart.png',
        'Test Set Comparison'
    )
     # 示例调用：绘制 Validation 集的 SBERT 频率折线图
    plot_frequency_line_chart(
        csv_file='../output_dataset/sbert_stsb_validation.csv',
        score_column='SBERT score',
        model_name='SBERT',
        dataset_name='Validation',
        output_image='../figures/sbert_validation_line_chart.png',
        color='orange'
    )

    # 示例调用：绘制 Test 集的 SBERT 频率折线图
    plot_frequency_line_chart(
        csv_file='../output_dataset/sbert_stsb_test.csv',
        score_column='SBERT score',
        model_name='SBERT',
        dataset_name='Test',
        output_image='../figures/sbert_test_line_chart.png',
        color='purple'
    )

    #两个网格图能不能也做个合并 左边validation那个 右边test那个
    plot_combined_metrics_grid(
        validation_data=validation_data,
        test_data=test_data,
        metrics=metrics,
        models=models,
        output_file='../figures/combined_metrics_grid.png',
        title=' '
    )