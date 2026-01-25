"""
模拟TrueSkill算法在不同比较次数下的rank error。

模拟设置：
- N=200个样本，latent score在[1,100]均匀分布
- 比较分类器：分对概率与latent差呈logistic分布
- latent差=10时分对概率73%，latent差=0时概率50%
- 研究K=4,8,16,32,64,128次比较下的rank error
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trueskill.calculate_trueskill_scores import calculate_trueskill_scores


def compute_logistic_k():
    """
    计算logistic函数的参数k。
    P(correct | diff) = 1 / (1 + exp(-k * diff))
    当diff=10时，P=0.73
    """
    # 0.73 = 1 / (1 + exp(-10k))
    # 1 + exp(-10k) = 1/0.73
    # exp(-10k) = 1/0.73 - 1
    # -10k = ln(1/0.73 - 1)
    # k = -ln(1/0.73 - 1) / 10
    k = -np.log(1/0.73 - 1) / 10
    return k


def compare_items(latent_i: float, latent_j: float, k: float, rng: np.random.Generator) -> str:
    """
    比较两个item，返回winner的id。
    
    Args:
        latent_i: item i的latent score
        latent_j: item j的latent score  
        k: logistic函数参数
        rng: 随机数生成器
        
    Returns:
        "i" 如果i胜出，"j" 如果j胜出
    """
    diff = latent_i - latent_j
    # P(i > j) = sigmoid(k * diff)
    prob_i_wins = 1 / (1 + np.exp(-k * diff))
    
    if rng.random() < prob_i_wins:
        return "i"
    else:
        return "j"


def generate_comparison_pairs(n_items: int, k_comparisons: int, rng: np.random.Generator) -> list:
    """
    生成比较对。
    
    通过shuffle id K次，使用shuffle_id1[i]对shuffle_id2[i]进行比较，然后去重。
    
    Args:
        n_items: item数量
        k_comparisons: 每个item平均参与的比较次数
        rng: 随机数生成器
        
    Returns:
        比较对列表 [(i, j), ...]，其中i < j
    """
    pairs = set()
    ids = list(range(n_items))
    
    # 为了让每个item平均参与k_comparisons次比较，我们需要生成足够多的pair
    # 每个pair贡献2次参与（两个item各参与一次）
    # 所以总参与次数 = 2 * num_pairs = n_items * k_comparisons
    # num_pairs = n_items * k_comparisons / 2
    target_pairs = n_items * k_comparisons // 2
    
    # 通过多次shuffle来生成pairs
    while len(pairs) < target_pairs:
        shuffle1 = ids.copy()
        shuffle2 = ids.copy()
        rng.shuffle(shuffle1)
        rng.shuffle(shuffle2)
        
        for i, j in zip(shuffle1, shuffle2):
            if i != j:
                # 使用有序pair去重
                pair = (min(i, j), max(i, j))
                pairs.add(pair)
                if len(pairs) >= target_pairs:
                    break
    
    return list(pairs)


def run_simulation(
    n_items: int = 200,
    k_values: list = None,
    n_trials: int = 10,
    seed: int = 42
) -> dict:
    """
    运行模拟实验。
    
    Args:
        n_items: item数量
        k_values: 每个item平均参与的比较次数列表
        n_trials: 每个k值重复实验次数
        seed: 随机种子
        
    Returns:
        {k: mean_normalized_rank_error}
    """
    if k_values is None:
        k_values = [4, 8, 16, 32, 64]
    
    rng = np.random.default_rng(seed)
    logistic_k = compute_logistic_k()
    
    print(f"Logistic参数 k = {logistic_k:.6f}")
    print(f"验证: 当diff=10时, P(correct) = {1/(1+np.exp(-logistic_k*10)):.4f}")
    print(f"N = {n_items}, 重复实验 {n_trials} 次")
    print("-" * 50)
    
    results = {}
    
    for k_comp in tqdm(k_values, desc="K values", position=0):
        trial_errors = []
        
        for trial in tqdm(range(n_trials), desc=f"K={k_comp} trials", position=1, leave=False):
            # 生成latent scores
            latent_scores = rng.uniform(1, 100, n_items)
            item_ids = [str(i) for i in range(n_items)]
            
            # 计算ground truth rank (从高到低)
            gt_rank_order = np.argsort(-latent_scores)  # 降序排列的索引
            gt_rank = np.zeros(n_items, dtype=int)
            for rank, idx in enumerate(gt_rank_order):
                gt_rank[idx] = rank
            
            # 生成比较对
            pairs = generate_comparison_pairs(n_items, k_comp, rng)
            
            # 模拟比较并生成边表
            edges = []
            for i, j in pairs:
                winner = compare_items(latent_scores[i], latent_scores[j], logistic_k, rng)
                if winner == "i":
                    edges.append((item_ids[i], item_ids[j]))
                else:
                    edges.append((item_ids[j], item_ids[i]))
            
            # 计算TrueSkill分数
            ts_scores = calculate_trueskill_scores(edges, n_shuffles=5, seed=trial)
            
            # 计算TrueSkill rank
            ts_rank_list = sorted(ts_scores.items(), key=lambda x: x[1], reverse=True)
            ts_rank = {item_id: rank for rank, (item_id, _) in enumerate(ts_rank_list)}
            
            # 计算rank error
            rank_errors = []
            for i in range(n_items):
                item_id = str(i)
                if item_id in ts_rank:
                    error = abs(gt_rank[i] - ts_rank[item_id])
                    rank_errors.append(error)
            
            mean_error = np.mean(rank_errors)
            trial_errors.append(mean_error)
        
        mean_normalized_error = np.mean(trial_errors) / n_items
        std_normalized_error = np.std(trial_errors) / n_items
        
        results[k_comp] = {
            'mean': mean_normalized_error,
            'std': std_normalized_error
        }
        
        tqdm.write(f"  K={k_comp}: rank_error/N = {mean_normalized_error:.4f} ± {std_normalized_error:.4f}")
    
    return results


def plot_results(results: dict, output_path: Path):
    """
    绘制K vs rank_error/N的图。
    
    Args:
        results: 模拟结果
        output_path: 输出路径
    """
    k_values = sorted(results.keys())
    means = [results[k]['mean'] for k in k_values]
    stds = [results[k]['std'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制误差条
    plt.errorbar(k_values, means, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, color='#2E86AB', 
                 ecolor='#A23B72', capthick=2)
    
    plt.xlabel('K (Average Comparisons per Item)', fontsize=12)
    plt.ylabel('Normalized Rank Error (rank_error / N)', fontsize=12)
    plt.title('TrueSkill Rank Error vs Number of Comparisons\n(N=200 items, Logistic comparison model)', 
              fontsize=14)
    
    # 设置x轴为对数刻度
    plt.xscale('log', base=2)
    plt.xticks(k_values, [str(k) for k in k_values])
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    # 运行模拟
    results = run_simulation(
        n_items=200,
        k_values=[4, 8, 16, 32, 64],
        n_trials=10,
        seed=42
    )
    
    # 绘图并保存
    output_dir = project_root / "local_data" / "visualization" / "trueskill_simu"
    output_path = output_dir / "k_vs_rank_error.jpg"
    
    plot_results(results, output_path)
    
    print("\n模拟完成!")


if __name__ == "__main__":
    main()

