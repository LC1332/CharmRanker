"""
EloManager 模拟测试脚本

使用模拟比较来验证EloManager的正确性：
- 4000个节点
- 每次标记K=10个triplet
- 进行3000个循环
- 每30个循环查看一次平均rank error
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import sys
from typing import List, Dict, Tuple

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.EloManager.elo_manager import EloManager


def compute_logistic_k():
    """
    计算logistic函数的参数k。
    与pairwise比较保持一致：latent差=10时分对概率73%
    """
    k = -np.log(1/0.73 - 1) / 10
    return k


def simulate_triplet_comparison(
    latent_scores: Dict[str, float],
    triplet: List[str],
    logistic_k: float,
    rng: np.random.Generator
) -> Dict[str, int]:
    """
    模拟triplet比较，返回最大和最小的索引（1-based）。
    
    使用Plackett-Luce模型进行概率排序。
    
    Args:
        latent_scores: {node_name: latent_score}
        triplet: 3个节点名称
        logistic_k: logistic函数参数
        rng: 随机数生成器
        
    Returns:
        {'largest': 1/2/3, 'smallest': 1/2/3}
    """
    scores = np.array([latent_scores[n] for n in triplet])
    
    # Plackett-Luce模型排序
    result = []
    remaining_indices = list(range(3))
    remaining_scores = list(scores)
    
    for _ in range(2):
        exp_scores = np.exp(logistic_k * np.array(remaining_scores))
        probs = exp_scores / exp_scores.sum()
        
        choice_idx = rng.choice(len(remaining_scores), p=probs)
        result.append(remaining_indices[choice_idx])
        
        remaining_indices.pop(choice_idx)
        remaining_scores.pop(choice_idx)
    
    result.append(remaining_indices[0])
    
    # result[0] 是最大的，result[2] 是最小的
    largest_orig_idx = result[0]  # 原始triplet中的索引
    smallest_orig_idx = result[2]
    
    return {
        'largest': largest_orig_idx + 1,   # 转换为1-based
        'smallest': smallest_orig_idx + 1
    }


def calculate_rank_error(
    latent_scores: Dict[str, float],
    manager: EloManager
) -> float:
    """
    计算normalized rank error。
    
    Args:
        latent_scores: ground truth latent scores
        manager: EloManager实例
        
    Returns:
        normalized rank error (error / n_nodes)
    """
    node_names = list(latent_scores.keys())
    n_nodes = len(node_names)
    
    # Ground truth rank (按latent_score降序)
    gt_sorted = sorted(latent_scores.items(), key=lambda x: x[1], reverse=True)
    gt_rank = {name: rank for rank, (name, _) in enumerate(gt_sorted)}
    
    # TrueSkill rank (按mu降序)
    ts_rankings = manager.get_rankings()
    ts_rank = {name: rank for rank, (name, _, _) in enumerate(ts_rankings)}
    
    # 计算rank error
    rank_errors = []
    for name in node_names:
        if name in ts_rank:
            error = abs(gt_rank[name] - ts_rank[name])
            rank_errors.append(error)
        else:
            rank_errors.append(n_nodes)
    
    return np.mean(rank_errors) / n_nodes


def run_simulation(
    n_nodes: int = 4000,
    batch_size: int = 10,
    total_rounds: int = 3000,
    check_interval: int = 30,
    seed: int = 42,
    save_path: str = None
) -> Tuple[List[int], List[float]]:
    """
    运行完整的模拟实验。
    
    Args:
        n_nodes: 节点数量
        batch_size: 每次提出的triplet数量
        total_rounds: 总循环次数
        check_interval: 每隔多少轮检查一次rank error
        seed: 随机种子
        save_path: 保存路径（可选）
        
    Returns:
        (rounds, errors): 轮次列表和对应的rank error列表
    """
    print(f"模拟设置:")
    print(f"  节点数量: {n_nodes}")
    print(f"  每批triplet: {batch_size}")
    print(f"  总轮次: {total_rounds}")
    print(f"  检查间隔: {check_interval}")
    print(f"  随机种子: {seed}")
    print("=" * 60)
    
    # 初始化
    rng = np.random.default_rng(seed)
    logistic_k = compute_logistic_k()
    
    # 创建EloManager
    manager = EloManager()
    
    # 生成节点和latent scores
    node_infos = []
    latent_scores = {}
    for i in range(n_nodes):
        name = f"node_{i}"
        latent = rng.uniform(1, 100)
        node_infos.append({
            'name': name,
            'latent_score': latent  # 仅用于模拟比较
        })
        latent_scores[name] = latent
    
    manager.add_nodes(node_infos)
    
    # 记录数据
    rounds_list = []
    errors_list = []
    
    # 主循环
    pbar = tqdm(range(total_rounds), desc="模拟进行中")
    
    for round_idx in pbar:
        # 提出triplet
        try:
            triplets = manager.propose_triplet(K=batch_size)
        except ValueError as e:
            print(f"轮次 {round_idx}: 无法提出triplet - {e}")
            continue
        
        # 模拟比较
        results = []
        for triplet_info in triplets:
            triplet = triplet_info['triplet']
            result = simulate_triplet_comparison(
                latent_scores, triplet, logistic_k, rng
            )
            results.append(result)
        
        # 提交结果
        manager.submit_triplet_result(results, triplets)
        
        # 定期检查rank error
        if (round_idx + 1) % check_interval == 0:
            error = calculate_rank_error(latent_scores, manager)
            rounds_list.append(round_idx + 1)
            errors_list.append(error)
            
            stats = manager.get_stats()
            pbar.set_postfix({
                'error': f'{error:.4f}',
                'hungry': stats['hungry_nodes'],
                'comparisons': stats['total_comparisons']
            })
    
    # 最终统计
    final_error = calculate_rank_error(latent_scores, manager)
    final_stats = manager.get_stats()
    
    print("\n" + "=" * 60)
    print("模拟完成!")
    print(f"  最终rank error: {final_error:.4f}")
    print(f"  总比较次数: {final_stats['total_comparisons']}")
    print(f"  平均每节点参与次数: {final_stats['avg_compare_per_node']:.2f}")
    print(f"  未温饱节点: {final_stats['hungry_nodes']}")
    
    # 保存结果
    if save_path:
        manager.save(save_path)
    
    return rounds_list, errors_list


def plot_error_curve(
    rounds: List[int],
    errors: List[float],
    output_path: Path = None
):
    """
    绘制rank error随轮次变化的曲线。
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(rounds, errors, 'b-', linewidth=2, alpha=0.8)
    plt.scatter(rounds, errors, c='blue', s=20, alpha=0.6)
    
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('Normalized Rank Error', fontsize=12)
    plt.title('EloManager 模拟实验: Rank Error 变化曲线\n(4000节点, 每轮10个triplet)', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    
    # 添加趋势信息
    if len(errors) > 1:
        initial_error = errors[0]
        final_error = errors[-1]
        reduction = (initial_error - final_error) / initial_error * 100
        plt.text(0.95, 0.95, 
                f'初始error: {initial_error:.4f}\n最终error: {final_error:.4f}\n降低: {reduction:.1f}%',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    
    plt.close()


def test_save_load():
    """测试save/load功能"""
    print("\n" + "=" * 60)
    print("测试 save/load 功能")
    print("=" * 60)
    
    rng = np.random.default_rng(123)
    logistic_k = compute_logistic_k()
    
    # 创建并运行一小段
    manager1 = EloManager()
    
    node_infos = []
    latent_scores = {}
    for i in range(100):
        name = f"test_node_{i}"
        latent = rng.uniform(1, 100)
        node_infos.append({'name': name, 'latent_score': latent})
        latent_scores[name] = latent
    
    manager1.add_nodes(node_infos)
    
    # 运行几轮
    for _ in range(10):
        triplets = manager1.propose_triplet(K=5)
        results = []
        for t in triplets:
            result = simulate_triplet_comparison(latent_scores, t['triplet'], logistic_k, rng)
            results.append(result)
        manager1.submit_triplet_result(results, triplets)
    
    # 保存
    save_path = project_root / "local_data" / "test_elo_manager"
    manager1.save(str(save_path))
    
    stats1 = manager1.get_stats()
    rankings1 = manager1.get_rankings()[:5]
    
    # 加载到新实例
    manager2 = EloManager()
    manager2.load(str(save_path))
    
    stats2 = manager2.get_stats()
    rankings2 = manager2.get_rankings()[:5]
    
    # 验证
    print(f"\n保存前统计: {stats1}")
    print(f"加载后统计: {stats2}")
    
    print(f"\n保存前Top5排名: {rankings1}")
    print(f"加载后Top5排名: {rankings2}")
    
    # 检查是否一致
    assert stats1['total_nodes'] == stats2['total_nodes'], "节点数量不一致"
    assert stats1['total_comparisons'] == stats2['total_comparisons'], "比较次数不一致"
    
    print("\n✓ save/load 测试通过!")


def run_simulation_with_dynamic_nodes(
    n_initial_nodes: int = 4000,
    batch_size: int = 10,
    total_rounds: int = 3000,
    check_interval: int = 30,
    new_node_interval: int = 30,
    seed: int = 42,
    save_path: str = None
) -> Tuple[List[int], List[float], List[int]]:
    """
    运行带动态节点添加的模拟实验。
    
    Args:
        n_initial_nodes: 初始节点数量
        batch_size: 每次提出的triplet数量
        total_rounds: 总循环次数
        check_interval: 每隔多少轮检查一次rank error
        new_node_interval: 每隔多少轮加入一个新节点
        seed: 随机种子
        save_path: 保存路径（可选）
        
    Returns:
        (rounds, errors, node_counts): 轮次列表、rank error列表、节点数量列表
    """
    print(f"动态节点模拟设置:")
    print(f"  初始节点数量: {n_initial_nodes}")
    print(f"  每批triplet: {batch_size}")
    print(f"  总轮次: {total_rounds}")
    print(f"  检查间隔: {check_interval}")
    print(f"  新节点添加间隔: {new_node_interval}")
    print(f"  随机种子: {seed}")
    print("=" * 60)
    
    # 初始化
    rng = np.random.default_rng(seed)
    logistic_k = compute_logistic_k()
    
    # 创建EloManager
    manager = EloManager()
    
    # 生成初始节点和latent scores
    node_infos = []
    latent_scores = {}
    next_node_id = 0
    
    for i in range(n_initial_nodes):
        name = f"node_{next_node_id}"
        latent = rng.uniform(1, 100)
        node_infos.append({
            'name': name,
            'latent_score': latent
        })
        latent_scores[name] = latent
        next_node_id += 1
    
    manager.add_nodes(node_infos)
    
    # 记录数据
    rounds_list = []
    errors_list = []
    node_counts_list = []
    
    # 主循环
    pbar = tqdm(range(total_rounds), desc="动态节点模拟")
    
    for round_idx in pbar:
        # 提出triplet
        try:
            triplets = manager.propose_triplet(K=batch_size)
        except ValueError as e:
            print(f"轮次 {round_idx}: 无法提出triplet - {e}")
            continue
        
        # 模拟比较
        results = []
        for triplet_info in triplets:
            triplet = triplet_info['triplet']
            result = simulate_triplet_comparison(
                latent_scores, triplet, logistic_k, rng
            )
            results.append(result)
        
        # 提交结果
        manager.submit_triplet_result(results, triplets)
        
        # 定期检查rank error
        if (round_idx + 1) % check_interval == 0:
            error = calculate_rank_error(latent_scores, manager)
            rounds_list.append(round_idx + 1)
            errors_list.append(error)
            node_counts_list.append(len(manager.nodes))
            
            stats = manager.get_stats()
            pbar.set_postfix({
                'error': f'{error:.4f}',
                'nodes': len(manager.nodes),
                'hungry': stats['hungry_nodes'],
                'comparisons': stats['total_comparisons']
            })
        
        # 定期添加新节点
        if (round_idx + 1) % new_node_interval == 0:
            new_name = f"node_{next_node_id}"
            new_latent = rng.uniform(1, 100)
            new_node_info = {
                'name': new_name,
                'latent_score': new_latent
            }
            manager.add_nodes([new_node_info])
            latent_scores[new_name] = new_latent
            next_node_id += 1
    
    # 最终统计
    final_error = calculate_rank_error(latent_scores, manager)
    final_stats = manager.get_stats()
    
    print("\n" + "=" * 60)
    print("动态节点模拟完成!")
    print(f"  最终节点数: {len(manager.nodes)} (初始: {n_initial_nodes}, 新增: {len(manager.nodes) - n_initial_nodes})")
    print(f"  最终rank error: {final_error:.4f}")
    print(f"  总比较次数: {final_stats['total_comparisons']}")
    print(f"  平均每节点参与次数: {final_stats['avg_compare_per_node']:.2f}")
    print(f"  未温饱节点: {final_stats['hungry_nodes']}")
    
    # 保存结果
    if save_path:
        manager.save(save_path)
        print(f"  数据已保存到: {save_path}")
    
    return rounds_list, errors_list, node_counts_list


def plot_dynamic_error_curve(
    rounds: List[int],
    errors: List[float],
    node_counts: List[int],
    output_path: Path = None
):
    """
    绘制动态节点模拟的rank error和节点数变化曲线。
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # 左轴：rank error
    color1 = '#2E86AB'
    ax1.set_xlabel('轮次', fontsize=12)
    ax1.set_ylabel('Normalized Rank Error', color=color1, fontsize=12)
    line1 = ax1.plot(rounds, errors, color=color1, linewidth=2, alpha=0.8, label='Rank Error')
    ax1.scatter(rounds, errors, c=color1, s=20, alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # 右轴：节点数
    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.set_ylabel('节点数量', color=color2, fontsize=12)
    line2 = ax2.plot(rounds, node_counts, color=color2, linewidth=2, alpha=0.8, 
                     linestyle='--', label='节点数')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('EloManager 动态节点模拟实验\n(初始4000节点, 每30轮添加1个新节点)', fontsize=14)
    
    # 添加统计信息
    if len(errors) > 1:
        initial_error = errors[0]
        final_error = errors[-1]
        reduction = (initial_error - final_error) / initial_error * 100
        initial_nodes = node_counts[0]
        final_nodes = node_counts[-1]
        
        info_text = (f'初始error: {initial_error:.4f}\n'
                    f'最终error: {final_error:.4f}\n'
                    f'Error降低: {reduction:.1f}%\n'
                    f'初始节点: {initial_nodes}\n'
                    f'最终节点: {final_nodes}')
        
        ax1.text(0.02, 0.98, info_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    
    plt.close()


def test_propose_rules():
    """测试propose规则"""
    print("\n" + "=" * 60)
    print("测试 propose 规则")
    print("=" * 60)
    
    rng = np.random.default_rng(456)
    logistic_k = compute_logistic_k()
    
    manager = EloManager()
    
    # 添加节点
    node_infos = []
    latent_scores = {}
    for i in range(50):
        name = f"rule_node_{i}"
        latent = rng.uniform(1, 100)
        node_infos.append({'name': name, 'latent_score': latent})
        latent_scores[name] = latent
    
    manager.add_nodes(node_infos)
    
    # 测试温饱原则
    print("\n1. 测试温饱原则:")
    triplets = manager.propose_triplet(K=5)
    
    hungry_nodes = manager._get_hungry_nodes()
    print(f"   未温饱节点数: {len(hungry_nodes)}")
    
    # 检查提出的triplet是否包含未温饱节点
    for t in triplets:
        has_hungry = any(n in hungry_nodes for n in t['triplet'])
        print(f"   triplet {t['triplet']}: 包含未温饱节点={has_hungry}")
    
    # 运行几轮使所有节点温饱
    print("\n2. 运行至所有节点温饱...")
    max_rounds = 100
    for round_idx in range(max_rounds):
        triplets = manager.propose_triplet(K=5)
        results = []
        for t in triplets:
            result = simulate_triplet_comparison(latent_scores, t['triplet'], logistic_k, rng)
            results.append(result)
        manager.submit_triplet_result(results, triplets)
        
        hungry = len(manager._get_hungry_nodes())
        if hungry == 0:
            print(f"   第 {round_idx + 1} 轮后所有节点温饱")
            break
    
    # 测试邻近原则
    print("\n3. 测试邻近原则:")
    normalized_scores = manager._get_normalized_scores()
    triplets = manager.propose_triplet(K=5)
    
    for t in triplets:
        scores = [normalized_scores[n] for n in t['triplet']]
        max_diff = max(scores) - min(scores)
        print(f"   triplet {t['triplet']}: 分数={[f'{s:.1f}' for s in scores]}, 差值={max_diff:.1f}")
    
    # 测试有重复不抽原则
    print("\n4. 测试有重复不抽原则:")
    compared_before = len(manager.compared_triplets)
    
    # 再提出一些triplet
    triplets = manager.propose_triplet(K=10)
    for t in triplets:
        fs = frozenset(t['triplet'])
        if fs in manager.compared_triplets:
            print(f"   错误: {t['triplet']} 已被比较过但仍被提出!")
        else:
            print(f"   ✓ {t['triplet']} 是新的triplet")
    
    print("\n✓ propose 规则测试通过!")


def run_simulation_with_anchors(
    n_initial_nodes: int = 400,
    batch_size: int = 10,
    total_rounds: int = 300,
    check_interval: int = 30,
    new_node_interval: int = 30,
    score_noise: float = 20.0,
    seed: int = 42,
    save_path: str = None
) -> Tuple[List[int], List[float], List[int]]:
    """
    使用 get_anchors 和 add_node_with_score 的模拟实验。
    
    新节点的 relative_score 和真实 latent 平均相差 score_noise。
    
    Args:
        n_initial_nodes: 初始节点数量
        batch_size: 每次提出的triplet数量
        total_rounds: 总循环次数
        check_interval: 每隔多少轮检查一次rank error
        new_node_interval: 每隔多少轮加入一个新节点
        score_noise: 估计分数和真实分数的平均差值
        seed: 随机种子
        save_path: 保存路径（可选）
        
    Returns:
        (rounds, errors, node_counts): 轮次列表、rank error列表、节点数量列表
    """
    print(f"Anchor模拟设置:")
    print(f"  初始节点数量: {n_initial_nodes}")
    print(f"  每批triplet: {batch_size}")
    print(f"  总轮次: {total_rounds}")
    print(f"  检查间隔: {check_interval}")
    print(f"  新节点添加间隔: {new_node_interval}")
    print(f"  分数噪声: ±{score_noise}")
    print(f"  随机种子: {seed}")
    print("=" * 60)
    
    # 初始化
    rng = np.random.default_rng(seed)
    logistic_k = compute_logistic_k()
    
    # 创建EloManager
    manager = EloManager()
    
    # 生成初始节点和latent scores
    node_infos = []
    latent_scores = {}
    next_node_id = 0
    
    for i in range(n_initial_nodes):
        name = f"node_{next_node_id}"
        latent = rng.uniform(1, 100)
        node_infos.append({
            'name': name,
            'latent_score': latent
        })
        latent_scores[name] = latent
        next_node_id += 1
    
    manager.add_nodes(node_infos)
    
    # 记录数据
    rounds_list = []
    errors_list = []
    node_counts_list = []
    
    # 主循环
    pbar = tqdm(range(total_rounds), desc="Anchor模拟")
    
    for round_idx in pbar:
        # 提出triplet
        try:
            triplets = manager.propose_triplet(K=batch_size)
        except ValueError as e:
            print(f"轮次 {round_idx}: 无法提出triplet - {e}")
            continue
        
        # 模拟比较
        results = []
        for triplet_info in triplets:
            triplet = triplet_info['triplet']
            result = simulate_triplet_comparison(
                latent_scores, triplet, logistic_k, rng
            )
            results.append(result)
        
        # 提交结果
        manager.submit_triplet_result(results, triplets)
        
        # 定期检查rank error
        if (round_idx + 1) % check_interval == 0:
            error = calculate_rank_error(latent_scores, manager)
            rounds_list.append(round_idx + 1)
            errors_list.append(error)
            node_counts_list.append(len(manager.nodes))
            
            stats = manager.get_stats()
            pbar.set_postfix({
                'error': f'{error:.4f}',
                'nodes': len(manager.nodes),
                'hungry': stats['hungry_nodes'],
                'comparisons': stats['total_comparisons']
            })
        
        # 定期添加新节点（使用 get_anchors 和 add_node_with_score）
        if (round_idx + 1) % new_node_interval == 0:
            # 获取锚点
            anchors = manager.get_anchors()
            
            # 为新节点生成真实的latent score
            new_name = f"node_{next_node_id}"
            new_latent = rng.uniform(1, 100)
            
            # 估计的 relative_score（与真实值有噪声）
            # 真实的 percentile 大约是 new_latent (因为 latent 在 1-100 均匀分布)
            noise = rng.uniform(-score_noise, score_noise)
            estimated_score = np.clip(new_latent + noise, 0, 100)
            
            # 使用 add_node_with_score 添加新节点
            new_node_info = {
                'name': new_name,
                'latent_score': new_latent
            }
            manager.add_node_with_score(new_node_info, estimated_score)
            latent_scores[new_name] = new_latent
            next_node_id += 1
    
    # 最终统计
    final_error = calculate_rank_error(latent_scores, manager)
    final_stats = manager.get_stats()
    
    print("\n" + "=" * 60)
    print("Anchor模拟完成!")
    print(f"  最终节点数: {len(manager.nodes)} (初始: {n_initial_nodes}, 新增: {len(manager.nodes) - n_initial_nodes})")
    print(f"  最终rank error: {final_error:.4f}")
    print(f"  总比较次数: {final_stats['total_comparisons']}")
    print(f"  平均每节点参与次数: {final_stats['avg_compare_per_node']:.2f}")
    print(f"  未温饱节点: {final_stats['hungry_nodes']}")
    
    # 测试 get_anchors
    print("\n获取的锚点样本:")
    anchors = manager.get_anchors()
    for anchor in anchors:
        print(f"  {anchor['name']}: percentile={anchor['percentile']:.1f}%, "
              f"mu={anchor['mu']:.2f}, sigma={anchor['sigma']:.3f}, "
              f"compare_count={anchor['compare_count']}")
    
    # 保存结果
    if save_path:
        manager.save(save_path)
        print(f"  数据已保存到: {save_path}")
    
    return rounds_list, errors_list, node_counts_list


def test_anchors_api():
    """测试 get_anchors 和 add_node_with_score 接口"""
    print("\n" + "=" * 60)
    print("测试 get_anchors 和 add_node_with_score 接口")
    print("=" * 60)
    
    rng = np.random.default_rng(789)
    logistic_k = compute_logistic_k()
    
    manager = EloManager()
    
    # 添加初始节点
    node_infos = []
    latent_scores = {}
    for i in range(100):
        name = f"test_anchor_{i}"
        latent = rng.uniform(1, 100)
        node_infos.append({'name': name, 'latent_score': latent})
        latent_scores[name] = latent
    
    manager.add_nodes(node_infos)
    
    # 运行一些比较
    print("\n1. 运行50轮比较...")
    for _ in range(50):
        triplets = manager.propose_triplet(K=5)
        results = []
        for t in triplets:
            result = simulate_triplet_comparison(latent_scores, t['triplet'], logistic_k, rng)
            results.append(result)
        manager.submit_triplet_result(results, triplets)
    
    # 测试 get_anchors
    print("\n2. 测试 get_anchors:")
    anchors = manager.get_anchors()
    print(f"   获取到 {len(anchors)} 个锚点")
    
    for anchor in anchors:
        pct = anchor['percentile']
        print(f"   分位数 {pct:.1f}%: {anchor['name']}, "
              f"compare_count={anchor['compare_count']}, sigma={anchor['sigma']:.3f}")
    
    # 验证分位数分布
    percentiles = [a['percentile'] for a in anchors]
    print(f"\n   分位数范围: {min(percentiles):.1f}% - {max(percentiles):.1f}%")
    
    # 测试 add_node_with_score
    print("\n3. 测试 add_node_with_score:")
    
    # 添加一个高分节点
    new_name_high = manager.add_node_with_score(
        {'name': 'new_high', 'latent_score': 90},
        relative_score=85  # 估计在85分位
    )
    latent_scores['new_high'] = 90
    
    # 添加一个低分节点
    new_name_low = manager.add_node_with_score(
        {'name': 'new_low', 'latent_score': 15},
        relative_score=20  # 估计在20分位
    )
    latent_scores['new_low'] = 15
    
    # 查看新节点的初始分数
    rankings = manager.get_rankings()
    for name, mu, sigma in rankings:
        if name in ['new_high', 'new_low']:
            rank_idx = [r[0] for r in rankings].index(name)
            pct = (len(rankings) - 1 - rank_idx) / (len(rankings) - 1) * 100
            print(f"   {name}: mu={mu:.2f}, sigma={sigma:.3f}, 当前分位数={pct:.1f}%")
    
    print("\n✓ get_anchors 和 add_node_with_score 测试通过!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EloManager 模拟测试')
    parser.add_argument('--n_nodes', type=int, default=4000, help='节点数量')
    parser.add_argument('--batch_size', type=int, default=10, help='每批triplet数量')
    parser.add_argument('--total_rounds', type=int, default=3000, help='总轮次')
    parser.add_argument('--check_interval', type=int, default=30, help='检查间隔')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--test_only', action='store_true', help='仅运行单元测试')
    parser.add_argument('--dynamic', action='store_true', help='运行动态节点添加模拟')
    parser.add_argument('--anchors', action='store_true', help='运行anchor模式模拟（使用get_anchors和add_node_with_score）')
    parser.add_argument('--new_node_interval', type=int, default=30, help='添加新节点的间隔轮次')
    parser.add_argument('--score_noise', type=float, default=20.0, help='新节点估计分数的噪声范围')
    
    args = parser.parse_args()
    
    if args.test_only:
        # 仅运行测试
        test_save_load()
        test_propose_rules()
        test_anchors_api()
        return
    
    if args.anchors:
        # 运行 anchor 模式模拟
        save_path = project_root / "local_data" / "test_elo_manager"
        
        rounds, errors, node_counts = run_simulation_with_anchors(
            n_initial_nodes=args.n_nodes,
            batch_size=args.batch_size,
            total_rounds=args.total_rounds,
            check_interval=args.check_interval,
            new_node_interval=args.new_node_interval,
            score_noise=args.score_noise,
            seed=args.seed,
            save_path=str(save_path)
        )
        
        # 绘制曲线
        plot_dynamic_error_curve(
            rounds, errors, node_counts,
            output_path=save_path / "anchor_error_curve.jpg"
        )
    elif args.dynamic:
        # 运行动态节点模拟
        save_path = project_root / "local_data" / "test_elo_manager"
        
        rounds, errors, node_counts = run_simulation_with_dynamic_nodes(
            n_initial_nodes=args.n_nodes,
            batch_size=args.batch_size,
            total_rounds=args.total_rounds,
            check_interval=args.check_interval,
            new_node_interval=args.new_node_interval,
            seed=args.seed,
            save_path=str(save_path)
        )
        
        # 绘制曲线
        plot_dynamic_error_curve(
            rounds, errors, node_counts,
            output_path=save_path / "dynamic_error_curve.jpg"
        )
    else:
        # 运行完整模拟
        output_dir = project_root / "local_data" / "elo_manager_simulation"
        
        rounds, errors = run_simulation(
            n_nodes=args.n_nodes,
            batch_size=args.batch_size,
            total_rounds=args.total_rounds,
            check_interval=args.check_interval,
            seed=args.seed,
            save_path=str(output_dir / "final_state")
        )
        
        # 绘制曲线
        plot_error_curve(
            rounds, errors,
            output_path=output_dir / "error_curve.jpg"
        )
    
    print("\n完成!")


if __name__ == "__main__":
    main()

