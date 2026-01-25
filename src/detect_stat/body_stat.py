"""
统计 body 框的边长（按短边算），分 30 个 bin，
分别统计 Album_A、univer-light 和合并数据，
输出一张合并的 jpg 图到 visualization 目录。
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 路径配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOCAL_DATA = PROJECT_ROOT / "local_data"
VISUALIZATION_DIR = LOCAL_DATA / "visualization"

ALBUM_A_JSONL = LOCAL_DATA / "Album_A_detect_result.jsonl"
UNIVER_LIGHT_JSONL = LOCAL_DATA / "univer-light_detect_result.jsonl"

NUM_BINS = 30


def load_body_short_edges(jsonl_path: Path) -> list[float]:
    """
    从 jsonl 文件中加载所有 body 框的短边长度（像素）
    """
    short_edges = []
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 跳过没有 body_bbox 的记录
            if "body_bbox" not in data:
                continue
            
            body_bbox = data["body_bbox"]
            image_width = data["image_width"]
            image_height = data["image_height"]
            
            # 计算实际像素尺寸
            box_width = body_bbox["width"] * image_width
            box_height = body_bbox["height"] * image_height
            
            # 取短边
            short_edge = min(box_width, box_height)
            short_edges.append(short_edge)
    
    return short_edges


def plot_histograms(album_a_edges: list[float], 
                   univer_light_edges: list[float],
                   output_path: Path):
    """
    绘制三个直方图：Album_A、univer-light 和合并数据
    """
    combined_edges = album_a_edges + univer_light_edges
    
    # 确定统一的 bin 范围
    all_edges = np.array(combined_edges)
    min_edge = np.floor(all_edges.min())
    max_edge = np.ceil(all_edges.max())
    bins = np.linspace(min_edge, max_edge, NUM_BINS + 1)
    
    # 创建 1x3 子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Body Box Short Edge Distribution (30 bins)", fontsize=14, fontweight="bold")
    
    # 颜色配置
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    
    datasets = [
        (album_a_edges, "Album_A", colors[0]),
        (univer_light_edges, "univer-light", colors[1]),
        (combined_edges, "Combined", colors[2]),
    ]
    
    for ax, (edges, name, color) in zip(axes, datasets):
        if len(edges) == 0:
            ax.set_title(f"{name}\n(No data)")
            continue
        
        edges_arr = np.array(edges)
        
        # 绘制直方图
        counts, _, patches = ax.hist(edges_arr, bins=bins, color=color, 
                                      edgecolor="white", alpha=0.8)
        
        # 添加统计信息
        stats_text = (
            f"Count: {len(edges):,}\n"
            f"Mean: {edges_arr.mean():.1f} px\n"
            f"Median: {np.median(edges_arr):.1f} px\n"
            f"Min: {edges_arr.min():.1f} px\n"
            f"Max: {edges_arr.max():.1f} px\n"
            f"Std: {edges_arr.std():.1f} px"
        )
        
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment="top", horizontalalignment="right",
                fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                         edgecolor="gray", alpha=0.9))
        
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Short Edge Length (pixels)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        
        # 设置 x 轴刻度
        ax.set_xlim(min_edge, max_edge)
    
    plt.tight_layout()
    
    # 保存图片
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", 
                facecolor="white", edgecolor="none")
    plt.close()
    
    print(f"统计图已保存至: {output_path}")


def main():
    print("加载 Album_A 数据...")
    album_a_edges = load_body_short_edges(ALBUM_A_JSONL)
    print(f"  找到 {len(album_a_edges):,} 个 body 框")
    
    print("加载 univer-light 数据...")
    univer_light_edges = load_body_short_edges(UNIVER_LIGHT_JSONL)
    print(f"  找到 {len(univer_light_edges):,} 个 body 框")
    
    print(f"\n开始绘制直方图 ({NUM_BINS} bins)...")
    output_path = VISUALIZATION_DIR / "body_short_edge_histogram.jpg"
    plot_histograms(album_a_edges, univer_light_edges, output_path)
    
    print("\n完成!")


if __name__ == "__main__":
    main()

