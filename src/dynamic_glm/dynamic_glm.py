"""
Dynamic GLM - 使用 GLM-4.6v 进行动态图片颜值比较

功能：
- 并行调用 GLM API 进行 triplet 颜值比较
- 使用 EloManager 进行分数管理和 triplet 提案
- 支持断点续传
- 记录所有响应日志
"""

import os
import sys
import json
import time
import base64
import random
import requests
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, List, Optional, Tuple
from queue import Queue
from dotenv import load_dotenv
import yaml

# 添加项目根目录到 path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.EloManager.elo_manager import EloManager
from src.triplet2message.triplet2message import ATTRACTIVENESS_PROMPT

# 加载环境变量
load_dotenv(_PROJECT_ROOT / ".env")

GLM_API_KEY = os.getenv("GLM_API_KEY")
GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class DynamicGLM:
    """动态 GLM 颜值比较器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化
        
        Args:
            config_path: 配置文件路径，默认使用同目录下的 config.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config = self._load_config(config_path)
        
        # 数据路径
        self.input_dir = _PROJECT_ROOT / self.config['data']['input_dir']
        self.save_path = _PROJECT_ROOT / self.config['data']['save_path']
        self.sample_n = self.config['data'].get('sample_n')
        
        # 并行配置
        self.max_workers = self.config['parallel']['max_workers']
        self.k_one_time = self.config['parallel']['k_one_time']
        self.wait_seconds = self.config['parallel']['wait_seconds']
        self.timeout_warning_seconds = self.config['parallel']['timeout_warning_seconds']
        
        # API 配置
        self.model = self.config['api']['model']
        self.temperature = self.config['api']['temperature']
        self.api_timeout = self.config['api']['timeout']
        self.budget = self.config['api']['budget']
        
        # 确保保存目录存在
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化 EloManager（使用专用配置）
        elo_config_path = self._create_elo_config()
        self.elo_manager = EloManager(config_path=elo_config_path)
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 统计
        self.api_call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
        # 响应日志文件路径
        self.response_log_path = self.save_path / "response_logs.jsonl"
        
        # 任务队列（用于跟踪进行中的任务）
        self.pending_tasks: Dict[str, Future] = {}  # task_id -> Future
        self.task_info: Dict[str, Dict] = {}  # task_id -> triplet info
        
        print(f"DynamicGLM 初始化完成")
        print(f"  输入目录: {self.input_dir}")
        print(f"  保存路径: {self.save_path}")
        print(f"  最大并行: {self.max_workers}")
        print(f"  每批数量: {self.k_one_time}")
        print(f"  API 预算: {self.budget}")
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_elo_config(self) -> Path:
        """为 EloManager 创建临时配置文件"""
        elo_config = {
            'trueskill': self.config['trueskill'],
            'propose': self.config['propose'],
            'simulation': self.config['simulation']
        }
        
        elo_config_path = self.save_path / "elo_config.yaml"
        with open(elo_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(elo_config, f, default_flow_style=False)
        
        return elo_config_path
    
    def _encode_image(self, image_path: Path) -> str:
        """编码图片为 base64 data URL"""
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode()
        return f"data:image/jpeg;base64,{base64_data}"
    
    def _build_glm_message(self, image_a: Path, image_b: Path, image_c: Path) -> dict:
        """构建 GLM 消息格式（OpenAI 兼容）"""
        content = [
            {"type": "text", "text": ATTRACTIVENESS_PROMPT},
            {"type": "text", "text": "Image A:"},
            {"type": "image_url", "image_url": {"url": self._encode_image(image_a)}},
            {"type": "text", "text": "Image B:"},
            {"type": "image_url", "image_url": {"url": self._encode_image(image_b)}},
            {"type": "text", "text": "Image C:"},
            {"type": "image_url", "image_url": {"url": self._encode_image(image_c)}},
        ]
        
        return {"messages": [{"role": "user", "content": content}]}
    
    def _call_glm_api(self, message: dict) -> str:
        """调用 GLM API"""
        if not GLM_API_KEY:
            raise ValueError("请在 .env 文件中设置 GLM_API_KEY")
        
        url = f"{GLM_BASE_URL}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {GLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            **message,
            "model": self.model,
            "temperature": self.temperature,
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=self.api_timeout)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"].get("content")
        
        if content is None:
            refusal = result["choices"][0]["message"].get("refusal", "Unknown reason")
            raise ValueError(f"GLM 拒绝了请求: {refusal}")
        
        return content
    
    def _parse_response(self, response_text: str) -> Optional[Dict]:
        """解析 GLM 响应"""
        try:
            cleaned = response_text.strip()
            
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1].split('```')[0]
            
            data = json.loads(cleaned.strip())
            
            # 标准化结果
            most_attractive = str(data.get('most_attractive', '')).strip().upper()
            least_attractive = str(data.get('least_attractive', '')).strip().upper()
            
            if most_attractive not in ['A', 'B', 'C']:
                most_attractive = None
            if least_attractive not in ['A', 'B', 'C']:
                least_attractive = None
            
            return {
                'analysis': data.get('analysis', ''),
                'most_attractive': most_attractive,
                'least_attractive': least_attractive
            }
        except Exception as e:
            print(f"    解析响应失败: {e}")
            return None
    
    def _convert_to_result(self, parsed: Dict, triplet: List[str]) -> Optional[Dict]:
        """将解析结果转换为 EloManager 格式"""
        if parsed is None:
            return None
        
        most = parsed.get('most_attractive')
        least = parsed.get('least_attractive')
        
        if most is None or least is None:
            return None
        
        if most == least:
            return None
        
        # 转换 A/B/C 到 1/2/3
        mapping = {'A': 1, 'B': 2, 'C': 3}
        
        return {
            'largest': mapping[most],
            'smallest': mapping[least]
        }
    
    def _log_response(self, triplet_info: Dict, raw_response: str, parsed: Optional[Dict], 
                      result: Optional[Dict], success: bool, error: Optional[str] = None):
        """记录响应日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'triplet': triplet_info['triplet'],
            'image_paths': [n.get('image_path', n.get('name', '')) for n in triplet_info['nodes']],
            'success': success,
            'raw_response': raw_response if not success or parsed is None else None,
            'parsed_response': parsed,
            'result': result,
            'error': error
        }
        
        with self.lock:
            with open(self.response_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def _worker_task(self, triplet_info: Dict) -> Tuple[Dict, Optional[Dict], str]:
        """Worker 任务：调用 GLM API 并返回结果"""
        triplet = triplet_info['triplet']
        nodes = triplet_info['nodes']
        
        # 获取图片路径
        image_paths = [Path(n['image_path']) for n in nodes]
        
        try:
            # 构建消息
            message = self._build_glm_message(*image_paths)
            
            # 调用 API
            raw_response = self._call_glm_api(message)
            
            # 解析响应
            parsed = self._parse_response(raw_response)
            
            # 转换为结果
            result = self._convert_to_result(parsed, triplet)
            
            # 记录日志
            self._log_response(triplet_info, raw_response, parsed, result, success=True)
            
            return triplet_info, result, raw_response
            
        except Exception as e:
            error_msg = str(e)
            self._log_response(triplet_info, "", None, None, success=False, error=error_msg)
            return triplet_info, None, f"Error: {error_msg}"
    
    def _load_images(self) -> List[Dict]:
        """加载图片并创建节点信息"""
        # 查找所有 jpg 图片
        image_files = list(self.input_dir.glob("*.jpg"))
        
        if not image_files:
            raise ValueError(f"输入目录中没有找到 jpg 图片: {self.input_dir}")
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 采样
        if self.sample_n and self.sample_n < len(image_files):
            random.seed(self.config['simulation']['seed'])
            image_files = random.sample(image_files, self.sample_n)
            print(f"采样 {self.sample_n} 张图片")
        
        # 创建节点信息
        nodes = []
        for img_path in image_files:
            nodes.append({
                'name': img_path.stem,  # 文件名（不含扩展名）作为 name
                'image_path': str(img_path)
            })
        
        return nodes
    
    def _try_resume(self) -> bool:
        """尝试从断点恢复"""
        nodes_path = self.save_path / "nodes.json"
        
        if nodes_path.exists():
            try:
                self.elo_manager.load(str(self.save_path))
                
                # 统计已完成的 API 调用数（从响应日志）
                if self.response_log_path.exists():
                    with open(self.response_log_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                log = json.loads(line.strip())
                                self.api_call_count += 1
                                if log.get('success'):
                                    self.successful_calls += 1
                                else:
                                    self.failed_calls += 1
                            except:
                                pass
                
                print(f"从断点恢复: 已完成 {self.api_call_count} 次 API 调用")
                print(f"  成功: {self.successful_calls}, 失败: {self.failed_calls}")
                return True
            except Exception as e:
                print(f"恢复失败: {e}，将重新开始")
                return False
        
        return False
    
    def _get_available_workers(self) -> int:
        """获取可用的 worker 数量"""
        # 清理已完成的任务
        completed = []
        for task_id, future in self.pending_tasks.items():
            if future.done():
                completed.append(task_id)
        
        for task_id in completed:
            del self.pending_tasks[task_id]
            if task_id in self.task_info:
                del self.task_info[task_id]
        
        return self.max_workers - len(self.pending_tasks)
    
    def run(self):
        """运行动态颜值比较"""
        print("\n" + "=" * 60)
        print("开始动态 GLM 颜值比较")
        print("=" * 60)
        
        # 尝试断点恢复
        resumed = self._try_resume()
        
        if not resumed:
            # 加载图片
            nodes = self._load_images()
            
            # 添加节点到 EloManager
            self.elo_manager.add_nodes(nodes)
            
            # 保存初始状态
            self.elo_manager.save(str(self.save_path))
        
        # 检查预算
        if self.api_call_count >= self.budget:
            print(f"已达到 API 预算 ({self.budget})，退出")
            return
        
        print(f"\n开始比较，剩余预算: {self.budget - self.api_call_count}")
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            is_first_batch = True
            last_completion_time = time.time()
            
            while self.api_call_count < self.budget:
                # 处理已完成任务的结果（每次循环都检查）
                completed_results = []
                completed_triplets = []
                
                for task_id, future in list(self.pending_tasks.items()):
                    if future.done():
                        try:
                            triplet_info, result, raw_response = future.result()
                            
                            with self.lock:
                                self.api_call_count += 1
                                if result:
                                    self.successful_calls += 1
                                    completed_results.append(result)
                                    completed_triplets.append(triplet_info)
                                else:
                                    self.failed_calls += 1
                            
                            last_completion_time = time.time()
                            
                        except Exception as e:
                            with self.lock:
                                self.api_call_count += 1
                                self.failed_calls += 1
                            print(f"任务异常: {e}")
                        
                        del self.pending_tasks[task_id]
                        if task_id in self.task_info:
                            del self.task_info[task_id]
                
                # 提交有效结果到 EloManager 并立即保存
                if completed_results and completed_triplets:
                    self.elo_manager.submit_triplet_result(completed_results, completed_triplets)
                    
                    # 每次有结果返回都保存
                    self.elo_manager.save(str(self.save_path))
                    stats = self.elo_manager.get_stats()
                    print(f"  [{self.api_call_count}/{self.budget}] "
                          f"成功: {self.successful_calls}, 失败: {self.failed_calls}, "
                          f"比较数: {stats['total_comparisons']}")
                
                # 检查是否还有预算
                remaining_budget = self.budget - self.api_call_count - len(self.pending_tasks)
                if remaining_budget <= 0:
                    # 等待所有任务完成后退出
                    if len(self.pending_tasks) == 0:
                        break
                    time.sleep(0.5)
                    continue
                
                # 获取可用 worker 数量
                available = self._get_available_workers()
                
                # 检查是否需要等待
                if not is_first_batch and available < self.k_one_time:
                    # 等待并检查超时
                    elapsed_since_completion = time.time() - last_completion_time
                    
                    if elapsed_since_completion > self.timeout_warning_seconds:
                        print(f"\n⚠️ 警告: 已等待 {int(elapsed_since_completion)} 秒没有 worker 完成")
                        last_completion_time = time.time()  # 重置，3分钟后再报警
                    
                    time.sleep(min(self.wait_seconds, 1))  # 每秒检查一次
                    continue
                
                # 计算需要提出多少个 triplet
                if available >= self.k_one_time:
                    k = min(self.k_one_time, remaining_budget, available)
                    
                    if k > 0:
                        # 发出任务前先保存当前状态
                        self.elo_manager.save(str(self.save_path))
                        
                        # 提出新的 triplet
                        try:
                            triplets = self.elo_manager.propose_triplet(K=k)
                            
                            # 提交任务
                            for triplet_info in triplets:
                                task_id = f"{time.time()}_{triplet_info['triplet'][0]}"
                                future = executor.submit(self._worker_task, triplet_info)
                                self.pending_tasks[task_id] = future
                                self.task_info[task_id] = triplet_info
                            
                            if is_first_batch:
                                print(f"  首批发出 {len(triplets)} 个任务")
                                is_first_batch = False
                            
                        except Exception as e:
                            print(f"propose_triplet 失败: {e}")
                            time.sleep(1)
                
                # 短暂等待
                if not is_first_batch:
                    time.sleep(0.1)
            
            # 等待所有剩余任务完成
            if self.pending_tasks:
                print(f"\n等待 {len(self.pending_tasks)} 个剩余任务完成...")
                
            for task_id, future in list(self.pending_tasks.items()):
                try:
                    triplet_info, result, raw_response = future.result(timeout=self.api_timeout)
                    
                    with self.lock:
                        self.api_call_count += 1
                        if result:
                            self.successful_calls += 1
                            self.elo_manager.submit_triplet_result([result], [triplet_info])
                            # 每次有结果返回都保存
                            self.elo_manager.save(str(self.save_path))
                        else:
                            self.failed_calls += 1
                            
                except Exception as e:
                    with self.lock:
                        self.api_call_count += 1
                        self.failed_calls += 1
                    print(f"最终任务异常: {e}")
        
        # 保存最终状态
        self.elo_manager.save(str(self.save_path))
        
        # 输出统计
        print("\n" + "=" * 60)
        print("比较完成!")
        print("=" * 60)
        print(f"总 API 调用: {self.api_call_count}")
        print(f"成功: {self.successful_calls}")
        print(f"失败: {self.failed_calls}")
        
        stats = self.elo_manager.get_stats()
        print(f"\nEloManager 统计:")
        print(f"  节点数: {stats['total_nodes']}")
        print(f"  比较数: {stats['total_comparisons']}")
        print(f"  平均参与次数: {stats['avg_compare_per_node']:.2f}")
        
        # 输出排名前10
        rankings = self.elo_manager.get_rankings()
        print(f"\n排名前 10:")
        for i, (name, mu, sigma) in enumerate(rankings[:10], 1):
            print(f"  {i}. {name}: mu={mu:.2f}, sigma={sigma:.2f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic GLM 颜值比较")
    parser.add_argument("--config", "-c", default=None,
                        help="配置文件路径 (默认: src/dynamic_glm/config.yaml)")
    args = parser.parse_args()
    
    # 检查 API Key
    if not GLM_API_KEY:
        print("错误: 请在 .env 文件中设置 GLM_API_KEY")
        sys.exit(1)
    
    # 运行
    dynamic_glm = DynamicGLM(config_path=args.config)
    dynamic_glm.run()


if __name__ == "__main__":
    main()
