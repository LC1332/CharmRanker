"""
Step 4: Annotate Triplets with LLM

For each triplet, ask LLM to select which image(s) have the largest/smallest attribute value.
The LLM can select:
- Only the largest
- Only the smallest  
- Both largest and smallest
This produces 1-3 edges per triplet.

Usage:
    # Single attribute
    python step4_annotate_triplets.py --attribute "haze level" --api zhipu
    
    # All attributes
    python step4_annotate_triplets.py --attribute all --api zhipu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    load_attribute_config, TRIPLETS_DIR, ANNOTATIONS_DIR,
    get_all_attributes, APIConfig, get_domain_from_attribute
)
from utils import (
    encode_image_to_base64, call_api, save_jsonl, load_jsonl,
    ProgressTracker
)


def build_annotation_message(
    triplet_dir: Path,
    image1: str,
    image2: str,
    image3: str,
    attribute_name: str,
    larger_word: str,
    smaller_word: str,
    comparative_word: str
) -> Dict:
    """
    构建三元组标注消息
    
    LLM需要识别:
    - Which image has the LARGEST attribute value?
    - Which image has the SMALLEST attribute value?
    """
    
    img1_path = triplet_dir / image1
    img2_path = triplet_dir / image2
    img3_path = triplet_dir / image3
    
    prompt = f"""You are an expert image analyst. Please analyze these THREE images and identify which image(s) have the LARGEST and SMALLEST {attribute_name}.

**Important**: Images with more {attribute_name} are considered {larger_word}, while images with less {attribute_name} are considered {smaller_word}.

Please examine all three images carefully:
- Image A (first image)
- Image B (second image)
- Image C (third image)

**IMPORTANT: You MUST respond with ONLY a valid JSON object, no other text.**

Output Format (copy this structure exactly):
{{
  "analysis": "Your detailed comparison of all three images",
  "largest": "A" or "B" or "C" or null (the image with LARGEST {attribute_name}, or null if uncertain),
  "smallest": "A" or "B" or "C" or null (the image with SMALLEST {attribute_name}, or null if uncertain)
}}

You may set either "largest" or "smallest" to null if you cannot confidently identify it.
If you can clearly identify both, provide both values.

Now analyze the three images and respond with ONLY the JSON object:
"""
    
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(img1_path)}},
        {"type": "text", "text": "Image A"},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(img2_path)}},
        {"type": "text", "text": "Image B"},
        {"type": "image_url", "image_url": {"url": encode_image_to_base64(img3_path)}},
        {"type": "text", "text": "Image C"}
    ]
    
    conversation = [{"role": "user", "content": content}]
    
    # Generate unique ID
    import uuid
    msg_id = str(uuid.uuid4())
    
    return {
        "conversation": conversation,
        "id": msg_id,
        "metadata": {
            "image1": image1,
            "image2": image2,
            "image3": image3,
            "triplet_dir": str(triplet_dir),
            "attribute_name": attribute_name
        }
    }


def parse_annotation_response(response_text: str) -> Dict[str, str]:
    """
    解析标注响应,提取largest和smallest
    
    Returns:
        {"largest": "A/B/C" or None, "smallest": "A/B/C" or None}
    """
    try:
        # 清理响应文本
        cleaned = response_text.strip()
        
        # 移除可能的markdown标记
        if '```json' in cleaned:
            cleaned = cleaned.split('```json')[1].split('```')[0]
        elif '```' in cleaned:
            cleaned = cleaned.split('```')[1].split('```')[0]
        
        # 移除GLM特殊标记
        if '<|begin_of_box|>' in cleaned:
            cleaned = cleaned.split('<|begin_of_box|>')[1]
        if '<|end_of_box|>' in cleaned:
            cleaned = cleaned.split('<|end_of_box|>')[0]
        
        # 解析JSON
        data = json.loads(cleaned.strip())
        
        result = {}
        # 处理largest字段,允许null
        if 'largest' in data:
            largest_val = data['largest']
            if largest_val and str(largest_val).lower() != 'null':
                result['largest'] = str(largest_val).strip().upper()
            else:
                result['largest'] = None
        
        # 处理smallest字段,允许null  
        if 'smallest' in data:
            smallest_val = data['smallest']
            if smallest_val and str(smallest_val).lower() != 'null':
                result['smallest'] = str(smallest_val).strip().upper()
            else:
                result['smallest'] = None
        
        return result
        
    except Exception as e:
        print(f"  ⚠️  Failed to parse response: {e}")
        print(f"      Response: {response_text[:200]}")
        return {}


def response_to_edges(
    response_result: Dict[str, str],
    image1: str,
    image2: str,
    image3: str
) -> List[Dict]:
    """
    将标注结果转换为edges
    
    返回格式参考 src/my_annotations_lcj.jsonl:
    每个edge包含: img1, img2, compare_bool (True表示img1 > img2)
    
    逻辑：
    - 如果有largest（不是null），生成 largest > 其他两张图片的所有组合
    - 如果有smallest（不是null），生成 smallest < 其他两张图片的所有组合
    """
    images = {'A': image1, 'B': image2, 'C': image3}
    edges = []
    
    largest = response_result.get('largest')
    smallest = response_result.get('smallest')
    
    # 如果有largest，生成 largest > 其他图片的所有组合
    if largest is not None and largest in images:
        largest_img = images[largest]
        other_images = [img for key, img in images.items() if key != largest]
        
        for other_img in other_images:
            edges.append({
                "img1": largest_img,
                "img2": other_img,
                "compare_bool": True  # largest > other
            })
    
    # 如果有smallest，生成 smallest < 其他图片的所有组合
    if smallest is not None and smallest in images:
        smallest_img = images[smallest]
        other_images = [img for key, img in images.items() if key != smallest]
        
        for other_img in other_images:
            edges.append({
                "img1": smallest_img,
                "img2": other_img,
                "compare_bool": False  # smallest < other (即 other > smallest)
            })
    
    return edges


def annotate_triplets_for_attribute(attribute: str, api_type: str = 'zhipu') -> bool:
    """
    为单个attribute的所有三元组进行标注
    
    Returns:
        True if successful
    """
    print(f"\n{'#'*70}")
    print(f"# Annotating Triplets for: {attribute}")
    print(f"# API: {api_type}")
    print(f"{'#'*70}")
    
    try:
        # 获取domain
        domain = get_domain_from_attribute(attribute)
        
        # 加载属性配置
        attr_config = load_attribute_config()
        if attribute not in attr_config:
            print(f"✗ Attribute '{attribute}' not found in configuration")
            return False
        
        config = attr_config[attribute]
        larger_word = config['larger_word']
        smaller_word = config['smaller_word']
        comparative_word = config['comparative_word']
        
        # 读取三元组 - 使用新的路径结构
        triplets_file = TRIPLETS_DIR / domain / attribute / api_type / "triplets.jsonl"
        if not triplets_file.exists():
            print(f"✗ Triplets file not found: {triplets_file}")
            return False
        
        triplets = load_jsonl(triplets_file)
        print(f"Found {len(triplets)} triplets")
        
        # 创建输出目录 - API分类子目录
        output_dir = ANNOTATIONS_DIR / domain / attribute / api_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成标注消息
        print("\nGenerating annotation messages...")
        messages = []
        
        for triplet in triplets:
            triplet_id = triplet['id']
            level_num = triplet['level_num']
            triplet_dir = TRIPLETS_DIR / domain / attribute / api_type / f"level_{level_num}" / triplet_id
            
            if not triplet_dir.exists():
                print(f"  ⚠️  Triplet directory not found: {triplet_dir}")
                continue
            
            msg = build_annotation_message(
                triplet_dir,
                triplet['image1'],
                triplet['image2'],
                triplet['image3'],
                attribute,
                larger_word,
                smaller_word,
                comparative_word
            )
            messages.append(msg)
        
        messages_file = output_dir / "annotation_messages.jsonl"
        save_jsonl(messages, messages_file)
        print(f"Saved {len(messages)} annotation messages")
        
        # 调用API
        print(f"\nCalling {api_type} API for annotations...")
        responses = []
        tracker = ProgressTracker(len(messages), "API calls")
        
        for msg in messages:
            response = call_api(
                msg['conversation'],  # messages参数
                api_type=api_type,
                temperature=0.1
            )
            
            responses.append({
                "id": msg['id'],
                "metadata": msg['metadata'],
                "response": response.get('content', str(response))
            })
            tracker.update()
        
        responses_file = output_dir / "annotation_responses.jsonl"
        save_jsonl(responses, responses_file)
        print(f"\nSaved {len(responses)} responses")
        
        # 解析响应并生成annotations
        print("\nParsing annotations...")
        annotations = []
        
        for resp in responses:
            metadata = resp['metadata']
            response_result = parse_annotation_response(resp['response'])
            
            if response_result:
                # 转换为edges
                edges = response_to_edges(
                    response_result,
                    metadata['image1'],
                    metadata['image2'],
                    metadata['image3']
                )
                
                # 添加完整的元数据
                for edge in edges:
                    edge.update({
                        "attribute_name": attribute,
                        "triplet_id": Path(metadata['triplet_dir']).name,
                        "img1": str(Path(metadata['triplet_dir']) / edge['img1']),
                        "img2": str(Path(metadata['triplet_dir']) / edge['img2'])
                    })
                    annotations.append(edge)
        
        # 保存annotations
        annotations_file = output_dir / "annotations.jsonl"
        save_jsonl(annotations, annotations_file)
        
        print(f"\n{'='*70}")
        print(f"✓ Annotation completed for: {attribute}")
        print(f"  Total triplets: {len(messages)}")
        print(f"  Total edges: {len(annotations)}")
        print(f"  Annotations file: {annotations_file}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Annotation failed for: {attribute}")
        print(f"  Error: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Step 4: Annotate Triplets with LLM'
    )
    parser.add_argument('--attribute', type=str, required=True,
                       help='Attribute name or "all"')
    parser.add_argument('--api', type=str, default=APIConfig.DEFAULT_API,
                       choices=APIConfig.SUPPORTED_APIS,
                       help=f'API type (default: {APIConfig.DEFAULT_API})')
    
    args = parser.parse_args()
    
    # 确定要处理的attributes
    if args.attribute.lower() == 'all':
        attributes = get_all_attributes()
        print(f"Processing {len(attributes)} attributes")
    else:
        attributes = [args.attribute]
    
    # 处理每个attribute
    success_count = 0
    for attr in attributes:
        if annotate_triplets_for_attribute(attr, args.api):
            success_count += 1
    
    # 输出总结
    print(f"\n{'#'*70}")
    print(f"# ANNOTATION SUMMARY")
    print(f"{'#'*70}")
    print(f"Total: {len(attributes)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {len(attributes) - success_count}")
    print(f"{'#'*70}\n")
    
    sys.exit(0 if success_count == len(attributes) else 1)


if __name__ == "__main__":
    main()
