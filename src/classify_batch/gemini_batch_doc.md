Batch API



Gemini Batch API 旨在以标准费用 50% 的价格异步处理大量请求。目标处理时间为 24 小时，但在大多数情况下，处理速度会快得多。

对于大规模非紧急任务（例如数据预处理或运行评估），如果不需要立即响应，请使用 Batch API。

创建批处理作业
您可以通过以下两种方式在 Batch API 中提交请求：

内嵌请求：直接包含在批量创建请求中的 GenerateContentRequest 对象列表。此方法适用于总请求大小不超过 20 MB 的较小批次。从模型返回的输出是一个 inlineResponse 对象列表。
输入文件：一个 JSON Lines (JSONL) 文件，其中每行都包含一个完整的 GenerateContentRequest 对象。 建议针对较大请求使用此方法。模型返回的输出是一个 JSONL 文件，其中每行都是一个 GenerateContentResponse 或状态对象。
内嵌请求
对于少量请求，您可以直接将 GenerateContentRequest 对象嵌入到 BatchGenerateContentRequest 中。以下示例使用内嵌请求调用 BatchGenerateContent 方法：

Python
JavaScript
REST


from google import genai
from google.genai import types

client = genai.Client()

# A list of dictionaries, where each is a GenerateContentRequest
inline_requests = [
    {
        'contents': [{
            'parts': [{'text': 'Tell me a one-sentence joke.'}],
            'role': 'user'
        }]
    },
    {
        'contents': [{
            'parts': [{'text': 'Why is the sky blue?'}],
            'role': 'user'
        }]
    }
]

inline_batch_job = client.batches.create(
    model="models/gemini-3-flash-preview",
    src=inline_requests,
    config={
        'display_name': "inlined-requests-job-1",
    },
)

print(f"Created batch job: {inline_batch_job.name}")
输入文件
对于较多的请求，请准备一个 JSON 行 (JSONL) 文件。此文件中的每一行都必须是一个 JSON 对象，其中包含用户定义的键和请求对象，并且请求是有效的 GenerateContentRequest 对象。用户定义的键用于响应中，以指明哪个输出是哪个请求的结果。例如，如果请求中定义的键为 request-1，则相应响应也会使用相同的键名称进行注释。

此文件是使用 File API 上传的。输入文件的最大允许大小为 2GB。

以下是 JSONL 文件的示例。您可以将其保存在名为 my-batch-requests.json 的文件中：


{"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}], "generation_config": {"temperature": 0.7}}}
{"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}
与内嵌请求类似，您可以在每个 JSON 请求中指定其他参数，例如系统指令、工具或其他配置。

您可以使用 File API 上传此文件，如以下示例所示。如果您要处理多模态输入，可以在 JSONL 文件中引用其他已上传的文件。

Python
JavaScript
REST


import json
from google import genai
from google.genai import types

client = genai.Client()

# Create a sample JSONL file
with open("my-batch-requests.jsonl", "w") as f:
    requests = [
        {"key": "request-1", "request": {"contents": [{"parts": [{"text": "Describe the process of photosynthesis."}]}]}},
        {"key": "request-2", "request": {"contents": [{"parts": [{"text": "What are the main ingredients in a Margherita pizza?"}]}]}}
    ]
    for req in requests:
        f.write(json.dumps(req) + "\n")

# Upload the file to the File API
uploaded_file = client.files.upload(
    file='my-batch-requests.jsonl',
    config=types.UploadFileConfig(display_name='my-batch-requests', mime_type='jsonl')
)

print(f"Uploaded file: {uploaded_file.name}")
以下示例使用 File API 上传的输入文件调用 BatchGenerateContent 方法：

Python
JavaScript
REST

from google import genai

# Assumes `uploaded_file` is the file object from the previous step
client = genai.Client()
file_batch_job = client.batches.create(
    model="gemini-3-flash-preview",
    src=uploaded_file.name,
    config={
        'display_name': "file-upload-job-1",
    },
)

print(f"Created batch job: {file_batch_job.name}")
创建批处理作业时，系统会返回作业名称。此名称可用于监控作业状态，也可用于在作业完成后检索结果。

以下是包含作业名称的输出示例：



Created batch job from file: batches/123456789

支持批量嵌入
您可以使用 Batch API 与 Embeddings 模型进行交互，以实现更高的吞吐量。如需使用内嵌请求或输入文件创建嵌入批量作业，请使用 batches.create_embeddings API 并指定嵌入模型。

Python
JavaScript

from google import genai

client = genai.Client()

# Creating an embeddings batch job with an input file request:
file_job = client.batches.create_embeddings(
    model="gemini-embedding-001",
    src={'file_name': uploaded_batch_requests.name},
    config={'display_name': "Input embeddings batch"},
)

# Creating an embeddings batch job with an inline request:
batch_job = client.batches.create_embeddings(
    model="gemini-embedding-001",
    # For a predefined list of requests `inlined_requests`
    src={'inlined_requests': inlined_requests},
    config={'display_name': "Inlined embeddings batch"},
)
如需查看更多示例，请参阅批量 API Cookbook 中的“嵌入”部分。

请求配置
您可以包含在标准非批量请求中使用的任何请求配置。例如，您可以指定温度、系统指令，甚至传入其他模态。以下示例展示了一个内嵌请求示例，其中包含针对其中一个请求的系统指令：

Python
JavaScript

inline_requests_list = [
    {'contents': [{'parts': [{'text': 'Write a short poem about a cloud.'}]}]},
    {'contents': [{
        'parts': [{
            'text': 'Write a short poem about a cat.'
            }]
        }],
    'config': {
        'system_instruction': {'parts': [{'text': 'You are a cat. Your name is Neko.'}]}}
    }
]
同样，也可以指定要用于请求的工具。以下示例展示了启用 Google 搜索工具的请求：

Python
JavaScript

inlined_requests = [
{'contents': [{'parts': [{'text': 'Who won the euro 1998?'}]}]},
{'contents': [{'parts': [{'text': 'Who won the euro 2025?'}]}],
 'config':{'tools': [{'google_search': {}}]}}]
您还可以指定结构化输出。 以下示例展示了如何为批量请求指定。

Python
JavaScript

import time
from google import genai
from pydantic import BaseModel, TypeAdapter

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

client = genai.Client()

# A list of dictionaries, where each is a GenerateContentRequest
inline_requests = [
    {
        'contents': [{
            'parts': [{'text': 'List a few popular cookie recipes, and include the amounts of ingredients.'}],
            'role': 'user'
        }],
        'config': {
            'response_mime_type': 'application/json',
            'response_schema': list[Recipe]
        }
    },
    {
        'contents': [{
            'parts': [{'text': 'List a few popular gluten free cookie recipes, and include the amounts of ingredients.'}],
            'role': 'user'
        }],
        'config': {
            'response_mime_type': 'application/json',
            'response_schema': list[Recipe]
        }
    }
]

inline_batch_job = client.batches.create(
    model="models/gemini-3-flash-preview",
    src=inline_requests,
    config={
        'display_name': "structured-output-job-1"
    },
)

# wait for the job to finish
job_name = inline_batch_job.name
print(f"Polling status for job: {job_name}")

while True:
    batch_job_inline = client.batches.get(name=job_name)
    if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
        break
    print(f"Job not finished. Current state: {batch_job_inline.state.name}. Waiting 30 seconds...")
    time.sleep(30)

print(f"Job finished with state: {batch_job_inline.state.name}")

# print the response
for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses, start=1):
    print(f"\n--- Response {i} ---")

    # Check for a successful response
    if inline_response.response:
        # The .text property is a shortcut to the generated text.
        print(inline_response.response.text)

以下是此作业的输出示例：


--- Response 1 ---
[
  {
    "recipe_name": "Chocolate Chip Cookies",
    "ingredients": [
      "1 cup (2 sticks) unsalted butter, softened",
      "3/4 cup granulated sugar",
      "3/4 cup packed light brown sugar",
      "1 large egg",
      "1 teaspoon vanilla extract",
      "2 1/4 cups all-purpose flour",
      "1 teaspoon baking soda",
      "1/2 teaspoon salt",
      "1 1/2 cups chocolate chips"
    ]
  },
  {
    "recipe_name": "Oatmeal Raisin Cookies",
    "ingredients": [
      "1 cup (2 sticks) unsalted butter, softened",
      "1 cup packed light brown sugar",
      "1/2 cup granulated sugar",
      "2 large eggs",
      "1 teaspoon vanilla extract",
      "1 1/2 cups all-purpose flour",
      "1 teaspoon baking soda",
      "1 teaspoon ground cinnamon",
      "1/2 teaspoon salt",
      "3 cups old-fashioned rolled oats",
      "1 cup raisins"
    ]
  },
  {
    "recipe_name": "Sugar Cookies",
    "ingredients": [
      "1 cup (2 sticks) unsalted butter, softened",
      "1 1/2 cups granulated sugar",
      "1 large egg",
      "1 teaspoon vanilla extract",
      "2 3/4 cups all-purpose flour",
      "1 teaspoon baking powder",
      "1/2 teaspoon salt"
    ]
  }
]

--- Response 2 ---
[
  {
    "recipe_name": "Gluten-Free Chocolate Chip Cookies",
    "ingredients": [
      "1 cup (2 sticks) unsalted butter, softened",
      "3/4 cup granulated sugar",
      "3/4 cup packed light brown sugar",
      "2 large eggs",
      "1 teaspoon vanilla extract",
      "2 1/4 cups gluten-free all-purpose flour blend (with xanthan gum)",
      "1 teaspoon baking soda",
      "1/2 teaspoon salt",
      "1 1/2 cups chocolate chips"
    ]
  },
  {
    "recipe_name": "Gluten-Free Peanut Butter Cookies",
    "ingredients": [
      "1 cup (250g) creamy peanut butter",
      "1/2 cup (100g) granulated sugar",
      "1/2 cup (100g) packed light brown sugar",
      "1 large egg",
      "1 teaspoon vanilla extract",
      "1/2 teaspoon baking soda",
      "1/4 teaspoon salt"
    ]
  },
  {
    "recipe_name": "Gluten-Free Oatmeal Raisin Cookies",
    "ingredients": [
      "1/2 cup (1 stick) unsalted butter, softened",
      "1/2 cup granulated sugar",
      "1/2 cup packed light brown sugar",
      "1 large egg",
      "1 teaspoon vanilla extract",
      "1 cup gluten-free all-purpose flour blend",
      "1/2 teaspoon baking soda",
      "1/2 teaspoon ground cinnamon",
      "1/4 teaspoon salt",
      "1 1/2 cups gluten-free rolled oats",
      "1/2 cup raisins"
    ]
  }
]
监控作业状态
使用创建批处理作业时获得的操作名称来轮询其状态。批处理作业的状态字段将指示其当前状态。批量作业可能处于以下任一状态：

JOB_STATE_PENDING：作业已创建，正在等待服务处理。
JOB_STATE_RUNNING：作业正在处理中。
JOB_STATE_SUCCEEDED：作业已成功完成。您现在可以检索结果了。
JOB_STATE_FAILED：作业失败。如需了解详情，请查看错误详情。
JOB_STATE_CANCELLED：作业已被用户取消。
JOB_STATE_EXPIRED：作业已过期，因为其运行或等待时间超过 48 小时。相应作业将没有任何结果可供检索。 您可以尝试重新提交作业，或将请求拆分为较小的批次。
您可以定期轮询作业状态，以检查作业是否已完成。

Python
JavaScript

import time
from google import genai

client = genai.Client()

# Use the name of the job you want to check
# e.g., inline_batch_job.name from the previous step
job_name = "YOUR_BATCH_JOB_NAME"  # (e.g. 'batches/your-batch-id')
batch_job = client.batches.get(name=job_name)

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])

print(f"Polling status for job: {job_name}")
batch_job = client.batches.get(name=job_name) # Initial get
while batch_job.state.name not in completed_states:
  print(f"Current state: {batch_job.state.name}")
  time.sleep(30) # Wait for 30 seconds before polling again
  batch_job = client.batches.get(name=job_name)

print(f"Job finished with state: {batch_job.state.name}")
if batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")
检索结果
当作业状态表明您的批处理作业已成功完成时，结果会显示在 response 字段中。

Python
JavaScript
REST

import json
from google import genai

client = genai.Client()

# Use the name of the job you want to check
# e.g., inline_batch_job.name from the previous step
job_name = "YOUR_BATCH_JOB_NAME"
batch_job = client.batches.get(name=job_name)

if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

    # If batch job was created with a file
    if batch_job.dest and batch_job.dest.file_name:
        # Results are in a file
        result_file_name = batch_job.dest.file_name
        print(f"Results are in file: {result_file_name}")

        print("Downloading result file content...")
        file_content = client.files.download(file=result_file_name)
        # Process file_content (bytes) as needed
        print(file_content.decode('utf-8'))

    # If batch job was created with inline request
    # (for embeddings, use batch_job.dest.inlined_embed_content_responses)
    elif batch_job.dest and batch_job.dest.inlined_responses:
        # Results are inline
        print("Results are inline:")
        for i, inline_response in enumerate(batch_job.dest.inlined_responses):
            print(f"Response {i+1}:")
            if inline_response.response:
                # Accessing response, structure may vary.
                try:
                    print(inline_response.response.text)
                except AttributeError:
                    print(inline_response.response) # Fallback
            elif inline_response.error:
                print(f"Error: {inline_response.error}")
    else:
        print("No results found (neither file nor inline).")
else:
    print(f"Job did not succeed. Final state: {batch_job.state.name}")
    if batch_job.error:
        print(f"Error: {batch_job.error}")
列出批处理作业
您可以列出最近的批处理作业。

Python
JavaScript
REST

batch_jobs = client.batches.list()

# Optional query config:
# batch_jobs = client.batches.list(config={'page_size': 5})

for batch_job in batch_jobs:
    print(batch_job)
取消批量作业
您可以使用正在进行的批处理作业的名称来取消该作业。当作业被取消时，它会停止处理新请求。

Python
JavaScript
REST

client.batches.cancel(name=batch_job_to_cancel.name)
删除批处理作业
您可以使用现有批处理作业的名称来删除该作业。删除作业后，该作业会停止处理新请求，并从批处理作业列表中移除。

Python
JavaScript
REST

client.batches.delete(name=batch_job_to_delete.name)
批量生成图片
如果您使用的是 Gemini Nano Banana，并且需要生成大量图片，则可以使用 Batch API 来获得更高的速率限制，但相应地，处理时间最长可达 24 小时。

您可以针对小批量请求（小于 20MB）使用内嵌请求，也可以针对大批量请求使用 JSONL 输入文件（建议用于图片生成）：

内嵌请求 输入文件

Python
JavaScript
REST

import json
import time
import base64
from google import genai
from google.genai import types
from PIL import Image

client = genai.Client()

# 1. Create and upload file
file_name = "my-batch-image-requests.jsonl"
with open(file_name, "w") as f:
    requests = [
        {"key": "request-1", "request": {"contents": [{"parts": [{"text": "A big letter A surrounded by animals starting with the A letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}},
        {"key": "request-2", "request": {"contents": [{"parts": [{"text": "A big letter B surrounded by animals starting with the B letter"}]}], "generation_config": {"responseModalities": ["TEXT", "IMAGE"]}}}
    ]
    for req in requests:
        f.write(json.dumps(req) + "\n")

uploaded_file = client.files.upload(
    file=file_name,
    config=types.UploadFileConfig(display_name='my-batch-image-requests', mime_type='jsonl')
)
print(f"Uploaded file: {uploaded_file.name}")

# 2. Create batch job
file_batch_job = client.batches.create(
    model="gemini-3-pro-image-preview",
    src=uploaded_file.name,
    config={
        'display_name': "file-image-upload-job-1",
    },
)
print(f"Created batch job: {file_batch_job.name}")

# 3. Monitor job status
job_name = file_batch_job.name
print(f"Polling status for job: {job_name}")

completed_states = set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
])

batch_job = client.batches.get(name=job_name) # Initial get
while batch_job.state.name not in completed_states:
  print(f"Current state: {batch_job.state.name}")
  time.sleep(10) # Wait for 10 seconds before polling again
  batch_job = client.batches.get(name=job_name)

print(f"Job finished with state: {batch_job.state.name}")

# 4. Retrieve results
if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
    result_file_name = batch_job.dest.file_name
    print(f"Results are in file: {result_file_name}")
    print("Downloading result file content...")
    file_content_bytes = client.files.download(file=result_file_name)
    file_content = file_content_bytes.decode('utf-8')
    # The result file is also a JSONL file. Parse and print each line.
    for line in file_content.splitlines():
      if line:
        parsed_response = json.loads(line)
        if 'response' in parsed_response and parsed_response['response']:
            for part in parsed_response['response']['candidates'][0]['content']['parts']:
              if part.get('text'):
                print(part['text'])
              elif part.get('inlineData'):
                print(f"Image mime type: {part['inlineData']['mimeType']}")
                data = base64.b64decode(part['inlineData']['data'])
        elif 'error' in parsed_response:
            print(f"Error: {parsed_response['error']}")
elif batch_job.state.name == 'JOB_STATE_FAILED':
    print(f"Error: {batch_job.error}")
技术详情
支持的模型：Batch API 支持一系列 Gemini 模型。 如需了解每种模型对 Batch API 的支持情况，请参阅“模型”页面。Batch API 支持的模态与交互式（或非批量）API 支持的模态相同。
价格：批量 API 的使用费用为相应模型的标准交互式 API 费用的 50%。如需了解详情，请参阅价格页面。如需详细了解此功能的速率限制，请参阅速率限制页面。
服务等级目标 (SLO)：批量作业旨在在 24 小时内完成。许多作业可能会更快完成，具体取决于其大小和当前系统负载。
缓存：已为批量请求启用上下文缓存。如果批处理中的某个请求导致缓存命中，则缓存令牌的定价与非批处理 API 流量相同。
最佳做法
使用输入文件处理大型请求：对于大量请求，请始终使用文件输入方法，以便更好地进行管理，并避免达到 BatchGenerateContent 调用的请求大小限制。请注意，每个输入文件的大小上限为 2GB。
错误处理：作业完成后，检查 batchStats 是否有 failedRequestCount。如果使用文件输出，请解析每一行，以检查该行是 GenerateContentResponse 还是指示相应特定请求出现错误的状态对象。如需查看完整的错误代码列表，请参阅问题排查指南。
仅提交一次作业：批量作业的创建不是幂等的。如果您两次发送相同的创建请求，系统会创建两个单独的批处理作业。
拆分非常大的批次：虽然目标周转时间为 24 小时，但实际处理时间可能会因系统负载和作业大小而异。对于大型作业，如果需要尽快获得中间结果，请考虑将其拆分为较小的批次。
后续步骤
如需查看更多示例，请参阅批量 API 笔记本。
OpenAI 兼容性层支持 Batch API。请参阅 OpenAI 兼容性页面上的示例。