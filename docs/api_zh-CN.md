# WeVisOne-VL 异步推理服务部署

服务启动时加载一个模型。一次 HTTP 请求可以上传一张或多张图片，多图会进入同一次 `predict_batch` 调用。模型先批量生成文本，然后按需逐图解码掩码。并发请求在服务器端排队，等待期间仍可访问健康检查。每个 HTTP 请求对应一个模型批次。

## 安装与启动

以下命令适用于 Linux、Bash 和 Python 3.12。GPU 部署需要能运行仓库所固定 PyTorch 版本的 NVIDIA 驱动，可先用 `nvidia-smi` 查看设备。

```bash
git clone https://github.com/Mappland/WeVisOne-VL.git
cd WeVisOne-VL
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[api]'
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

export WEVISONE_MODEL=tencent/WeVisOne-VL-2B
export WEVISONE_DEVICE=cuda:0
CUDA_VISIBLE_DEVICES=0 python -m uvicorn wevisone_vl.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

首次启动需要能访问 Hugging Face 并有空间缓存权重。离线部署时，先将完整模型快照复制到服务器，再把 `WEVISONE_MODEL` 设为该目录的路径。目录应包含模型权重、处理器、分词器以及 `mask_decoder.pt`。看到日志 `Ready` 后，在另一个终端验证服务。

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/v1/predict \
  -F images=@demo/examples/images/livingroom.jpg \
  -F images=@demo/examples/images/dogs.jpg \
  -F task=detection -F query=dog -F max_new_tokens=256
```

`/healthz` 返回模型名和允许的最大批量。上例的 `results[0]` 对应客厅照片，`results[1]` 对应狗的照片。交互式接口文档位于 `/docs`。

## Python 客户端

客户端机器只需 Python 3.10+、`httpx` 和仓库根目录的 `wevisone_client.py`。在仓库根目录运行以下命令。异机调用时，用 `--url` 指定服务器可访问的地址。

```bash
python -m pip install 'httpx>=0.27,<1'
python wevisone_client.py --task detection --query dog \
  demo/examples/images/dogs.jpg demo/examples/images/livingroom.jpg
python wevisone_client.py --task matting --query person \
  --output matting.json demo/examples/images/seabreeze.jpg
```

在 Python 应用中，复用同一个客户端可以共享连接池。下面的两个请求可以同时提交。

```python
import asyncio
import base64
from pathlib import Path
from wevisone_client import WeVisOneClient


async def main():
    photos = Path("demo/examples/images")
    async with WeVisOneClient(timeout=300) as client:
        batch, matte = await asyncio.gather(
            client.predict_batch([photos / "dogs.jpg", photos / "livingroom.jpg"],
                                 task="detection", query=["dog", "sofa"]),
            client.predict(photos / "seabreeze.jpg", task="matting", query="person"),
        )
    print([item["predictions"] for item in batch])
    for i, mask in enumerate(matte["masks_png_base64"]):
        Path(f"person-{i}.png").write_bytes(base64.b64decode(mask))


asyncio.run(main())
```

## 请求与响应

`POST /v1/predict` 接受 `multipart/form-data`。使用重复的 `images` 文件字段传入 JPEG、PNG 或 WebP。每张最多 8 MiB 和 1600 万像素；批内图片共用下列表单字段。

| 字段 | 默认值 | 用法 |
| --- | --- | --- |
| `task` | `detection` | 支持 `detection`、`referring`、`point`、`count_box`、`count_point`、`ocr`、`ocr_grounding`、`segmentation`、`matting`、`visual_prompt`、`freeform` |
| `query` | 空 | 目标类别或描述。多类别用换行分隔。OCR 可省略，计数任务留空则统计物体 |
| `prompt` | 空 | 自定义完整提示词，优先于任务模板 |
| `reference_boxes` | `[]` | 视觉提示任务的像素坐标，例如 `[[10,20,100,200]]`。批内每张图都须容纳这些框 |
| `decode_matting` | `true` | 设为 `false` 可跳过像素解码并减少响应体积 |
| `max_new_tokens` | `1024` | 生成长度上限，可取 1 至 4096 |

成功响应为 `{"request_id":"…","results":[…]}`。每项包含原图宽高、实际提示词 `prompt`、原始生成文本 `response` 以及按类别分组的 `predictions`。其中的框和点使用原图像素坐标。像素任务另外返回归一化到 0–1 的 `seg_boxes_normalized_xyxy`，以及顺序对应的 `masks_png_base64`。每个掩码都是原图尺寸的 8 位灰度 PNG，0 表示透明，255 表示不透明。没有对应输出时集合为空。

## 运行配置

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `WEVISONE_MODEL` | `tencent/WeVisOne-VL-4B` | 模型 ID 或本地快照路径。上面的启动示例显式选择 2B |
| `WEVISONE_BACKEND` | `transformers` | 若在支持 vLLM 的 Linux 环境部署，可安装 `pip install -e '.[api,vllm]'` 并设为 `vllm`。此模式还会加载用于掩码解码的 Transformers 模型，需要为两者预留显存 |
| `WEVISONE_DEVICE` | `auto` | 自动选择设备，也可指定 `cuda:0` 或 `cpu` |
| `WEVISONE_DTYPE` | `bfloat16` | 根据 GPU 能力可改为 `float16`。仓库引擎在 CPU 上使用 `float32` |
| `WEVISONE_MAX_BATCH` | `4` | 单次请求最多接收的图片数 |
| `WEVISONE_MAX_INFLIGHT` | `4` | 同时接纳的推理请求数，包括执行中和排队中的请求 |
| `WEVISONE_LOG_LEVEL` | `INFO` | 应用日志级别。终端按级别着色，可设 `NO_COLOR=1` 关闭颜色 |

每张 GPU 启动一个进程并使用 `--workers 1`，避免重复加载权重。扩容时可以给不同进程分配独立的 GPU 和端口。对外提供服务时，通过带鉴权和 HTTPS 的反向代理转发到本机端口，并在代理层配置整个上传请求的大小限制和读取超时。默认批量下可用 34 MiB 作为上传上限，为表单头预留空间。

| 现象 | 处理方式 |
| --- | --- |
| HTTP 413 或 422 | 按错误详情缩小图片、减少张数或修正参数。单个输入出错时整批返回错误 |
| HTTP 429 | 已达到并发上限。按 `Retry-After` 等待后重试，或降低客户端并发 |
| HTTP 500 | 用响应中的请求 ID 查找后端异常。如果显存不足，减少每批图片数、生成长度或切换 2B |
| 客户端超时 | 核对模型耗时和排队情况，按需调高客户端与代理的超时。客户端超时后，已开始的批次仍可能继续运行 |
