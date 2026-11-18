"""Standalone async Python client and CLI. Requires Python 3.10+ and httpx."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

import httpx


class WeVisOneClient:
    def __init__(self, base_url="http://127.0.0.1:8000", timeout=300.0):
        self.http = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def __aenter__(self):
        await self.http.__aenter__()
        return self

    async def __aexit__(self, *exc):
        await self.http.__aexit__(*exc)

    async def aclose(self):
        await self.http.aclose()

    async def health(self):
        response = await self.http.get("/healthz")
        response.raise_for_status()
        return response.json()

    async def predict_batch(
        self,
        images: Sequence[str | Path],
        *,
        task="detection",
        query: str | Sequence[str] = "",
        prompt: str = "",
        reference_boxes: Sequence[Sequence[float]] = (),
        decode_matting=True,
        max_new_tokens=1024,
    ) -> list[dict]:
        """Upload local images in one request and return one result per image, in order."""
        if not images or isinstance(images, (str, bytes)):
            raise ValueError("pass a nonempty sequence of image paths")
        files = []
        for image in images:
            path = Path(image)
            contents = await asyncio.to_thread(path.read_bytes)
            files.append(("images", (path.name, contents, "application/octet-stream")))
        data = {
            "task": task,
            "query": query if isinstance(query, str) else "\n".join(query),
            "prompt": prompt,
            "reference_boxes": json.dumps(list(reference_boxes)),
            "decode_matting": str(decode_matting).lower(),
            "max_new_tokens": str(max_new_tokens),
        }
        response = await self.http.post("/v1/predict", data=data, files=files)
        response.raise_for_status()
        payload = await asyncio.to_thread(response.json)
        return payload["results"]

    async def predict(self, image: str | Path, **options) -> dict:
        return (await self.predict_batch([image], **options))[0]


async def main():
    parser = argparse.ArgumentParser(description="WeVisOne-VL batch inference client")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--task", default="detection")
    parser.add_argument("--query", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--reference-boxes", type=json.loads, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON results to this file")
    args = parser.parse_args()
    try:
        async with WeVisOneClient(args.url, timeout=args.timeout) as client:
            results = await client.predict_batch(
                args.images,
                task=args.task,
                query=args.query,
                prompt=args.prompt,
                reference_boxes=args.reference_boxes,
                max_new_tokens=args.max_new_tokens,
                decode_matting=not args.no_masks,
            )
        output = json.dumps(results, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    except httpx.HTTPStatusError as exc:
        parser.exit(1, f"HTTP {exc.response.status_code}: {exc.response.text}\n")
    except (httpx.RequestError, OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
