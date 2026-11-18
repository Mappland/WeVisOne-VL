import asyncio
import base64
import threading
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")
pytest.importorskip("colorlog")
pytest.importorskip("python_multipart")

from wevisone_vl import WeVisOneVL
from wevisone_vl.api import create_app


class StubModel(WeVisOneVL):
    """Exercise the real predict_batch wrapper with deterministic model outputs."""

    def __init__(self):
        self.calls = []

    def generate_text_batch(self, images, prompts, **kwargs):
        self.calls.append((len(images), kwargs))
        return ["<ref>person</ref><box><0><0><1000><1000><OBJ></box>"] * len(images)

    def decode_masks(self, image, prompt, response):
        return [[0, 0, 1, 1]], [np.full((image.height, image.width), 0.5)]


def file(width):
    buffer = BytesIO()
    Image.new("RGB", (width, 8)).save(buffer, format="PNG")
    return "images", ("image.png", buffer.getvalue(), "image/png")


@asynccontextmanager
async def service(model):
    app = create_app(lambda *args, **kwargs: model)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client


def test_batch_order_masks_and_validation(monkeypatch):
    monkeypatch.setenv("WEVISONE_MAX_BATCH", "2")
    model = StubModel()

    async def check():
        async with service(model) as client:
            files = [file(12), file(20)]
            data = {"task": "matting", "query": "person", "max_new_tokens": "32"}
            response = await client.post("/v1/predict", files=files, data=data)
            response.raise_for_status()
            items = response.json()["results"]
            assert [item["width"] for item in items] == [12, 20]
            assert model.calls == [(2, {"max_new_tokens": 32})]
            for item in items:
                with Image.open(BytesIO(base64.b64decode(item["masks_png_base64"][0]))) as mask:
                    assert mask.size == (item["width"], 8)
                    assert mask.getpixel((0, 0)) == 128
                assert item["predictions"]["person"][0]["coords"] == [0, 0, item["width"], 8]

            invalid = await client.post("/v1/predict", files=files * 2, data=data)
            assert invalid.status_code == 422
            broken = await client.post("/v1/predict", files={"images": ("x", b"bad")}, data=data)
            assert broken.status_code == 422
            assert len(model.calls) == 1

    asyncio.run(check())


def test_busy_engine_serves_health_and_retains_slot_after_cancel(monkeypatch):
    monkeypatch.setenv("WEVISONE_MAX_INFLIGHT", "1")
    started, release = threading.Event(), threading.Event()

    class WaitingModel(StubModel):
        def generate_text_batch(self, *args, **kwargs):
            started.set()
            assert release.wait(5)
            return super().generate_text_batch(*args, **kwargs)

    async def check():
        async with service(WaitingModel()) as client:

            def predict():
                return client.post("/v1/predict", files=[file(12)], data={"task": "ocr"})

            request = asyncio.create_task(predict())
            try:
                assert await asyncio.to_thread(started.wait, 3)
                assert (await client.get("/healthz")).json()["status"] == "ready"
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                busy = await predict()
                assert busy.status_code == 429
                assert busy.headers["Retry-After"] == "1"
            finally:
                release.set()

            for _ in range(100):
                retry = await predict()
                if retry.status_code != 429:
                    break
                await asyncio.sleep(0.01)
            assert retry.status_code == 200

    asyncio.run(check())
