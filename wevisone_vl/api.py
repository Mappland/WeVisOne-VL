"""Async HTTP service for the native WeVisOne-VL batch engine."""

import asyncio
import base64
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Annotated

import colorlog
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import FiniteFloat, TypeAdapter, ValidationError

from . import TASKS, WeVisOneVL, build_prompt
from .engine import DEFAULT_MODEL_ID
from .visualization import parse_predictions

logger = logging.getLogger("wevisone.api")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
Box = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
box_list = TypeAdapter(list[Box])


def configure_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)s%(reset)s %(message)s",
            stream=sys.stderr,
        )
    )
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(os.getenv("WEVISONE_LOG_LEVEL", "INFO").upper())


def positive_env(name, default):
    value = int(os.getenv(name, default))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def read_image(blob, index):
    try:
        with Image.open(BytesIO(blob), formats=("JPEG", "PNG", "WEBP")) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(413, f"images[{index}] exceeds the pixel limit")
            return image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(413, f"images[{index}] exceeds the pixel limit") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(422, f"images[{index}] is not a readable JPEG, PNG or WebP") from exc


def png_base64(mask):
    data = np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(data).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_batch(
    engine, blobs, task, query, prompt, boxes, decode_matting, max_new_tokens, request_id
):
    started = time.monotonic()
    images = [read_image(blob, i) for i, blob in enumerate(blobs)]
    prompts = []
    for index, image in enumerate(images):
        if any(
            not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height)
            for x1, y1, x2, y2 in boxes
        ):
            raise HTTPException(422, f"invalid reference box for images[{index}]")
        prompts.append(
            prompt or build_prompt(task, query, reference_boxes=boxes, image_size=image.size)
        )
    results = engine.predict_batch(
        images, prompt=prompts, decode_matting=decode_matting, max_new_tokens=max_new_tokens
    )
    items = [
        dict(
            width=image.width,
            height=image.height,
            prompt=result.prompt,
            response=result.response,
            predictions=parse_predictions(result.response, image.width, image.height),
            seg_boxes_normalized_xyxy=result.boxes,
            masks_png_base64=[png_base64(mask) for mask in result.masks],
        )
        for image, result in zip(images, results, strict=True)
    ]
    response = JSONResponse({"request_id": request_id, "results": items})
    logger.info(
        "request=%s task=%s images=%d elapsed=%.3fs",
        request_id,
        task,
        len(images),
        time.monotonic() - started,
    )
    return response


def create_app(engine_factory=WeVisOneVL):
    @asynccontextmanager
    async def lifespan(app):
        configure_logging()
        state = app.state
        state.model_id = os.getenv("WEVISONE_MODEL", DEFAULT_MODEL_ID)
        state.max_batch = positive_env("WEVISONE_MAX_BATCH", 4)
        state.admission = asyncio.Semaphore(positive_env("WEVISONE_MAX_INFLIGHT", 4))
        logger.info("Loading model=%s", state.model_id)
        state.engine = engine_factory(
            state.model_id,
            backend=os.getenv("WEVISONE_BACKEND", "transformers"),
            device=os.getenv("WEVISONE_DEVICE", "auto"),
            dtype=os.getenv("WEVISONE_DTYPE", "bfloat16"),
        )
        logger.info("Ready model=%s max_batch=%d", state.model_id, state.max_batch)
        # The tokenizer and mask hooks are mutable. Give the engine one worker.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference") as pool:
            state.pool = pool
            try:
                yield
            finally:
                del state.engine

    app = FastAPI(title="WeVisOne-VL inference", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "ready", "model": app.state.model_id, "max_batch": app.state.max_batch}

    @app.post("/v1/predict")
    async def predict(
        images: Annotated[list[UploadFile], File(description="One file per image, in order")],
        task: Annotated[str, Form()] = "detection",
        query: Annotated[str, Form(max_length=4096)] = "",
        prompt: Annotated[str, Form(max_length=8192)] = "",
        reference_boxes: Annotated[str, Form(max_length=4096)] = "[]",
        decode_matting: Annotated[bool, Form()] = True,
        max_new_tokens: Annotated[int, Form(ge=1, le=4096)] = 1024,
    ):
        state = app.state
        if not 1 <= len(images) <= state.max_batch:
            raise HTTPException(422, f"expected 1 to {state.max_batch} images")
        if task not in TASKS:
            raise HTTPException(422, f"task must be one of {', '.join(TASKS)}")
        try:
            boxes = box_list.validate_json(reference_boxes)
        except ValidationError as exc:
            raise HTTPException(422, "reference_boxes requires JSON [[x1,y1,x2,y2], ...]") from exc
        if task == "visual_prompt" and not boxes and not prompt:
            raise HTTPException(422, "visual_prompt requires reference_boxes or prompt")
        if not (query.strip() or prompt.strip()) and task not in {
            "ocr",
            "visual_prompt",
            "count_box",
            "count_point",
        }:
            raise HTTPException(422, "supply a query or prompt for this task")
        if state.admission.locked():
            raise HTTPException(429, "inference queue full", headers={"Retry-After": "1"})
        await state.admission.acquire()

        request_id = uuid.uuid4().hex
        try:
            blobs = []
            for i, upload in enumerate(images):
                blob = await upload.read(MAX_IMAGE_BYTES + 1)
                if len(blob) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, f"images[{i}] exceeds 8 MiB")
                blobs.append(blob)
            job = asyncio.get_running_loop().run_in_executor(
                state.pool,
                run_batch,
                state.engine,
                blobs,
                task,
                query,
                prompt,
                boxes,
                decode_matting,
                max_new_tokens,
                request_id,
            )
        except BaseException:
            state.admission.release()
            raise

        def completed(future):
            state.admission.release()
            # Observe failures even if the HTTP caller disconnected.
            error = future.exception()
            if error and not isinstance(error, HTTPException):
                logger.error("Inference failed request=%s", request_id, exc_info=error)

        job.add_done_callback(completed)
        try:
            return await asyncio.shield(job)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, f"inference failed (request {request_id})") from exc

    return app


app = create_app()
