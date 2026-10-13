"""EP18 prospective pixel-only and separate text feature adapters.

This module performs no work at import. It does not download assets, open a
source image/event table or establish a filesystem boundary. The coordinator
must supply qualified local assets and enforce the declared worker delivery.
All invocation/loading belongs inside a coordinated Slurm allocation.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import socket


CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
BLIP_REVISION = "82a37760796d32b1411fe092ab5d4e227313294b"
DINOV2_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
DIMENSIONS = {"low_level_851": 851, "supervised_resnet50_v1": 2048,
              "dino_resnet50": 2048, "clip_vit_b32_image": 512,
              "blip_caption_clip_text": 512, "dinov2_vit_b14": 1536}


def require_compute():
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in socket.gethostname():
        raise RuntimeError("A coordinator-owned, verified compute allocation is required")


def pixels_only(array):
    """Accept the declared decoded-pixel interface, not a filename/PIL object."""
    import numpy as np
    if type(array) is not np.ndarray or array.dtype != np.uint8:
        raise ValueError("Input must be an ordinary uint8 RGB ndarray")
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
        raise ValueError("Expected nonempty HxWx3 decoded RGB pixels")
    return array


def validate_ids(ids, count):
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("Batch IDs must be unique and match its image count")
    if any(not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{32}", key) is None for key in ids):
        raise ValueError("Only opaque 32-hex image IDs may enter the image worker")


def gabor_response(gray, kernel):
    """Convolution using the recipe's half-sample-symmetric reflect boundary."""
    import numpy as np
    from scipy.signal import fftconvolve
    border = (kernel.shape[0]-1)//2
    return fftconvolve(np.pad(gray, border, mode="symmetric"), kernel, mode="valid")


def low_level(array):
    """The fixed 851-D recipe; no fitted state and no source-file access."""
    require_compute()
    import numpy as np
    from PIL import Image
    from skimage.feature import hog

    array = pixels_only(array)
    height, width = array.shape[:2]
    rgb = np.asarray(Image.fromarray(array).resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float64) / 255.
    gray = rgb @ np.array([.2126, .7152, .0722])
    values = []
    for row in range(4):
        for column in range(4):
            cell = rgb[row*64:(row+1)*64, column*64:(column+1)*64]
            for channel in range(3):
                values.extend((cell[..., channel].mean(), cell[..., channel].std(ddof=0)))
    for channel in range(3):
        values.extend(np.histogram(rgb[..., channel], bins=16, range=(0., 1.))[0] / 65536.)

    spectrum = np.fft.fft2(gray - gray.mean(), norm="forward")
    fy, fx = np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij")
    radius, angle = np.hypot(fx, fy), np.mod(np.arctan2(fy, fx), np.pi)
    radial_edges = [0., 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, np.sqrt(.5)]
    power = np.log1p(np.abs(spectrum)**2)
    for index, (lower, upper) in enumerate(zip(radial_edges[:-1], radial_edges[1:])):
        radial = (radius >= lower) & ((radius <= upper) if index == 7 else (radius < upper)) & (radius > 0)
        for orientation in range(8):
            mask = radial & (angle >= orientation*np.pi/8) & (angle < (orientation+1)*np.pi/8)
            values.append(power[mask].mean() if np.any(mask) else 0.)

    for frequency in [1/32, 1/16, 1/8, 1/4]:
        sigma = .56/frequency
        border = int(np.ceil(3*sigma))
        y, x = np.mgrid[-border:border+1, -border:border+1]
        # np.pad symmetric is scipy.ndimage's half-sample-symmetric reflect.
        # FFT convolution changes cost, not the selected boundary/kernel.
        for orientation in range(8):
            theta = orientation*np.pi/8
            kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma)) * np.exp(2j*np.pi*frequency*(x*np.cos(theta) + y*np.sin(theta)))
            kernel -= kernel.mean()
            kernel /= np.sqrt(np.sum(np.abs(kernel)**2))
            response = np.abs(gabor_response(gray, kernel))
            values.extend((response.mean(), response.std(ddof=0)))
    values.extend(hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1),
                      block_norm="L2-Hys", visualize=False, transform_sqrt=False,
                      feature_vector=True, channel_axis=None))
    values.extend((width/height, gray.mean(), gray.std(ddof=0)))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (851,) or not np.isfinite(result).all():
        raise ValueError("The fixed low-level descriptor is incomplete or nonfinite")
    return result


def local_snapshot(path, revision):
    """Require the previously qualified exact-revision local snapshot path.

    Basename matching is a routing assertion, not proof of file provenance.
    Actual acquisition/source qualification remains the coordinator's work.
    """
    path = Path(path)
    if path.name != revision or not path.is_dir():
        raise ValueError(f"Supply the qualified local snapshot directory named {revision}")
    return str(path)


def configure_torch(device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(1)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this allocation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def normalize_rows(tensor):
    import torch
    norms = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    if not torch.isfinite(tensor).all() or not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Nonfinite or zero-norm embedding; no exclusion/replacement")
    return tensor / norms


def load_hf_model(model_class, root):
    """Fail if a selected layer would otherwise be initialized without weights."""
    model, loading = model_class.from_pretrained(root, local_files_only=True, output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Incomplete selected model weights: {loading}")
    # A CLIP projected submodel legitimately ignores the other modality and
    # logit_scale in the complete official checkpoint. Preserve loading info
    # for the eventual qualified feature record; do not treat it as a receipt.
    model.ep18_loading_info = loading
    return model


class TextEncoder:
    """Shared pinned CLIP projection; caller separately owns label/caption text."""

    def __init__(self, clip_snapshot, device="cpu"):
        require_compute()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        self.torch, self.device = configure_torch(device), device
        root = local_snapshot(clip_snapshot, CLIP_REVISION)
        self.tokenizer = CLIPTokenizer.from_pretrained(root, local_files_only=True)
        self.tokenizer.truncation_side = "right"
        self.model = load_hf_model(CLIPTextModelWithProjection, root).float().to(device).eval()

    def encode(self, texts):
        require_compute()
        if any(not isinstance(value, str) for value in texts):
            raise TypeError("Text encoder requires strings")
        original = self.tokenizer(texts, padding=False, truncation=False)["input_ids"]
        tokenized = self.tokenizer(texts, max_length=77, truncation=True, padding="max_length", return_tensors="pt")
        # Padding may equal EOS: the final non-padding token is recovered via mask.
        lengths = tokenized["attention_mask"].sum(dim=1)
        for row, length in enumerate(lengths.tolist()):
            if int(tokenized["input_ids"][row, length-1]) != self.tokenizer.eos_token_id:
                raise ValueError("Truncation did not retain terminal EOS")
        with self.torch.inference_mode():
            result = normalize_rows(self.model(**{key: value.to(self.device) for key, value in tokenized.items()}).text_embeds)
        array = result.cpu().numpy()
        if array.shape != (len(texts), 512):
            raise ValueError("Unexpected CLIP projected text dimension")
        return array, {"original_token_lengths": [len(value) for value in original],
                       "truncated": [len(value) > 77 for value in original]}


class ImageExtractor:
    """One registered image route. Only pixels and opaque IDs enter encode()."""

    def __init__(self, family, assets, device="cpu"):
        require_compute()
        if family not in DIMENSIONS:
            raise ValueError("Unknown/unregistered image family")
        self.family, self.device = family, device
        if family == "low_level_851":
            return
        self.torch = configure_torch(device)
        if family in {"supervised_resnet50_v1", "dino_resnet50"}:
            from torchvision import models, transforms
            expected = "resnet50-0676ba61.pth" if family == "supervised_resnet50_v1" else "dino_resnet50_pretrain.pth"
            path = Path(assets[family])
            if path.name != expected or not path.is_file():
                raise ValueError("Supply the selected official checkpoint file")
            self.model = models.resnet50(weights=None)
            if family == "dino_resnet50":
                self.model.fc = self.torch.nn.Identity()
            self.model.load_state_dict(self.torch.load(path, map_location="cpu", weights_only=True), strict=True)
            self.model.fc = self.torch.nn.Identity()
            mode = transforms.InterpolationMode.BILINEAR if family == "supervised_resnet50_v1" else transforms.InterpolationMode.BICUBIC
            self.processor = transforms.Compose([transforms.Resize(256, interpolation=mode), transforms.CenterCrop(224),
                                                  transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        elif family == "clip_vit_b32_image":
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
            root = local_snapshot(assets["clip"], CLIP_REVISION)
            self.processor = CLIPImageProcessor.from_pretrained(root, local_files_only=True)
            self.model = load_hf_model(CLIPVisionModelWithProjection, root)
        elif family == "dinov2_vit_b14":
            from transformers import AutoImageProcessor, Dinov2Model
            root = local_snapshot(assets["dinov2"], DINOV2_REVISION)
            self.processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
            self.model = load_hf_model(Dinov2Model, root)
        else:
            from transformers import BlipForConditionalGeneration, BlipProcessor
            root = local_snapshot(assets["blip"], BLIP_REVISION)
            self.processor = BlipProcessor.from_pretrained(root, local_files_only=True)
            self.model = load_hf_model(BlipForConditionalGeneration, root)
            self.caption_encoder = TextEncoder(assets["clip"], device)
        self.model = self.model.float().to(device).eval()

    def encode(self, opaque_ids, pixel_arrays):
        require_compute()
        import numpy as np
        from PIL import Image
        validate_ids(opaque_ids, len(pixel_arrays))
        arrays = [pixels_only(value) for value in pixel_arrays]
        if not arrays:
            raise ValueError("Empty batch")
        if self.family == "low_level_851":
            return np.stack([low_level(value) for value in arrays]), {}
        images = [Image.fromarray(value) for value in arrays]
        details = {}
        with self.torch.inference_mode():
            if self.family in {"supervised_resnet50_v1", "dino_resnet50"}:
                inputs = self.torch.stack([self.processor(value) for value in images]).to(self.device)
                result = self.model(inputs)
            else:
                inputs = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
                if self.family == "clip_vit_b32_image":
                    result = normalize_rows(self.model(pixel_values=inputs).image_embeds)
                elif self.family == "dinov2_vit_b14":
                    tokens = self.model(pixel_values=inputs).last_hidden_state
                    if tokens.shape[1:] != (257, 768):
                        raise ValueError("Unexpected DINOv2 token geometry")
                    result = self.torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
                else:
                    generated = self.model.generate(pixel_values=inputs, do_sample=False, num_beams=1,
                                                    num_return_sequences=1, max_new_tokens=32, min_new_tokens=0,
                                                    repetition_penalty=1., no_repeat_ngram_size=0)
                    captions = [value.strip() for value in self.processor.batch_decode(generated, skip_special_tokens=True,
                                                                                      clean_up_tokenization_spaces=False)]
                    result_array, text_details = self.caption_encoder.encode(captions)
                    token_rows = generated.cpu().tolist()
                    eos = self.model.config.text_config.sep_token_id
                    # BLIP's unconditional generate starts from one BOS token
                    # and its v4.46.3 implementation stops on SEP, not config EOS.
                    terminated = [eos in row[1:] for row in token_rows]
                    details = {"captions": captions, "empty_caption": [not value for value in captions],
                               "generated_token_ids": token_rows,
                               "terminated_with_SEP": terminated,
                               "length_cap_without_SEP": [not stopped and len(row)-1 >= 32
                                                          for row, stopped in zip(token_rows, terminated)],
                               "text": text_details}
                    result = self.torch.as_tensor(result_array)
            result_array = result.cpu().numpy().astype(np.float32, copy=False)
        if result_array.shape != (len(arrays), DIMENSIONS[self.family]) or not np.isfinite(result_array).all():
            raise ValueError("Selected feature output is incomplete or nonfinite")
        return result_array, details
