import os
import torch
import numpy as np
from PIL import Image
import spandrel

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ESRGAN_PATH = os.environ.get(
    "PIPELINE_ESRGAN_PATH",
    os.path.join(_REPO_ROOT, "models", "upscale", "4x-UltraSharp.pth"),
)


class UpscaleEngine:
    def __init__(self, model_path=DEFAULT_ESRGAN_PATH, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Upscaler model not found at {self.model_path}\n"
                    f"Run 'python setup/download_models.py' first (or set "
                    f"PIPELINE_ESRGAN_PATH env var)."
                )
            loader = spandrel.ModelLoader(device=self.device)
            model_desc = loader.load_from_file(self.model_path)
            self.model = model_desc.model
            self.model.eval()

    def upscale(self, pil_image, target_min_dim=1024):
        """
        Upscales PIL image using 4x-UltraSharp ESRGAN if dimensions are smaller than target_min_dim.
        Returns upscaled PIL Image.
        """
        w, h = pil_image.size
        if min(w, h) >= target_min_dim:
            return pil_image  # already large enough

        self._load_model()

        # Convert PIL to Torch tensor [1, C, H, W] in [0, 1]
        img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out_tensor = self.model(tensor)

        out_np = out_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out_pil = Image.fromarray((out_np * 255.0).round().astype(np.uint8))

        # If upscaled exceeds target by a lot, resize cleanly with Lanczos to optimal bounding
        out_w, out_h = out_pil.size
        scale_down = max(target_min_dim / out_w, target_min_dim / out_h)
        if scale_down < 1.0:
            new_w = int(out_w * scale_down)
            new_h = int(out_h * scale_down)
            out_pil = out_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        return out_pil
