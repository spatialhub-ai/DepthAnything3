import argparse
import logging
from pathlib import Path
import sys

import onnx
import torch
import torch.nn as nn

sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

sys.path.append(str(Path(__file__).parent / "src"))

from depth_anything_3.api import DepthAnything3

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class DA3ONNXWrapper(nn.Module):
    """PyTorch nn.Module wrapper around DepthAnything3 for ONNX export.

    Unpacks model output dictionary into a deterministic output tuple for ONNX tracing.
    """

    def __init__(
        self,
        da3_model: DepthAnything3,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> None:
        """Initialize DA3 ONNX wrapper.

        Args:
            da3_model: Pretrained DepthAnything3 model instance.
            infer_gs: Flag to trigger Gaussian Splatting parameter head.
            use_ray_pose: Flag to use ray pose parameterization.
            ref_view_strategy: Reference view selection strategy name.
        """
        super().__init__()
        self.model = da3_model
        self.infer_gs = infer_gs
        self.use_ray_pose = use_ray_pose
        self.ref_view_strategy = ref_view_strategy

    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute forward pass and return tuple of predictions.

        Args:
            image: Input image tensor of shape (B, N, 3, H, W).
            extrinsics: Optional camera extrinsics tensor of shape (B, N, 4, 4).
            intrinsics: Optional camera intrinsics tensor of shape (B, N, 3, 3).

        Returns:
            Tuple of (depth, depth_conf, sky, extrinsics_out, intrinsics_out).
        """
        out = self.model(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            infer_gs=self.infer_gs,
            use_ray_pose=self.use_ray_pose,
            ref_view_strategy=self.ref_view_strategy,
        )

        depth = out.get("depth", torch.empty(0, device=image.device))
        conf = out.get("depth_conf", torch.empty(0, device=image.device))
        sky = out.get("sky", torch.empty(0, device=image.device))
        extrinsics_out = out.get("extrinsics", torch.empty(0, device=image.device))
        intrinsics_out = out.get("intrinsics", torch.empty(0, device=image.device))

        return depth, conf, sky, extrinsics_out, intrinsics_out


def load_model(
    model_name: str = "depth-anything/DA3-BASE",
    device: str = "cpu",
) -> DepthAnything3:
    """Load pretrained DepthAnything3 PyTorch model.

    Args:
        model_name: Model preset name or local Hugging Face path.
        device: Target hardware device ('cpu' or 'cuda').

    Returns:
        DepthAnything3: Loaded evaluation-mode model.
    """
    logger.info("Loading DepthAnything3 preset '%s' onto device '%s'...", model_name, device)
    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(device)
    return model.eval()


def export_onnx(
    model: DepthAnything3,
    onnx_path: str | Path,
    device: str = "cpu",
    opset_version: int = 18,
    num_views: int = 2,
    height: int = 504,
    width: int = 504,
) -> Path:
    """Export DepthAnything3 model to ONNX format.

    Args:
        model: DepthAnything3 model instance.
        onnx_path: Destination path for exported .onnx model file.
        device: Hardware device to use during export tracing.
        opset_version: ONNX operator set version (default: 18).
        num_views: Dummy view count for tracing.
        height: Dummy image height (must be patch-divisible).
        width: Dummy image width (must be patch-divisible).

    Returns:
        Path: Destination path of exported ONNX model file.
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wrapped_model = DA3ONNXWrapper(
        da3_model=model,
        infer_gs=False,
        use_ray_pose=False,
        ref_view_strategy="saddle_balanced",
    ).to(device)
    wrapped_model.eval()

    logger.info("Generating dummy input tensors (B=1, N=%d, H=%d, W=%d)...", num_views, height, width)
    dummy_image = torch.randn(1, num_views, 3, height, width, device=device, dtype=torch.float32)
    dummy_ext = torch.eye(4, device=device).reshape(1, 1, 4, 4).repeat(1, num_views, 1, 1)
    dummy_int = torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(1, num_views, 1, 1)
    dummy_inputs = (dummy_image, dummy_ext, dummy_int)

    input_names = ["image", "extrinsics_in", "intrinsics_in"]
    output_names = ["depth", "depth_conf", "sky", "extrinsics_out", "intrinsics_out"]

    dynamic_axes = {
        "image": {1: "num_views", 3: "height", 4: "width"},
        "extrinsics_in": {1: "num_views"},
        "intrinsics_in": {1: "num_views"},
        "depth": {1: "num_views"},
        "depth_conf": {1: "num_views"},
        "sky": {1: "num_views"},
        "extrinsics_out": {1: "num_views"},
        "intrinsics_out": {1: "num_views"},
    }

    logger.info("Exporting ONNX graph (opset %d) -> %s...", opset_version, onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            str(onnx_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    logger.info("ONNX export completed successfully!")
    return onnx_path


def validate_onnx(onnx_path: str | Path) -> bool:
    """Validate exported ONNX model graph.

    Args:
        onnx_path: Path to exported .onnx model file.

    Returns:
        bool: True if graph is clean and valid.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at {onnx_path}")

    logger.info("Checking ONNX model integrity at %s...", onnx_path)
    model = onnx.load(str(onnx_path))

    try:
        onnx.checker.check_model(model)
        logger.info("ONNX graph validation passed cleanly!")
        return True
    except onnx.checker.ValidationError as err:
        logger.error("ONNX graph validation failed: %s", err)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Depth Anything 3 (DA3) PyTorch model to ONNX format.")
    parser.add_argument("--model-name", type=str, default="depth-anything/DA3-BASE", help="DA3 model preset or Hugging Face path.")
    parser.add_argument("--onnx-path", type=str, default="weights/da3_base.onnx", help="Destination path for exported .onnx file.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use during export ('cpu' or 'cuda').")
    parser.add_argument("--opset", type=int, default=18, help="ONNX operator set version.")
    parser.add_argument("--views", type=int, default=2, help="Number of dummy views for tracing.")
    parser.add_argument("--height", type=int, default=504, help="Dummy height (must be patch divisible).")
    parser.add_argument("--width", type=int, default=504, help="Dummy width (must be patch divisible).")
    args = parser.parse_args()

    model = load_model(model_name=args.model_name, device=args.device)

    exported_path = export_onnx(
        model=model,
        onnx_path=args.onnx_path,
        device=args.device,
        opset_version=args.opset,
        num_views=args.views,
        height=args.height,
        width=args.width,
    )

    validate_onnx(onnx_path=exported_path)


if __name__ == "__main__":
    main()


