# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse, affine_inverse_np
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

from depth_anything_3.utils.process_sky import process_mono_sky_estimation_np

torch.backends.cudnn.benchmark = False
# logger.info("CUDNN Benchmark Disabled")

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()

        # Device management (set by user)
        self.device = None

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
        export_feat_layers: list[int] = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation instead of camera decoder.
            ref_view_strategy: Strategy for selecting reference view from multiple views.

        Returns:
            Dictionary containing model predictions
        """
        # Disable autocast during ONNX export. Mixed-precision tracing introduces unsupported dtype/cast operations in the exported graph, preventing successful export
        with torch.no_grad():
            return self.model(
                image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
            )

    def preprocess(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> dict[str, np.ndarray | None]:
        """
        NumPy Pre-processing.
        Outputs a dictionary of NumPy arrays ready for model inference.
        """
        # 1. Base processing (Returns NumPy arrays due to our previous changes)
        imgs_cpu, exts, ixts = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # 2. Prepare inputs (Adds batch dimension in NumPy)
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, exts, ixts)

        # 3. Normalize extrinsics (Uses NumPy affine_inverse_np)
        ex_t_norm = self._normalize_extrinsics(ex_t.copy() if ex_t is not None else None)

        return {
            "imgs": imgs,
            "ex_t_norm": ex_t_norm,
            "in_t": in_t,
            "imgs_cpu": imgs_cpu,  # Preserved for post-processing visualization
            "original_extrinsics": extrinsics,
            "original_intrinsics": intrinsics
        }
    
    def infer_pytorch(
        self,
        input: dict[str, np.ndarray | None],
        export_feat_layers: Sequence[int] = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, np.ndarray]:
        """
        PyTorch Inference Boundary.
        Takes NumPy arrays, converts to PyTorch tensors, runs the PyTorch model and converts outputs immediately back to NumPy arrays.
        """
        device = self._get_model_device()

        # Determine Batch and Num Views from the image tensor
        B = input["imgs"].shape[0]
        N = input["imgs"].shape[1]

        if input["ex_t_norm"] is None:
            input["ex_t_norm"] = np.full((B, N, 4, 4), -1.0, dtype=np.float32)
            
        if input["in_t"] is None:
            input["in_t"] = np.full((B, N, 3, 3), -1.0, dtype=np.float32)

        # Convert NumPy inputs to PyTorch tensors and move to device
        imgs_t = torch.from_numpy(input["imgs"]).to(device)
        ex_t_norm_t = torch.from_numpy(input["ex_t_norm"]).to(device)
        in_t_t = torch.from_numpy(input["in_t"]).to(device)
        
        raw_output = self._run_model_forward(
            imgs_t, ex_t_norm_t, in_t_t, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
        )

        # Convert PyTorch outputs back to NumPy instantly at the boundary
        output_dict = {}
        for key, value in raw_output.items():
            if isinstance(value, torch.Tensor):
                # Detach from graph, move to CPU, and convert to NumPy
                output_dict[key] = value.detach().cpu().numpy()
            else:
                output_dict[key] = value

        return output_dict
    
    def postprocess(
        self, 
        np_outputs: dict[str, np.ndarray], 
        np_inputs: dict[str, np.ndarray], 
        align_to_input_ext_scale: bool = True
    ) -> Prediction:
        """
        NumPy Post-processing.
        Takes NumPy outputs from inference and formats them into the Prediction object.
        """
        # Process mono sky estimation in NumPy
        np_outputs["depth"], np_outputs["depth_conf"] = process_mono_sky_estimation_np(depth=np_outputs["depth"], depth_conf=np_outputs.get("depth_conf", None), sky=np_outputs.get("sky", None))

        # Convert raw NumPy outputs to Prediction object
        prediction = self._convert_to_prediction(np_outputs)

        # Align prediction to extrinsics (NumPy math)
        prediction = self._align_to_input_extrinsics_intrinsics(
            np_inputs["original_extrinsics"], 
            np_inputs["original_intrinsics"], 
            prediction, 
            align_to_input_ext_scale
        )

        # Add processed images for visualization (NumPy transpositions)
        prediction = self._add_processed_images(prediction, np_inputs["imgs_cpu"])

        return prediction

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] = [],
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
    ) -> Prediction:
        """
        Run inference on input images.

        Args:
            image: List of input images (numpy arrays, PIL Images, or file paths)
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            align_to_input_ext_scale: whether to align the input pose scale to the prediction
            infer_gs: Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)
            use_ray_pose: Use ray-based pose estimation instead of camera decoder (default: False)
            ref_view_strategy: Strategy for selecting reference view from multiple views.
                Options: "first", "middle", "saddle_balanced", "saddle_sim_range".
                Default: "saddle_balanced". For single view input (S ≤ 2), no reordering is performed.
            render_exts: Optional render extrinsics for Gaussian video export
            render_ixts: Optional render intrinsics for Gaussian video export
            render_hw: Optional render resolution for Gaussian video export
            process_res: Processing resolution
            process_res_method: Resize method for processing
            export_dir: Directory to export results
            export_format: Export format (mini_npz, npz, glb, ply, gs, gs_video)
            export_feat_layers: Layer indices to export intermediate features from
            conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold (default: 40.0) # noqa: E501
            num_max_points: [GLB] Maximum number of points in the point cloud (default: 1,000,000)
            show_cameras: [GLB] Show camera wireframes in the exported scene (default: True)
            feat_vis_fps: [FEAT_VIS] Frame rate for output video (default: 15)
            export_kwargs: additional arguments to export functions.

        Returns:
            Prediction object containing depth maps and camera parameters
        """
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."

        # Preprocess images
        input_dict = self.preprocess(image, extrinsics, intrinsics, process_res, process_res_method)

        # Run model forward pass
        output_dict = self.infer_pytorch(input_dict, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy)

        # Convert raw output to prediction
        prediction = self.postprocess(output_dict, input_dict, align_to_input_ext_scale)

        # Export if requested
        if export_dir is not None:

            if "gs" in export_format:
                if infer_gs and "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format:
                    if "gs_video" not in export_kwargs:
                        export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update(
                        {
                            "extrinsics": render_exts,
                            "intrinsics": render_ixts,
                            "out_image_hw": render_hw,
                        }
                    )
            # Add GLB export parameters
            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update(
                    {
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "num_max_points": num_max_points,
                        "show_cameras": show_cameras,
                    }
                )
            # Add Feat_vis export parameters
            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update(
                    {
                        "fps": feat_vis_fps,
                    }
                )
            # Add COLMAP export parameters
            if "colmap" in export_format:
                if "colmap" not in export_kwargs:
                    export_kwargs["colmap"] = {}
                export_kwargs["colmap"].update(
                    {
                        "image_paths": image,
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "process_res_method": process_res_method,
                    }
                )
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _prepare_model_inputs(
        self,
        imgs_cpu: np.ndarray,
        extrinsics: np.ndarray | None,
        intrinsics: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Prepare tensors for model input."""

        # Add batch dimension
        imgs = np.expand_dims(imgs_cpu, axis=0).astype(np.float32)  # (N, 3, H, W) -> (1, N, 3, H, W)

        ex_t = np.expand_dims(extrinsics, axis=0).astype(np.float32) if extrinsics is not None else None
        in_t = np.expand_dims(intrinsics, axis=0).astype(np.float32) if intrinsics is not None else None

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: np.ndarray | None) -> np.ndarray | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse_np(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse_np(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = np.linalg.norm(translations, axis=-1)
        median_dist = np.median(dists)
        median_dist = np.clip(median_dist, 1e-1, None)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor,
        in_t: torch.Tensor,
        export_feat_layers: Sequence[int] = [],
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        output = self.forward(imgs, ex_t, in_t, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: np.ndarray) -> Prediction:
        """Add processed images to prediction for visualization."""
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = np.transpose(imgs_cpu, (0, 2, 3, 1))  # (N, H, W, 3)

        # Denormalize from ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(
        self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:
        """Export results to specified format and directory."""
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")
