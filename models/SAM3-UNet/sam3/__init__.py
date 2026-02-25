# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

def build_sam3_image_model(*args, **kwargs):
	from .model_builder import build_sam3_image_model as _build
	return _build(*args, **kwargs)

__version__ = "0.1.0"

__all__ = ["build_sam3_image_model"]
