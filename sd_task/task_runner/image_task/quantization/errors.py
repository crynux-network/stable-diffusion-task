from __future__ import annotations


class UnsupportedImageQuantizationBackend(ValueError):
    def __init__(self, quant_method: str) -> None:
        super().__init__(f"Unsupported image quantization backend: {quant_method}")
        self.quant_method = quant_method


class UnsupportedComponentFormat(ValueError):
    def __init__(self, format_name: str) -> None:
        super().__init__(f"Unsupported component format: {format_name}")
        self.format_name = format_name


class ImageQuantizationRuntimeUnsupported(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ImageQuantizationConfigConflict(ValueError):
    def __init__(self, component: str) -> None:
        super().__init__(
            f"Component {component} has both embedded and task quantization"
        )
        self.component = component
