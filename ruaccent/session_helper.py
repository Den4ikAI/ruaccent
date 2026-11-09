from onnxruntime import InferenceSession, SessionOptions, get_available_providers


def create_session(model_path: str, device: str = "CPU") -> InferenceSession:
    """
    Creates an ONNX Runtime InferenceSession configured for the specified device/provider.

    Supported device values:
      - 'CPU': Standard CPU execution provider (default).
      - 'CoreML' / 'MPS' / 'Apple': Uses CoreMLExecutionProvider for hardware acceleration
        on Apple Silicon (M1/M2/M3/M4 via Neural Engine and GPU) with CPU fallback.
      - 'CUDA': Uses CUDAExecutionProvider for NVIDIA GPUs with CPU fallback.
      - 'auto': Automatically selects CoreML on Apple Silicon, CUDA on NVIDIA, or CPU.
    """
    device_upper = (device or "CPU").upper()
    sess_opts = SessionOptions()
    sess_opts.log_severity_level = 3  # Suppress partition warnings on CoreML

    available = get_available_providers()

    if device_upper in ("COREML", "MPS", "APPLE", "ANE"):
        if "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    elif device_upper == "CUDA":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    elif device_upper in ("AUTO", "DETECT"):
        if "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    return InferenceSession(model_path, sess_options=sess_opts, providers=providers)
