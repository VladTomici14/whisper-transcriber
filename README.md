# Whisper Engine Transcriber

Lightweight command\-line tool that uses OpenAI Whisper to transcribe and translate audio/video files to SRT subtitle files. Interactive device selection (CPU, CUDA, MPS) and automatic SRT generation from model segments.

## Features
- Transcribe and translate audio/video using Whisper.
- Outputs standard SRT subtitle files.
- Interactive device selection: `cpu`, `cuda:{n}`, or `mps` (Apple Silicon GPU).
- Automatically derives output name from `input` when `-o` is omitted.

## Requirements
- Python 3.8+
- PyTorch (with CUDA support if using GPU)
- `whisper` package
- `ffmpeg` (installed system\-wide)
- macOS note: `mps` support available on Apple Silicon with compatible PyTorch build

## Installation
Install dependencies (example):
```bash
python -m pip install -r requirements.txt
```

Ensure `ffmpeg` is installed (Homebrew on macOS):
```bash
brew install ffmpeg
```

## Usage
Run the transcriber:
```bash
python transcriber.py -i path/to/`input.mov` -o path/to/`output.srt`
```

If `-o` is omitted, the output file defaults to `output.srt` or is derived from the input filename (e.g. `input.mov` -> `input.srt`) when `-i` is provided.

When started the script lists available devices and prompts:
- Enter the number for the device to use (e.g. `0` for `cpu`, `1` for `cuda:0`, etc.)

Example:
```bash
python transcriber.py -i `lecture.mov`
# then select a device by number when prompted
```

## Notes & Troubleshooting
- The script bypasses SSL verification internally to handle restrictive network environments.
- If using CUDA, verify `torch.cuda.is_available()` and compatible drivers.
- On Apple Silicon, install a PyTorch build with `mps` support to use `mps` device.
- If you see `FileNotFoundError`, confirm the `-i` path exists.

