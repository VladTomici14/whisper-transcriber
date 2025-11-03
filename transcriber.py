import whisper
import ssl
import os
import argparse
from pathlib import Path
import torch

# ----- parsing the input / output files path -----
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Name of the input file")
parser.add_argument("-o", "--output", help="Name of the output file")
args = parser.parse_args()

input_file_path = args.input if args.input else "input.mov"
output_file_path = args.output if args.output else "output.srt"

if not os.path.isfile(input_file_path):
    raise FileNotFoundError(f"Input file not found: {input_file_path}")

if input_file_path and args.output is None:
    output_file_path = f"{Path(args.input).stem}.srt"

# ----- list available devices -----
available_devices = ["cpu"]

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        available_devices.append(f"cuda:{i}")

if torch.backends.mps.is_available():
    available_devices.append("mps")

print("\nAvailable devices:")
for idx, dev in enumerate(available_devices):
    if dev.startswith("cuda"):
        print(f"  [{idx}] {dev} - {torch.cuda.get_device_name(int(dev.split(':')[1]))}")
    elif dev == "mps":
        print(f"  [{idx}] {dev} (Apple Silicon GPU)")
    else:
        print(f"  [{idx}] {dev}")

# ----- ask user to choose a device -----
while True:
    try:
        choice = int(input("\nSelect a device by number: "))
        if 0 <= choice < len(available_devices):
            device = available_devices[choice]
            break
        else:
            print("Invalid selection. Try again.")
    except ValueError:
        print("Please enter a valid number.")

print(f"\n✅ Using device: {device}")

# ----- bypassing SSL verification (needed for your network environment) -----
ssl._create_default_https_context = ssl._create_unverified_context

# ----- loading model -----
print("Loading model...")
model = whisper.load_model("medium", device=device)

# ----- extracting the text -----
print("Transcribing and translating...")
result = model.transcribe(
    input_file_path,
    language="ro",
    task="translate",
    verbose=True
)

# ----- saving SRT file -----
print("Saving SRT file...")

# ----- writing in the file -----
with open(output_file_path, "w", encoding="utf-8") as f:
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        
        # ----- converting seconds to SRT time format (HH:MM:SS,mmm) -----
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        # ----- writing in file -----
        f.write(f"{segment['id'] + 1}\n")
        f.write(f"{format_time(start)} --> {format_time(end)}\n")
        f.write(f"{text}\n\n")

print(f"Done! SRT file saved as: {output_file_path}")
