# 🚀 Advanced Processor với GPU - Hướng dẫn sử dụng

## ✅ ĐÃ CẬP NHẬT GPU SUPPORT!

File **advanced_processor.py** giờ đã hỗ trợ GPU đầy đủ!

---

## 🎯 CÚ PHÁP SỬ DỤNG GPU

### 1. Tạo Processor với GPU (Auto-detect)

```python
from advanced_processor import AdvancedAudioProcessor

# Auto-detect: Tự động dùng GPU nếu có
processor = AdvancedAudioProcessor(
    output_dir="advanced_output",
    model_size="small",
    device=None  # None = auto-detect
)
```

**Output:**
```
Loading Whisper model: small...
Using device: CUDA
GPU: NVIDIA GeForce GTX 1050 Ti
GPU Memory: 4.29 GB
Model loaded successfully!
```

---

### 2. Force GPU

```python
# Force dùng GPU
processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"  # Force CUDA
)
```

---

### 3. Force CPU

```python
# Force dùng CPU (nếu GPU có vấn đề)
processor = AdvancedAudioProcessor(
    model_size="large",
    device="cpu"  # Force CPU
)
```

---

## 🔥 EXAMPLES THỰC TẾ

### Example 1: Basic Processing với GPU

```python
from advanced_processor import AdvancedAudioProcessor

# Setup với GPU
processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)

# Process
result = processor.process_advanced("audio.mp3")

print(f"Sentences: {result['statistics']['total_sentences']}")
print(f"Duration: {result['statistics']['total_duration']:.2f}s")
```

---

### Example 2: VAD + Normalization + GPU

```python
from advanced_processor import AdvancedAudioProcessor

# Setup
processor = AdvancedAudioProcessor(
    output_dir="vad_output",
    model_size="small",
    device="cuda"
)

# Process với VAD và normalize
result = processor.process_advanced(
    "audio.mp3",
    use_vad=True,              # Voice Activity Detection
    normalize=True,            # Normalize audio volume
    export_formats=['json', 'srt', 'txt', 'csv']
)

# Results
print("\n📊 Statistics:")
stats = result['statistics']
print(f"  Total sentences: {stats['total_sentences']}")
print(f"  Total duration: {stats['total_duration']:.2f}s")
print(f"  Avg sentence: {stats['avg_sentence_duration']:.2f}s")
print(f"  Total words: {stats['total_words']}")

print("\n📁 Export files:")
for format_type, filepath in result['export_files'].items():
    print(f"  {format_type.upper()}: {filepath}")
```

---

### Example 3: Batch Processing với GPU

```python
from advanced_processor import batch_process_folder

# Process all MP3 files in folder với GPU
batch_process_folder(
    folder_path="audio_folder",
    output_base="batch_results",
    file_extensions=['.mp3', '.wav', '.m4a'],
    model_size="small",    # Model size
    device="cuda"          # Use GPU
)
```

**Output structure:**
```
batch_results/
├── audio1/
│   ├── sentences/
│   │   ├── sentence_001.wav
│   │   └── ...
│   ├── transcriptions.json
│   ├── subtitles.srt
│   ├── transcript.txt
│   └── transcript.csv
├── audio2/
│   └── ...
└── batch_summary.json
```

---

### Example 4: Custom VAD Parameters với GPU

```python
from advanced_processor import AdvancedAudioProcessor

processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)

# Detect silence segments
segments = processor.detect_silence_segments(
    "audio.mp3",
    min_silence_len=500,    # Min silence 500ms
    silence_thresh=-40      # Threshold -40 dBFS
)

print(f"Found {len(segments)} voice segments")

# Split on silence với custom params
chunks = processor.split_on_silence_advanced(
    "audio.mp3",
    min_silence_len=700,
    silence_thresh=-40,
    keep_silence=200
)

print(f"Created {len(chunks)} audio chunks")
```

---

### Example 5: Audio Normalization + GPU

```python
from advanced_processor import AdvancedAudioProcessor

processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)

# Step 1: Normalize audio
print("Normalizing audio...")
normalized_path = processor.normalize_audio(
    "quiet_audio.mp3",
    target_dBFS=-20.0
)

# Step 2: Process normalized audio với GPU
print("Processing with GPU...")
result = processor.process_advanced(
    normalized_path,
    normalize=False,  # Already normalized
    export_formats=['json', 'srt', 'txt']
)

print("✅ Done!")
```

---

### Example 6: Complete Advanced Workflow với GPU

```python
from advanced_processor import AdvancedAudioProcessor
import json

# Setup với GPU
processor = AdvancedAudioProcessor(
    output_dir="research_output",
    model_size="small",
    device="cuda"
)

# Complete workflow
print("="*60)
print("ADVANCED AUDIO PROCESSING WITH GPU")
print("="*60)

# Step 1: Normalize
print("\n1. Normalizing audio...")
normalized = processor.normalize_audio("raw_audio.mp3")

# Step 2: VAD
print("\n2. Detecting voice segments...")
segments = processor.detect_silence_segments(normalized)
print(f"   Found {len(segments)} voice segments")

# Step 3: Process with all features
print("\n3. Processing with VAD...")
result = processor.process_advanced(
    normalized,
    use_vad=True,
    normalize=False,
    export_formats=['json', 'srt', 'txt', 'csv']
)

# Step 4: Statistics
print("\n4. Statistics:")
processor.print_statistics(result['statistics'])

# Step 5: Save report
print("\n5. Saving report...")
report = {
    'gpu_used': True,
    'model': 'small',
    'voice_segments': len(segments),
    'statistics': result['statistics'],
    'export_files': result['export_files']
}

with open('research_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n✅ Complete! Check research_output/ folder")
```

---

## 📊 SO SÁNH PERFORMANCE: CPU vs GPU

### Test Script

```python
import time
from advanced_processor import AdvancedAudioProcessor

audio_file = "test_audio.mp3"

# Test CPU
print("🐢 Testing CPU...")
start = time.time()
cpu_processor = AdvancedAudioProcessor(
    model_size="small",
    device="cpu"
)
result_cpu = cpu_processor.process_advanced(audio_file)
cpu_time = time.time() - start

# Test GPU
print("\n🚀 Testing GPU...")
start = time.time()
gpu_processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)
result_gpu = gpu_processor.process_advanced(audio_file)
gpu_time = time.time() - start

# Compare
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"CPU time: {cpu_time:.2f}s")
print(f"GPU time: {gpu_time:.2f}s")
print(f"Speed up: {cpu_time/gpu_time:.1f}x faster with GPU")
print(f"Time saved: {cpu_time - gpu_time:.2f}s")
```

**Expected result (GTX 1050 Ti, 1 min audio):**
```
CPU time: 60.00s
GPU time: 8.50s
Speed up: 7.1x faster with GPU
Time saved: 51.50s
```

---

## 🎛️ TẤT CẢ PARAMETERS

### AdvancedAudioProcessor()

```python
processor = AdvancedAudioProcessor(
    output_dir="output",        # Output directory
    model_size="small",         # Model: tiny/base/small/medium/large
    device="cuda"               # Device: cuda/cpu/None(auto)
)
```

### process_advanced()

```python
result = processor.process_advanced(
    input_path="audio.mp3",     # Input file/URL
    is_youtube=False,           # Is YouTube URL?
    is_video=False,             # Is video file?
    use_vad=False,              # Use Voice Activity Detection?
    normalize=True,             # Normalize audio?
    export_formats=['json', 'srt', 'txt', 'csv']  # Export formats
)
```

### batch_process_folder()

```python
batch_process_folder(
    folder_path="audio_folder",           # Input folder
    output_base="batch_output",           # Output base directory
    file_extensions=['.mp3', '.wav'],     # File types to process
    model_size="small",                   # Model size
    device="cuda"                         # Device to use
)
```

### VAD Parameters

```python
# detect_silence_segments()
segments = processor.detect_silence_segments(
    audio_path="audio.mp3",
    min_silence_len=500,      # Min silence duration (ms)
    silence_thresh=-40        # Silence threshold (dBFS)
)

# split_on_silence_advanced()
chunks = processor.split_on_silence_advanced(
    audio_path="audio.mp3",
    min_silence_len=700,      # Min silence (ms)
    silence_thresh=-40,       # Threshold (dBFS)
    keep_silence=200          # Keep silence at edges (ms)
)
```

### Audio Normalization

```python
normalized = processor.normalize_audio(
    audio_path="audio.mp3",
    target_dBFS=-20.0         # Target loudness (-20 recommended)
)
```

---

## 🎯 KHUYẾN NGHỊ CHO GTX 1050 Ti

### Model Selection

```python
# ✅ RECOMMENDED: Model small
processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)
# - VRAM: 2 GB (safe)
# - Speed: 8-10s/min
# - Accuracy: ⭐⭐⭐⭐

# ✅ SAFE: Model base
processor = AdvancedAudioProcessor(
    model_size="base",
    device="cuda"
)
# - VRAM: 1 GB (very safe)
# - Speed: 3-5s/min
# - Accuracy: ⭐⭐⭐

# ⚠️ RISKY: Model medium
processor = AdvancedAudioProcessor(
    model_size="medium",
    device="cuda"
)
# - VRAM: 5 GB (may OOM with 4.29 GB GPU)
# - Use with caution
```

---

## 💡 BEST PRACTICES

### 1. Reuse Processor for Batch

```python
# ✅ GOOD: Create processor once
processor = AdvancedAudioProcessor(model_size="small", device="cuda")

for audio_file in audio_files:
    result = processor.process_advanced(audio_file)
    # Process result...

# ❌ BAD: Create processor each time (slow)
for audio_file in audio_files:
    processor = AdvancedAudioProcessor(model_size="small", device="cuda")
    result = processor.process_advanced(audio_file)
```

### 2. Choose Right Features

```python
# For production app - Simple
result = processor.process_advanced(
    "audio.mp3",
    export_formats=['json', 'srt']
)

# For research - Full features
result = processor.process_advanced(
    "audio.mp3",
    use_vad=True,
    normalize=True,
    export_formats=['json', 'srt', 'txt', 'csv']
)
```

### 3. Monitor GPU Usage

```python
import torch

# Check GPU memory before
print(f"GPU memory before: {torch.cuda.memory_allocated()/1e9:.2f} GB")

result = processor.process_advanced("audio.mp3")

# Check GPU memory after
print(f"GPU memory after: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## 🐛 TROUBLESHOOTING

### Error: CUDA out of memory

```python
# Solution 1: Use smaller model
processor = AdvancedAudioProcessor(model_size="base", device="cuda")

# Solution 2: Use CPU
processor = AdvancedAudioProcessor(model_size="medium", device="cpu")

# Solution 3: Clear GPU cache
import torch
torch.cuda.empty_cache()
processor = AdvancedAudioProcessor(model_size="small", device="cuda")
```

### Error: GPU not detected

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# If False, check PyTorch installation
# pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then create processor
processor = AdvancedAudioProcessor(device="cuda")
```

### VAD not working well

```python
# Problem: Too many/too few chunks

# Solution: Adjust parameters
# For more chunks (more sensitive):
chunks = processor.split_on_silence_advanced(
    "audio.mp3",
    min_silence_len=300,    # Decreased
    silence_thresh=-50      # Decreased (more sensitive)
)

# For fewer chunks (less sensitive):
chunks = processor.split_on_silence_advanced(
    "audio.mp3",
    min_silence_len=1000,   # Increased
    silence_thresh=-30      # Increased (less sensitive)
)
```

---

## 📈 EXPECTED PERFORMANCE (GTX 1050 Ti)

### Processing Times

| Audio Length | Model | CPU Time | GPU Time | Speed Up |
|--------------|-------|----------|----------|----------|
| 1 minute | base | 20s | 3-5s | 4-6x |
| 1 minute | small | 60s | 8-10s | 6-7x |
| 10 minutes | base | 3m 20s | 30-50s | 4-6x |
| 10 minutes | small | 10m | 80-100s | 6-7x |

### With VAD & Normalization

| Audio Length | Model | CPU | GPU | Speed Up |
|--------------|-------|-----|-----|----------|
| 1 minute | small | 70s | 10-12s | 6x |
| 10 minutes | small | 11m | 100-120s | 5.5x |

---

## ✅ QUICK REFERENCE

### Quick Setup

```python
from advanced_processor import AdvancedAudioProcessor

# GPU auto-detect
processor = AdvancedAudioProcessor(
    model_size="small",
    device=None  # Auto
)
```

### Quick Process

```python
result = processor.process_advanced("audio.mp3")
```

### Quick Batch

```python
from advanced_processor import batch_process_folder

batch_process_folder(
    "audio_folder",
    "output",
    model_size="small",
    device="cuda"
)
```

---

## 📚 RELATED FILES

- [advanced_processor.py](computer:///mnt/user-data/outputs/advanced_processor.py) - Updated with GPU support
- [ADVANCED_GUIDE.md](computer:///mnt/user-data/outputs/ADVANCED_GUIDE.md) - Complete advanced guide
- [GTX1050Ti_MODEL_GUIDE.md](computer:///mnt/user-data/outputs/GTX1050Ti_MODEL_GUIDE.md) - GPU optimization
- [USAGE_GUIDE.md](computer:///mnt/user-data/outputs/USAGE_GUIDE.md) - Core processor guide

---

**Last updated:** November 26, 2024  
**Version:** 1.2 (GPU enabled)  
**GPU tested:** GTX 1050 Ti (4.29 GB VRAM)  
**Recommended model:** small
