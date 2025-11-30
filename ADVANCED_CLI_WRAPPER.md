# 🔬 Advanced Processor - Terminal Usage & VAD Explained

## ⚠️ QUAN TRỌNG: KHÔNG CÓ COMMAND LINE BUILT-IN!

**`advanced_processor.py` KHÔNG có command line interface như `audio_video_processor.py`!**

```powershell
# ❌ KHÔNG WORK
python advanced_processor.py audio.mp3  # ← Không có CLI này!

# ✅ PHẢI DÙNG CODE
python your_script.py  # Script với Python code
```

---

## 🛠️ GIẢI PHÁP: TẠO WRAPPER SCRIPT

### Tạo file `run_advanced.py`:

```python
"""
Wrapper script để chạy advanced_processor.py từ terminal
"""
import argparse
from advanced_processor import AdvancedAudioProcessor, batch_process_folder

def main():
    parser = argparse.ArgumentParser(
        description='Advanced Audio/Video Processor with VAD'
    )
    
    # Positional arguments
    parser.add_argument('input', help='Path to audio/video file or folder')
    
    # Optional arguments
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Batch process folder')
    parser.add_argument('--model', '-m', default='small',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: small)')
    parser.add_argument('--device', '-d', default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--output', '-o', default='advanced_output',
                       help='Output directory (default: advanced_output)')
    
    # Advanced features
    parser.add_argument('--vad', action='store_true',
                       help='Enable Voice Activity Detection')
    parser.add_argument('--normalize', '-n', action='store_true',
                       help='Normalize audio volume')
    parser.add_argument('--formats', nargs='+',
                       default=['json', 'srt'],
                       choices=['json', 'srt', 'txt', 'csv'],
                       help='Export formats (default: json srt)')
    
    # Video/YouTube
    parser.add_argument('--video', '-v', action='store_true',
                       help='Input is video file')
    parser.add_argument('--youtube', '-y', action='store_true',
                       help='Input is YouTube URL')
    
    args = parser.parse_args()
    
    # Batch processing
    if args.batch:
        print(f"Batch processing folder: {args.input}")
        batch_process_folder(
            folder_path=args.input,
            output_base=args.output,
            model_size=args.model,
            device=args.device
        )
    else:
        # Single file processing
        print(f"Processing file: {args.input}")
        
        processor = AdvancedAudioProcessor(
            output_dir=args.output,
            model_size=args.model,
            device=args.device
        )
        
        result = processor.process_advanced(
            args.input,
            is_youtube=args.youtube,
            is_video=args.video,
            use_vad=args.vad,
            normalize=args.normalize,
            export_formats=args.formats
        )
        
        print(f"\n✅ Processing complete!")
        print(f"   Sentences: {result['statistics']['total_sentences']}")
        print(f"   Duration: {result['statistics']['total_duration']:.2f}s")
        print(f"   Export files: {list(result['export_files'].keys())}")

if __name__ == "__main__":
    main()
```

---

## 🚀 CÚ PHÁP TERMINAL (Sau khi tạo wrapper)

### 1. Basic Processing

```powershell
# Đơn giản
python run_advanced.py audio.mp3

# Với model và device
python run_advanced.py audio.mp3 --model small --device cuda
```

---

### 2. With VAD + Normalize

```powershell
# Enable VAD và normalize
python run_advanced.py audio.mp3 --vad --normalize

# Viết tắt
python run_advanced.py audio.mp3 --vad -n
```

---

### 3. Custom Export Formats

```powershell
# Tất cả formats
python run_advanced.py audio.mp3 --formats json srt txt csv

# Chỉ TXT
python run_advanced.py audio.mp3 --formats txt
```

---

### 4. Complete Features

```powershell
# Tất cả features
python run_advanced.py audio.mp3 \
    --model small \
    --device cuda \
    --vad \
    --normalize \
    --formats json srt txt csv \
    --output my_output
```

---

### 5. Video File

```powershell
python run_advanced.py video.mp4 --video --vad -n
```

---

### 6. YouTube URL

```powershell
python run_advanced.py "https://youtube.com/watch?v=xxx" --youtube --vad
```

---

### 7. Batch Processing

```powershell
# Batch process folder
python run_advanced.py audio_folder --batch --model small --device cuda

# Với output directory
python run_advanced.py audio_folder --batch --output batch_results
```

---

## 📊 TẤT CẢ PARAMETERS

```powershell
python run_advanced.py [INPUT] [OPTIONS]

Positional Arguments:
  input                 Path to audio/video file or folder

Required for specific modes:
  --batch, -b          Batch process folder
  --video, -v          Input is video file
  --youtube, -y        Input is YouTube URL

Model & Device:
  --model, -m          Model size: tiny/base/small/medium/large (default: small)
  --device, -d         Device: cuda/cpu (default: auto-detect)
  --output, -o         Output directory (default: advanced_output)

Advanced Features:
  --vad                Enable Voice Activity Detection
  --normalize, -n      Normalize audio volume
  --formats            Export formats: json/srt/txt/csv (default: json srt)

Examples:
  python run_advanced.py audio.mp3
  python run_advanced.py audio.mp3 --vad --normalize
  python run_advanced.py audio.mp3 --formats json srt txt csv
  python run_advanced.py audio_folder --batch --model small --device cuda
  python run_advanced.py video.mp4 --video --vad
  python run_advanced.py "https://youtube.com/..." --youtube
```

---

## 🔍 VAD - THƯVIỆN ĐẶC BIỆT

### Advanced Processor sử dụng gì khác biệt?

**Core Processor (`audio_video_processor.py`):**
```python
# Chỉ dùng:
- Whisper (transcription)
- Pattern-based sentence detection (regex: [.!?])
- Pydub (audio manipulation cơ bản)
```

**Advanced Processor (`advanced_processor.py`):**
```python
# Thêm:
✅ Pydub.silence module (VAD - Voice Activity Detection)
   - detect_nonsilent()
   - split_on_silence()
✅ Audio normalization (pydub effects)
✅ Silence detection algorithms
```

---

### Chi tiết VAD trong Advanced Processor

**Thư viện sử dụng:**
```python
from pydub.silence import detect_nonsilent, split_on_silence
```

**Không phải AI/ML model!** VAD trong advanced_processor dựa trên:
- **Amplitude-based detection**: Phân tích volume (dBFS)
- **Threshold-based**: So sánh với ngưỡng silence_thresh
- **Simple signal processing**: Không dùng neural networks

---

### Cách hoạt động của VAD

```python
# detect_nonsilent()
segments = detect_nonsilent(
    audio,
    min_silence_len=500,    # Silence tối thiểu 500ms
    silence_thresh=-40      # Âm lượng < -40 dBFS = silence
)
```

**Thuật toán:**
1. Quét audio từ đầu đến cuối
2. Đo amplitude (volume) của mỗi chunk (thường 10ms)
3. Nếu amplitude < silence_thresh → Đánh dấu là silence
4. Nếu silence kéo dài >= min_silence_len → Tách đoạn
5. Return các segments không phải silence

**Ví dụ:**
```
Audio: [voice]----silence----[voice]--[voice]----silence----[voice]
            ↑         ↑         ↑        ↑          ↑          ↑
         Start1    End1      Start2   End2      Start3     End3

Returns: [(Start1, End1), (Start2, End2), (Start3, End3)]
```

---

### So sánh với ML-based VAD

**PyDub VAD (Advanced processor dùng):**
- ✅ Đơn giản, nhanh
- ✅ Không cần train
- ✅ CPU-friendly
- ❌ Kém chính xác với noise
- ❌ Không phân biệt speech vs music
- ❌ Sensitive to threshold tuning

**ML-based VAD (WebRTC VAD, Silero VAD):**
- ✅ Chính xác hơn nhiều
- ✅ Robust với noise
- ✅ Phân biệt được speech
- ❌ Phức tạp hơn
- ❌ Cần model files
- ❌ Chậm hơn

---

### Nếu muốn ML-based VAD tốt hơn

**Option 1: Silero VAD** (Recommended)

```python
# Install
pip install silero-vad

# Usage
import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

wav = read_audio('audio.wav')
speech_timestamps = get_speech_timestamps(wav, model)
```

**Option 2: WebRTC VAD**

```python
# Install
pip install webrtcvad

# Usage
import webrtcvad
vad = webrtcvad.Vad(3)  # Aggressiveness 0-3
is_speech = vad.is_speech(frame, sample_rate)
```

---

## 📊 SO SÁNH ĐẶC ĐIỂM

| Feature | Core Processor | Advanced Processor |
|---------|---------------|-------------------|
| **Transcription** | ✅ Whisper | ✅ Whisper |
| **Sentence detection** | ✅ Regex pattern | ✅ Regex pattern |
| **VAD** | ❌ | ✅ Pydub silence detection |
| **Audio normalization** | ❌ | ✅ Pydub effects |
| **Silence detection** | ❌ | ✅ detect_nonsilent() |
| **Split on silence** | ❌ | ✅ split_on_silence() |
| **Export formats** | JSON, SRT | JSON, SRT, TXT, CSV |
| **Batch processing** | Manual | ✅ Built-in |
| **Statistics** | Basic | ✅ Detailed |
| **Command line** | ✅ Built-in | ❌ Need wrapper |
| **Complexity** | ⭐⭐ | ⭐⭐⭐⭐ |

---

## 🔬 THƯ VIỆN DEPENDENCIES

### Core Processor

```txt
openai-whisper    # Transcription
pydub            # Basic audio manipulation
yt-dlp           # YouTube download
torch            # Whisper backend
torchaudio       # Audio processing
numpy            # Math operations
```

### Advanced Processor (Thêm)

```txt
# Advanced thừa kế tất cả dependencies của Core
# PLUS không có gì thêm! Vẫn dùng pydub

# Nhưng sử dụng advanced features của pydub:
pydub.silence    # VAD functions
pydub.effects    # Normalization
```

**Kết luận:** Advanced processor KHÔNG dùng thư viện bổ sung, chỉ dùng advanced features của Pydub!

---

## 💡 KHI NÀO DÙNG ADVANCED?

### Dùng Core Processor khi:
- ✅ Chỉ cần transcription + tách câu
- ✅ Muốn command line đơn giản
- ✅ Không cần VAD
- ✅ Chỉ cần JSON + SRT

### Dùng Advanced Processor khi:
- ✅ Cần Voice Activity Detection
- ✅ Audio có nhiều silence cần loại bỏ
- ✅ Cần normalize audio từ nhiều nguồn
- ✅ Cần export TXT, CSV
- ✅ Cần statistics chi tiết
- ✅ Batch processing nhiều files

---

## 🎯 QUICK EXAMPLES

### Core Processor (Command Line):

```powershell
# Đơn giản - có sẵn CLI
python audio_video_processor.py audio.mp3 --model small --device cuda
```

### Advanced Processor (Need Wrapper):

```powershell
# Tạo wrapper trước
# File: run_advanced.py (copy code ở trên)

# Sau đó chạy
python run_advanced.py audio.mp3 --model small --device cuda --vad --normalize
```

### Advanced Processor (Python Code - Recommended):

```python
# Cách tốt nhất cho advanced
from advanced_processor import AdvancedAudioProcessor

processor = AdvancedAudioProcessor(model_size="small", device="cuda")

result = processor.process_advanced(
    "audio.mp3",
    use_vad=True,
    normalize=True,
    export_formats=['json', 'srt', 'txt', 'csv']
)
```

---

## ✅ SUMMARY

### Advanced Processor đặc biệt ở:

1. **VAD (Voice Activity Detection)**
   - Library: `pydub.silence`
   - Method: Amplitude-based
   - Functions: `detect_nonsilent()`, `split_on_silence()`

2. **Audio Normalization**
   - Library: `pydub.effects`
   - Method: dBFS normalization

3. **Advanced exports**
   - TXT, CSV thêm vào JSON, SRT

4. **Batch processing**
   - Built-in batch function

5. **Statistics**
   - Detailed audio statistics

### Để chạy từ terminal:

1. ❌ Không có built-in CLI
2. ✅ Tạo wrapper script (code ở trên)
3. ✅ Hoặc dùng Python code trực tiếp (recommended)

---

## 📁 FILES CẦN TẠO

Download và tạo các file:

1. **[advanced_processor.py](computer:///mnt/user-data/outputs/advanced_processor.py)** - Main advanced processor
2. **[run_advanced.py](computer:///mnt/user-data/outputs/run_advanced.py)** - NEW wrapper script (will create)

---

**Last updated:** November 26, 2024  
**VAD Library:** Pydub.silence (amplitude-based)  
**Not using:** ML-based VAD models
