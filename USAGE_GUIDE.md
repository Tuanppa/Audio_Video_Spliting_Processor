# 📘 Hướng dẫn sử dụng Audio/Video Processor

## 📊 SO SÁNH 2 FILES

### `audio_video_processor.py` - CORE (Cơ bản)

**Tác dụng:**
- ✅ Xử lý audio/video/YouTube cơ bản
- ✅ Transcription với Whisper
- ✅ Tách câu tự động
- ✅ Tạo file audio cho từng câu
- ✅ Export JSON + SRT
- ✅ Hỗ trợ GPU (auto-detect)

**Khi nào dùng:**
- 🎯 **Sử dụng hàng ngày** - 90% use cases
- 📱 Tích hợp vào app/production
- 🚀 Cần đơn giản, dễ dùng
- 📝 Chỉ cần JSON + SRT output

**Dung lượng:** 14 KB  
**Độ phức tạp:** ⭐⭐ (Đơn giản)

---

### `advanced_processor.py` - ADVANCED (Nâng cao)

**Tác dụng:**
- ✅ **TẤT CẢ tính năng của core PLUS:**
- ✅ Voice Activity Detection (VAD) - tách câu chính xác hơn
- ✅ Normalize audio (chuẩn hóa âm lượng)
- ✅ Batch processing (xử lý hàng loạt)
- ✅ Export nhiều format: JSON, SRT, TXT, CSV
- ✅ Statistics & analytics
- ✅ Silence detection
- ✅ Advanced filtering

**Khi nào dùng:**
- 🔬 **Nghiên cứu/Testing** - cần control chi tiết
- 📊 Cần statistics và analytics
- 🎛️ Cần fine-tune parameters
- 📁 Batch processing nhiều file cùng lúc
- 📝 Cần export nhiều format khác nhau

**Dung lượng:** 13 KB  
**Độ phức tạp:** ⭐⭐⭐⭐ (Nâng cao)

---

## 🎯 BẢNG SO SÁNH CHI TIẾT

| Tính năng | audio_video_processor.py | advanced_processor.py |
|-----------|-------------------------|----------------------|
| **Basic transcription** | ✅ | ✅ |
| **Sentence splitting** | ✅ | ✅ |
| **Export JSON** | ✅ | ✅ |
| **Export SRT** | ✅ | ✅ |
| **Export TXT** | ❌ | ✅ |
| **Export CSV** | ❌ | ✅ |
| **GPU support** | ✅ | ✅ |
| **Voice Activity Detection** | ❌ | ✅ |
| **Audio normalization** | ❌ | ✅ |
| **Silence detection** | ❌ | ✅ |
| **Batch processing** | ❌ | ✅ |
| **Statistics** | ❌ | ✅ |
| **Command line** | ✅ | ❌ (Python only) |
| **Difficulty** | Dễ | Khó hơn |

---

## 📖 HƯỚNG DẪN SỬ DỤNG `audio_video_processor.py`

### 🚀 Cách 1: Command Line (Đơn giản nhất)

#### A. Xử lý Audio File

```powershell
# Cơ bản - Dùng mặc định (model base, auto GPU)
python audio_video_processor.py audio.mp3

# Chỉ định model
python audio_video_processor.py audio.mp3 --model small

# Chỉ định device
python audio_video_processor.py audio.mp3 --device cuda

# Chỉ định output folder
python audio_video_processor.py audio.mp3 --output my_output

# Kết hợp tất cả
python audio_video_processor.py audio.mp3 --model small --device cuda --output results
```

#### B. Xử lý Video File

```powershell
# Video file
python audio_video_processor.py video.mp4 --video

# Video với model small
python audio_video_processor.py video.mp4 --video --model small --device cuda
```

#### C. Xử lý YouTube Video

```powershell
# YouTube URL
python audio_video_processor.py "https://www.youtube.com/watch?v=VIDEO_ID" --youtube

# YouTube với options
python audio_video_processor.py "https://youtube.com/watch?v=xxx" --youtube --model small
```

---

### 🎛️ PARAMETERS CHI TIẾT

```powershell
python audio_video_processor.py <INPUT> [OPTIONS]
```

**INPUT (bắt buộc):**
- Audio file: `audio.mp3`, `audio.wav`, `audio.m4a`
- Video file: `video.mp4` (cần thêm `--video`)
- YouTube URL: `"https://youtube.com/..."` (cần thêm `--youtube`)

**OPTIONS (không bắt buộc):**

| Option | Short | Values | Default | Mô tả |
|--------|-------|--------|---------|-------|
| `--youtube` | `-y` | flag | false | Input là YouTube URL |
| `--video` | `-v` | flag | false | Input là video file |
| `--model` | `-m` | tiny/base/small/medium/large | base | Whisper model size |
| `--device` | `-d` | cuda/cpu | auto | Device để xử lý |
| `--output` | `-o` | path | output | Output directory |

---

### 💻 Cách 2: Python Code (Linh hoạt hơn)

#### A. Basic Usage

```python
from audio_video_processor import AudioVideoProcessor

# Tạo processor
processor = AudioVideoProcessor(
    output_dir="output",
    model_size="base",
    device="cuda"  # or "cpu" or None for auto
)

# Xử lý audio
result = processor.process("audio.mp3")

# Kết quả
print(f"Sentences: {result['total_sentences']}")
print(f"Audio files: {len(result['audio_files'])}")
print(f"JSON: {result['transcription_json']}")
print(f"SRT: {result['transcription_srt']}")
```

#### B. Xử lý Video

```python
processor = AudioVideoProcessor()

# Process video
result = processor.process("video.mp4", is_video=True)
```

#### C. Xử lý YouTube

```python
processor = AudioVideoProcessor()

# Process YouTube
result = processor.process(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    is_youtube=True
)
```

#### D. Batch Processing (Nhiều file)

```python
from pathlib import Path
from audio_video_processor import AudioVideoProcessor

# Tạo processor 1 lần (efficient)
processor = AudioVideoProcessor(
    model_size="small",
    device="cuda"
)

# Get all MP3 files
audio_files = list(Path("audio_folder").glob("*.mp3"))

# Process all
results = []
for audio_file in audio_files:
    print(f"Processing: {audio_file.name}")
    
    result = processor.process(str(audio_file))
    results.append({
        'file': audio_file.name,
        'sentences': result['total_sentences']
    })
    
    print(f"✅ Done: {result['total_sentences']} sentences")

# Summary
print(f"\n📊 Processed {len(results)} files")
for r in results:
    print(f"  {r['file']}: {r['sentences']} sentences")
```

#### E. Custom Configuration

```python
processor = AudioVideoProcessor(
    output_dir="my_output",
    model_size="small",
    device="cuda"
)

# Access specific functions
result = processor.process("audio.mp3")

# Lấy sentences
sentences = result['sentences']
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}:")
    print(f"  Text: {sentence['text']}")
    print(f"  Time: {sentence['start']:.2f}s - {sentence['end']:.2f}s")
    print(f"  Duration: {sentence['end'] - sentence['start']:.2f}s")
```

#### F. Error Handling

```python
from audio_video_processor import AudioVideoProcessor

processor = AudioVideoProcessor(model_size="small", device="cuda")

try:
    result = processor.process("audio.mp3")
    print(f"✅ Success: {result['total_sentences']} sentences")
    
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("❌ GPU out of memory!")
        print("💡 Try smaller model or CPU:")
        print("   processor = AudioVideoProcessor(model_size='base', device='cpu')")
    else:
        print(f"❌ Error: {e}")
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

---

## 📁 CẤU TRÚC OUTPUT

Sau khi chạy, output folder sẽ có cấu trúc:

```
output/
├── sentences/              # Audio files cho từng câu
│   ├── sentence_001.wav
│   ├── sentence_002.wav
│   ├── sentence_003.wav
│   └── ...
├── transcriptions.json     # Transcription (JSON format)
└── subtitles.srt          # Subtitle (SRT format)
```

### A. File JSON (transcriptions.json)

```json
[
  {
    "id": 1,
    "text": "Hello everyone, welcome to my channel.",
    "start_time": 0.0,
    "end_time": 2.5,
    "duration": 2.5
  },
  {
    "id": 2,
    "text": "Today we will learn about AI.",
    "start_time": 2.8,
    "end_time": 5.2,
    "duration": 2.4
  }
]
```

**Sử dụng JSON:**
```python
import json

with open('output/transcriptions.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(f"Sentence {item['id']}: {item['text']}")
    print(f"  Time: {item['start_time']}s - {item['end_time']}s")
```

### B. File SRT (subtitles.srt)

```
1
00:00:00,000 --> 00:00:02,500
Hello everyone, welcome to my channel.

2
00:00:02,800 --> 00:00:05,200
Today we will learn about AI.
```

**Sử dụng SRT:**
- Import vào video editor (Premiere, DaVinci)
- Dùng cho subtitle trên video
- Upload lên YouTube

### C. Audio Files (sentence_XXX.wav)

```
sentence_001.wav  # Câu 1
sentence_002.wav  # Câu 2
sentence_003.wav  # Câu 3
```

**Sử dụng:**
```python
from pydub import AudioSegment

# Load audio file
audio = AudioSegment.from_wav("output/sentences/sentence_001.wav")

# Play
from pydub.playback import play
play(audio)

# Export to MP3
audio.export("sentence_001.mp3", format="mp3")
```

---

## 🔧 EXAMPLES - USE CASES THỰC TẾ

### Example 1: App học tiếng Anh

```python
from audio_video_processor import AudioVideoProcessor
import json

# Setup
processor = AudioVideoProcessor(
    model_size="small",  # Good accuracy
    device="cuda",       # Fast with GPU
    output_dir="lessons"
)

# Process lesson
result = processor.process("lesson_01.mp3")

# Load transcriptions
with open(result['transcription_json'], 'r', encoding='utf-8') as f:
    sentences = json.load(f)

# Save to database
for sentence in sentences:
    # Insert to database
    db.insert({
        'lesson_id': 1,
        'sentence_order': sentence['id'],
        'text': sentence['text'],
        'start_time': sentence['start_time'],
        'end_time': sentence['end_time'],
        'audio_file': f"sentence_{sentence['id']:03d}.wav"
    })

print(f"✅ Processed {len(sentences)} sentences")
```

### Example 2: YouTube video → Subtitle

```python
from audio_video_processor import AudioVideoProcessor

processor = AudioVideoProcessor(model_size="base")

# Download và process YouTube
youtube_url = "https://www.youtube.com/watch?v=VIDEO_ID"
result = processor.process(youtube_url, is_youtube=True)

# SRT file ready
print(f"✅ Subtitle ready: {result['transcription_srt']}")
print("Upload to YouTube or use in video editor")
```

### Example 3: Podcast → Blog post

```python
from audio_video_processor import AudioVideoProcessor
import json

processor = AudioVideoProcessor(model_size="small")

# Process podcast
result = processor.process("podcast_episode.mp3")

# Load transcription
with open(result['transcription_json'], 'r') as f:
    sentences = json.load(f)

# Create blog post
blog_content = []
for sentence in sentences:
    blog_content.append(sentence['text'])

# Join into paragraphs (every 5 sentences)
paragraphs = []
for i in range(0, len(blog_content), 5):
    paragraph = ' '.join(blog_content[i:i+5])
    paragraphs.append(paragraph)

# Save blog post
with open('blog_post.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(paragraphs))

print("✅ Blog post created!")
```

### Example 4: Meeting recording → Minutes

```python
from audio_video_processor import AudioVideoProcessor
from datetime import datetime
import json

processor = AudioVideoProcessor(model_size="base", device="cuda")

# Process meeting
result = processor.process("meeting_2024-11-26.mp3")

# Load transcription
with open(result['transcription_json'], 'r') as f:
    sentences = json.load(f)

# Create meeting minutes
minutes = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'duration': sentences[-1]['end_time'],
    'total_sentences': len(sentences),
    'transcript': [s['text'] for s in sentences]
}

# Save
with open('meeting_minutes.json', 'w', encoding='utf-8') as f:
    json.dump(minutes, f, indent=2, ensure_ascii=False)

print(f"✅ Meeting minutes created")
print(f"   Duration: {minutes['duration']/60:.1f} minutes")
print(f"   Sentences: {minutes['total_sentences']}")
```

---

## 🎯 KHUYẾN NGHỊ CHO APP HỌC TIẾNG ANH

### Setup cho Production

```python
from audio_video_processor import AudioVideoProcessor

# Production config
processor = AudioVideoProcessor(
    model_size="small",      # Good accuracy for language learning
    device="cuda",           # Fast with GPU (or auto-detect)
    output_dir="lessons"     # Organized output
)

def process_lesson(lesson_file, lesson_id):
    """Process một lesson và save vào database"""
    
    # Process
    result = processor.process(lesson_file)
    
    # Load sentences
    import json
    with open(result['transcription_json'], 'r') as f:
        sentences = json.load(f)
    
    # Save to database
    for sentence in sentences:
        db.lessons.insert({
            'lesson_id': lesson_id,
            'sentence_order': sentence['id'],
            'text': sentence['text'],
            'start_time': sentence['start_time'],
            'end_time': sentence['end_time'],
            'duration': sentence['duration'],
            'audio_url': upload_to_cloudinary(
                result['audio_files'][sentence['id']-1]
            )
        })
    
    return {
        'success': True,
        'sentences_count': len(sentences),
        'duration': sentences[-1]['end_time']
    }

# Process multiple lessons
lessons = [
    ('lesson_01.mp3', 1),
    ('lesson_02.mp3', 2),
    ('lesson_03.mp3', 3)
]

for lesson_file, lesson_id in lessons:
    print(f"Processing lesson {lesson_id}...")
    result = process_lesson(lesson_file, lesson_id)
    print(f"✅ Done: {result['sentences_count']} sentences")
```

---

## 📊 PERFORMANCE TIPS

### 1. Reuse Processor Instance

```python
# ✅ GOOD - Load model once
processor = AudioVideoProcessor(model_size="small", device="cuda")
for audio in audio_files:
    result = processor.process(audio)

# ❌ BAD - Load model nhiều lần (chậm)
for audio in audio_files:
    processor = AudioVideoProcessor(model_size="small", device="cuda")
    result = processor.process(audio)
```

### 2. Choose Right Model

```python
# Fast processing (3-5s/min)
processor = AudioVideoProcessor(model_size="base")

# Balanced (8-10s/min)  ← Recommended for GTX 1050 Ti
processor = AudioVideoProcessor(model_size="small")

# Best accuracy (20-25s/min)
processor = AudioVideoProcessor(model_size="medium")
```

### 3. Use GPU When Available

```python
# Auto-detect (recommended)
processor = AudioVideoProcessor(device=None)  # Will use GPU if available

# Force GPU
processor = AudioVideoProcessor(device="cuda")

# Force CPU (if GPU has issues)
processor = AudioVideoProcessor(device="cpu")
```

---

## 🐛 TROUBLESHOOTING

### Lỗi: "FileNotFoundError: FFmpeg not found"

**Fix:**
```powershell
# Check FFmpeg
ffmpeg -version

# If not found
pip install ffmpeg-python
# Or add to PATH (see WINDOWS_SETUP.md)
```

### Lỗi: "CUDA out of memory"

**Fix:**
```python
# Use smaller model
processor = AudioVideoProcessor(model_size="base", device="cuda")

# Or use CPU
processor = AudioVideoProcessor(model_size="small", device="cpu")
```

### Lỗi: "Audio format not supported"

**Fix:**
```python
# Convert to WAV first
from pydub import AudioSegment
audio = AudioSegment.from_file("input.xyz")
audio.export("input.wav", format="wav")

# Then process
result = processor.process("input.wav")
```

---

## ✅ QUICK CHECKLIST

**Trước khi chạy:**
- [ ] FFmpeg installed: `ffmpeg -version`
- [ ] Python packages installed: `pip install -r requirements.txt`
- [ ] GPU ready (optional): `python test_gpu.py`

**Chạy lần đầu:**
- [ ] Test với audio ngắn: `python audio_video_processor.py test.mp3`
- [ ] Kiểm tra output folder: `ls output/`
- [ ] Check JSON file: `cat output/transcriptions.json`

**Production:**
- [ ] Chọn model phù hợp (small for GTX 1050 Ti)
- [ ] Enable GPU: `--device cuda`
- [ ] Organize output folders
- [ ] Test error handling

---

## 📚 RELATED FILES

- [audio_video_processor.py](computer:///mnt/user-data/outputs/audio_video_processor.py) - Main script
- [advanced_processor.py](computer:///mnt/user-data/outputs/advanced_processor.py) - Advanced version
- [demo.py](computer:///mnt/user-data/outputs/demo.py) - Interactive demo
- [GTX1050Ti_MODEL_GUIDE.md](computer:///mnt/user-data/outputs/GTX1050Ti_MODEL_GUIDE.md) - Model recommendations
- [GPU_SETUP.md](computer:///mnt/user-data/outputs/GPU_SETUP.md) - GPU setup guide

---

**Last updated:** November 26, 2024  
**Version:** 1.2  
**Recommended for:** Production use, app integration, daily tasks
