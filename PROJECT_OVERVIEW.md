# 📦 Audio/Video Processor - Project Overview

## 📁 Cấu trúc Project

```
audio-video-processor/
├── 📄 audio_video_processor.py    # Core processor (chính)
├── 📄 advanced_processor.py       # Advanced features với VAD
├── 📄 demo.py                     # Demo script (bắt đầu từ đây)
├── 📄 requirements.txt            # Python dependencies
├── 📄 config.json                 # Configuration file
├── 📖 README.md                   # Hướng dẫn đầy đủ
├── 📖 QUICKSTART.md              # Hướng dẫn nhanh
├── 📖 APP_INTEGRATION.md         # Tích hợp vào iOS app
└── 📄 .gitignore                  # Git ignore rules
```

## 🎯 Tính năng chính

### 1. Core Features (audio_video_processor.py)
- ✅ Xử lý audio files (MP3, WAV, M4A, FLAC)
- ✅ Xử lý video files (MP4, AVI, MOV)
- ✅ Download và xử lý YouTube videos
- ✅ Tách câu tự động (sentence segmentation)
- ✅ Transcription (giọng nói → text) bằng OpenAI Whisper
- ✅ Tạo file audio riêng cho mỗi câu
- ✅ Xuất kết quả: JSON, SRT (subtitle)
- ✅ Hỗ trợ nhiều ngôn ngữ

### 2. Advanced Features (advanced_processor.py)
- ✅ Voice Activity Detection (VAD) - tách câu chính xác hơn
- ✅ Normalize audio (chuẩn hóa âm lượng)
- ✅ Detect silence segments
- ✅ Batch processing (xử lý hàng loạt)
- ✅ Export nhiều format: JSON, SRT, TXT, CSV
- ✅ Statistics & analytics
- ✅ Lọc nhiễu

## 🚀 Quick Start

### Cài đặt
```bash
# 1. Cài FFmpeg
brew install ffmpeg  # MacOS
sudo apt-get install ffmpeg  # Ubuntu

# 2. Cài Python packages
pip install -r requirements.txt

# 3. Chạy demo
python demo.py
```

### Sử dụng cơ bản
```bash
# Audio file
python audio_video_processor.py audio.mp3

# Video file
python audio_video_processor.py video.mp4 --video

# YouTube
python audio_video_processor.py "https://youtube.com/watch?v=xxx" --youtube
```

## 📊 Models & Performance

| Model  | Tốc độ | RAM  | Độ chính xác | Use case |
|--------|--------|------|--------------|----------|
| tiny   | ⚡⚡⚡⚡⚡ | 1GB  | ⭐⭐ | Test nhanh |
| base   | ⚡⚡⚡⚡ | 1GB  | ⭐⭐⭐ | **Khuyến nghị** |
| small  | ⚡⚡⚡ | 2GB  | ⭐⭐⭐⭐ | Cân bằng tốt |
| medium | ⚡⚡ | 5GB  | ⭐⭐⭐⭐⭐ | Chất lượng cao |
| large  | ⚡ | 10GB | ⭐⭐⭐⭐⭐ | Tốt nhất |

## 💡 Use Cases

### 1. 🎓 Học ngoại ngữ
```python
# Tách audio lesson thành các câu để luyện dictation/shadowing
processor = AudioVideoProcessor()
result = processor.process("english_lesson.mp3")
# → Có các file audio từng câu + transcription
```

### 2. 🎬 Tạo subtitle tự động
```python
# Tạo subtitle cho video
processor = AudioVideoProcessor()
result = processor.process("video.mp4", is_video=True)
# → Có file subtitles.srt
```

### 3. 🎙️ Transcribe podcast/bài giảng
```python
# Chuyển podcast thành text
processor = AdvancedAudioProcessor()
result = processor.process_advanced(
    "podcast.mp3",
    export_formats=['json', 'txt', 'csv']
)
```

### 4. 🤖 Chuẩn bị dữ liệu ML
```python
# Batch processing nhiều file
from advanced_processor import batch_process_folder
batch_process_folder("audio_dataset", "processed_output")
```

## 🔧 Customization

### Config file (config.json)
```json
{
  "whisper": {
    "model_size": "base",
    "language": "en"  // "vi" for Vietnamese
  },
  "audio_processing": {
    "normalize_audio": true,
    "target_dbfs": -20.0
  }
}
```

### Programmatic
```python
from audio_video_processor import AudioVideoProcessor

processor = AudioVideoProcessor(
    output_dir="my_output",
    model_size="small"
)

result = processor.process("audio.mp3")
```

## 📱 iOS App Integration

Xem chi tiết: [APP_INTEGRATION.md](APP_INTEGRATION.md)

### Backend API
```python
from fastapi import FastAPI, UploadFile
app = FastAPI()

@app.post("/api/process-audio")
async def process_audio(file: UploadFile):
    # Process audio with AudioProcessor
    # Return sentences with timestamps
    pass
```

### iOS Client
```swift
let api = AudioProcessorAPI()
let jobId = try await api.uploadAudio(fileURL: url)
let sentences = try await api.getSentences(jobId: jobId)
```

## 📈 Architecture

```
┌─────────────────────────────────────────────────┐
│            Input Sources                         │
│  • Audio Files (MP3, WAV, M4A, FLAC)           │
│  • Video Files (MP4, AVI, MOV)                 │
│  • YouTube URLs                                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Audio Extraction & Processing            │
│  • FFmpeg extraction                            │
│  • Audio normalization                          │
│  • VAD (optional)                               │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│      Whisper Transcription Engine               │
│  • Word-level timestamps                        │
│  • Multi-language support                       │
│  • Configurable models                          │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Sentence Segmentation                    │
│  • Pattern-based detection                      │
│  • Timestamp mapping                            │
│  • Duration filtering                           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           Audio Splitting                        │
│  • Extract segments by timestamps               │
│  • Export individual sentence files             │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│             Output Generation                    │
│  • JSON (structured data)                       │
│  • SRT (subtitles)                              │
│  • TXT (plain text)                             │
│  • CSV (spreadsheet)                            │
│  • Individual audio files                       │
└─────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### FFmpeg not found
```bash
# Install FFmpeg first
brew install ffmpeg  # Mac
sudo apt install ffmpeg  # Linux
```

### CUDA out of memory
```bash
# Use smaller model
python audio_video_processor.py audio.mp3 --model tiny
```

### YouTube download error
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

### Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📚 Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) - Bắt đầu nhanh trong 5 phút
- **Full Guide**: [README.md](README.md) - Hướng dẫn chi tiết
- **App Integration**: [APP_INTEGRATION.md](APP_INTEGRATION.md) - Tích hợp vào iOS app
- **API Reference**: Code comments trong `audio_video_processor.py`

## 🎓 Examples

### Example 1: Basic processing
```python
from audio_video_processor import AudioVideoProcessor

processor = AudioVideoProcessor()
result = processor.process("audio.mp3")

print(f"Sentences: {result['total_sentences']}")
for sentence in result['sentences']:
    print(f"{sentence['text']} [{sentence['start']}-{sentence['end']}]")
```

### Example 2: Advanced with VAD
```python
from advanced_processor import AdvancedAudioProcessor

processor = AdvancedAudioProcessor()
result = processor.process_advanced(
    "audio.mp3",
    use_vad=True,
    normalize=True,
    export_formats=['json', 'srt', 'txt', 'csv']
)
```

### Example 3: YouTube processing
```python
processor = AudioVideoProcessor()
result = processor.process(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    is_youtube=True
)
```

### Example 4: Batch processing
```python
from advanced_processor import batch_process_folder

batch_process_folder(
    "audio_folder",
    "output_folder",
    file_extensions=['.mp3', '.wav']
)
```

## 🛣️ Roadmap

### Phase 1: Core ✅
- [x] Audio/Video processing
- [x] Sentence segmentation
- [x] Transcription
- [x] Multiple export formats

### Phase 2: Advanced ✅
- [x] VAD integration
- [x] Batch processing
- [x] Statistics
- [x] Audio normalization

### Phase 3: API (In Progress)
- [ ] FastAPI backend
- [ ] Job queue system
- [ ] Cloud storage integration
- [ ] Authentication

### Phase 4: iOS App
- [ ] SwiftUI interface
- [ ] API client
- [ ] Practice modes
- [ ] Progress tracking

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create feature branch
3. Make changes
4. Submit pull request

## 📄 License

MIT License - Free to use for any purpose

## 💬 Support

- 📧 Email: your@email.com
- 🐛 Issues: GitHub Issues
- 📖 Docs: This README

## 🙏 Acknowledgments

- OpenAI Whisper - Speech recognition
- FFmpeg - Audio/Video processing
- yt-dlp - YouTube download
- pydub - Audio manipulation

---

**Made with ❤️ for English Dictation & Shadowing App**

Last updated: 2024
