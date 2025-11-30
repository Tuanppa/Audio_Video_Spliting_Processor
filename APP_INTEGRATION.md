# 🎯 Tích hợp vào App Học Tiếng Anh

Hướng dẫn tích hợp Audio Processor vào app "English Dictation & Shadowing"

## Kiến trúc đề xuất

```
┌─────────────────────────────────────────────────────────┐
│                   iOS App (SwiftUI)                      │
│  ┌───────────────┐          ┌────────────────────────┐ │
│  │   Dictation   │          │      Shadowing         │ │
│  │   Practice    │          │      Practice          │ │
│  └───────────────┘          └────────────────────────┘ │
│         │                             │                 │
│         └─────────────┬───────────────┘                │
│                       │                                 │
└───────────────────────┼─────────────────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │   FastAPI Backend     │
            │   + Audio Processor   │
            └───────────────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │  Railway Deployment   │
            └───────────────────────┘
```

## Backend API Integration

### 1. Tạo FastAPI endpoints

```python
# audio_api.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from audio_video_processor import AudioVideoProcessor
import uuid
import os

app = FastAPI()
processor = AudioVideoProcessor(model_size="base")

@app.post("/api/process-audio")
async def process_audio(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload audio và xử lý
    Returns: job_id để track progress
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    temp_path = f"/tmp/{job_id}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Process in background
    background_tasks.add_task(
        process_audio_task,
        job_id,
        temp_path
    )
    
    return {
        "job_id": job_id,
        "status": "processing"
    }

def process_audio_task(job_id: str, audio_path: str):
    """Background task để xử lý audio"""
    try:
        result = processor.process(audio_path)
        
        # Save result to database hoặc file
        save_result(job_id, result)
        
    except Exception as e:
        save_error(job_id, str(e))

@app.get("/api/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check processing status"""
    status = get_job_status_from_db(job_id)
    return status

@app.get("/api/sentences/{job_id}")
async def get_sentences(job_id: str):
    """Get processed sentences"""
    result = get_result_from_db(job_id)
    return result["sentences"]

@app.get("/api/download-audio/{job_id}/{sentence_id}")
async def download_sentence_audio(job_id: str, sentence_id: int):
    """Download audio file cho 1 câu"""
    # Return audio file
    pass
```

### 2. Database Schema

```sql
-- Topics table
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sections table
CREATE TABLE sections (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id),
    title TEXT NOT NULL,
    order_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Lessons table
CREATE TABLE lessons (
    id INTEGER PRIMARY KEY,
    section_id INTEGER REFERENCES sections(id),
    title TEXT NOT NULL,
    audio_url TEXT,
    video_url TEXT,
    processing_status TEXT, -- 'pending', 'processing', 'completed', 'error'
    order_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentences table (kết quả xử lý)
CREATE TABLE sentences (
    id INTEGER PRIMARY KEY,
    lesson_id INTEGER REFERENCES lessons(id),
    sentence_order INTEGER,
    text TEXT NOT NULL,
    start_time REAL,
    end_time REAL,
    duration REAL,
    audio_file_url TEXT,
    difficulty_level TEXT, -- 'easy', 'medium', 'hard'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Progress table
CREATE TABLE user_progress (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    sentence_id INTEGER REFERENCES sentences(id),
    practice_type TEXT, -- 'dictation' or 'shadowing'
    completed_at TIMESTAMP,
    score REAL,
    attempts INTEGER DEFAULT 1
);
```

## iOS App Integration

### 1. Network Layer (Swift)

```swift
// AudioProcessorAPI.swift
import Foundation

class AudioProcessorAPI {
    private let baseURL = "https://your-api.railway.app"
    
    // Upload audio file
    func uploadAudio(fileURL: URL) async throws -> String {
        let endpoint = "\(baseURL)/api/process-audio"
        
        var request = URLRequest(url: URL(string: endpoint)!)
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        
        let data = try createMultipartBody(
            fileURL: fileURL,
            boundary: boundary
        )
        
        let (responseData, _) = try await URLSession.shared.upload(
            for: request,
            from: data
        )
        
        let response = try JSONDecoder().decode(
            JobResponse.self,
            from: responseData
        )
        
        return response.jobId
    }
    
    // Check processing status
    func checkStatus(jobId: String) async throws -> JobStatus {
        let endpoint = "\(baseURL)/api/job-status/\(jobId)"
        let url = URL(string: endpoint)!
        
        let (data, _) = try await URLSession.shared.data(from: url)
        let status = try JSONDecoder().decode(JobStatus.self, from: data)
        
        return status
    }
    
    // Get processed sentences
    func getSentences(jobId: String) async throws -> [Sentence] {
        let endpoint = "\(baseURL)/api/sentences/\(jobId)"
        let url = URL(string: endpoint)!
        
        let (data, _) = try await URLSession.shared.data(from: url)
        let sentences = try JSONDecoder().decode([Sentence].self, from: data)
        
        return sentences
    }
}

// Models
struct JobResponse: Codable {
    let jobId: String
    let status: String
}

struct JobStatus: Codable {
    let status: String // "processing", "completed", "error"
    let progress: Double?
}

struct Sentence: Codable, Identifiable {
    let id: Int
    let text: String
    let startTime: Double
    let endTime: Double
    let duration: Double
    let audioFileUrl: String?
    
    enum CodingKeys: String, CodingKey {
        case id, text, duration
        case startTime = "start_time"
        case endTime = "end_time"
        case audioFileUrl = "audio_file_url"
    }
}
```

### 2. Content Upload View

```swift
// ContentUploadView.swift
import SwiftUI

struct ContentUploadView: View {
    @State private var selectedFile: URL?
    @State private var isProcessing = false
    @State private var uploadProgress: Double = 0
    @State private var jobId: String?
    
    let api = AudioProcessorAPI()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Upload Audio/Video Content")
                .font(.title)
            
            // File picker
            Button("Select File") {
                // Show file picker
            }
            
            if let file = selectedFile {
                Text("Selected: \(file.lastPathComponent)")
                
                Button("Process") {
                    Task {
                        await processFile(file)
                    }
                }
                .disabled(isProcessing)
            }
            
            if isProcessing {
                ProgressView(value: uploadProgress)
                    .progressViewStyle(.linear)
                
                Text("Processing...")
            }
            
            if let jobId = jobId {
                Text("Job ID: \(jobId)")
                    .font(.caption)
            }
        }
        .padding()
    }
    
    private func processFile(_ fileURL: URL) async {
        isProcessing = true
        
        do {
            // Upload file
            let jobId = try await api.uploadAudio(fileURL: fileURL)
            self.jobId = jobId
            
            // Poll for status
            await pollStatus(jobId: jobId)
            
        } catch {
            print("Error: \(error)")
        }
        
        isProcessing = false
    }
    
    private func pollStatus(jobId: String) async {
        while true {
            do {
                let status = try await api.checkStatus(jobId: jobId)
                
                if let progress = status.progress {
                    uploadProgress = progress
                }
                
                if status.status == "completed" {
                    // Get sentences
                    let sentences = try await api.getSentences(jobId: jobId)
                    // Navigate to sentence list
                    break
                } else if status.status == "error" {
                    // Show error
                    break
                }
                
                // Wait before next poll
                try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
                
            } catch {
                print("Polling error: \(error)")
                break
            }
        }
    }
}
```

### 3. Lesson Practice View

```swift
// LessonPracticeView.swift
import SwiftUI
import AVFoundation

struct LessonPracticeView: View {
    let sentences: [Sentence]
    @State private var currentIndex = 0
    @State private var userInput = ""
    @State private var audioPlayer: AVPlayer?
    
    var currentSentence: Sentence {
        sentences[currentIndex]
    }
    
    var body: some View {
        VStack(spacing: 30) {
            // Progress
            ProgressView(
                value: Double(currentIndex + 1),
                total: Double(sentences.count)
            )
            
            Text("Sentence \(currentIndex + 1) of \(sentences.count)")
                .font(.caption)
            
            // Audio player controls
            HStack {
                Button("Play") {
                    playAudio()
                }
                
                Button("Slow (0.75x)") {
                    playAudio(rate: 0.75)
                }
            }
            
            // Dictation mode
            VStack {
                Text("Type what you hear:")
                    .font(.headline)
                
                TextField("Your answer", text: $userInput)
                    .textFieldStyle(.roundedBorder)
                    .autocapitalization(.none)
                
                Button("Check") {
                    checkAnswer()
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            
            // Navigation
            HStack {
                Button("Previous") {
                    if currentIndex > 0 {
                        currentIndex -= 1
                    }
                }
                .disabled(currentIndex == 0)
                
                Spacer()
                
                Button("Next") {
                    if currentIndex < sentences.count - 1 {
                        currentIndex += 1
                        userInput = ""
                    }
                }
                .disabled(currentIndex == sentences.count - 1)
            }
        }
        .padding()
    }
    
    private func playAudio(rate: Float = 1.0) {
        guard let urlString = currentSentence.audioFileUrl,
              let url = URL(string: urlString) else {
            return
        }
        
        audioPlayer = AVPlayer(url: url)
        audioPlayer?.rate = rate
        audioPlayer?.play()
    }
    
    private func checkAnswer() {
        let correct = userInput.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let expected = currentSentence.text.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Calculate similarity score
        let score = calculateSimilarity(correct, expected)
        
        // Show feedback
        // Save progress to database
    }
    
    private func calculateSimilarity(_ s1: String, _ s2: String) -> Double {
        // Implement Levenshtein distance or similar
        return 0.0
    }
}
```

## Workflow Tổng hợp

### 1. Content Creator (Admin) Flow

```
1. Admin upload audio/video qua web dashboard
   ↓
2. Backend xử lý với Audio Processor
   ↓
3. Tạo các file audio nhỏ cho từng câu
   ↓
4. Lưu transcription vào database
   ↓
5. Upload audio files lên cloud storage (S3/Cloudinary)
   ↓
6. Cập nhật URLs vào database
```

### 2. Student (User) Flow

```
1. User chọn topic/section/lesson trong app
   ↓
2. App fetch sentences từ API
   ↓
3. User practice dictation/shadowing
   ↓
4. App record progress & scores
   ↓
5. Sync progress với backend
```

## Cloud Storage Integration

### Sử dụng Cloudinary (khuyến nghị)

```python
# cloudinary_upload.py
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name="your_cloud_name",
    api_key="your_api_key",
    api_secret="your_api_secret"
)

def upload_audio_file(file_path: str, lesson_id: int, sentence_id: int) -> str:
    """Upload audio file to Cloudinary"""
    result = cloudinary.uploader.upload(
        file_path,
        resource_type="video",  # Audio files use "video" type
        folder=f"lessons/{lesson_id}",
        public_id=f"sentence_{sentence_id}",
        format="mp3"
    )
    return result["secure_url"]
```

## Performance Optimization

### 1. Cache Whisper Models
```python
# Preload model khi server start
@app.on_event("startup")
async def startup_event():
    global processor
    processor = AudioVideoProcessor(model_size="base")
    # Model đã được load sẵn
```

### 2. Use Task Queue (Celery)
```python
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def process_audio_task(job_id: str, audio_path: str):
    # Xử lý audio ở background worker
    pass
```

### 3. Batch Processing
```python
# Process multiple files cùng lúc
@app.post("/api/batch-process")
async def batch_process(files: List[UploadFile]):
    job_ids = []
    for file in files:
        job_id = create_processing_job(file)
        job_ids.append(job_id)
    return {"job_ids": job_ids}
```

## Testing

### Unit Test Example
```python
# test_audio_processor.py
import pytest
from audio_video_processor import AudioVideoProcessor

def test_sentence_detection():
    processor = AudioVideoProcessor()
    # Test với sample audio
    result = processor.process("test_audio.mp3")
    assert result["total_sentences"] > 0
    assert len(result["audio_files"]) == result["total_sentences"]
```

## Deployment Checklist

- [ ] FFmpeg installed trên Railway
- [ ] Environment variables configured
- [ ] Database migrations run
- [ ] Cloudinary/S3 setup
- [ ] API authentication implemented
- [ ] Rate limiting configured
- [ ] Error monitoring (Sentry)
- [ ] Logs aggregation
- [ ] Backup strategy
- [ ] iOS app pointing to production API

## Next Steps

1. **Phase 1**: Setup backend API với basic processing
2. **Phase 2**: Integrate với iOS app (upload flow)
3. **Phase 3**: Add practice modes (dictation/shadowing)
4. **Phase 4**: Implement progress tracking
5. **Phase 5**: Add gamification & scoring

---

📧 Questions? Contact: your@email.com
