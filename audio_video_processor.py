"""
Audio/Video Processor with Sentence Segmentation and Transcription
LOCAL MODEL VERSION - Models stored in project folder
Supports: Audio files, Video files, YouTube links
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import subprocess
import sys
import uuid
# ============================================================================
# FFMPEG PATH CONFIGURATION - IMPORTANT FOR WINDOWS
# ============================================================================
# If FFmpeg is not in PATH, set the path manually here:
FFMPEG_PATH = None  # Example: r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = None  # Example: r"C:\ffmpeg\bin\ffprobe.exe"

# Auto-detect FFmpeg location
if FFMPEG_PATH:
    os.environ["PATH"] = os.path.dirname(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")

try:
    import whisper
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    import yt_dlp
    import torch
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install: pip install openai-whisper pydub yt-dlp torch")
    print("Also install ffmpeg: brew install ffmpeg (Mac) or apt-get install ffmpeg (Linux)")
    exit(1)

# Set FFmpeg location for pydub if specified
if FFMPEG_PATH:
    AudioSegment.converter = FFMPEG_PATH
if FFPROBE_PATH:
    AudioSegment.ffprobe = FFPROBE_PATH


class AudioVideoProcessor:
    """Processor for audio/video with sentence segmentation and transcription"""
    
    def __init__(self, output_dir: str = "output", model_size: str = "base", 
                 device: str = None, models_dir: str = "models"):
        """
        Initialize processor
        
        Args:
            output_dir: Directory to save output files
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            models_dir: Directory to store Whisper models (relative to project)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ====================================================================
        # LOCAL MODEL STORAGE - Store in project folder
        # ====================================================================
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print("="*70)
        print(f"INITIALIZING AUDIO/VIDEO PROCESSOR (LOCAL MODELS)")
        print("="*70)
        
        # Model info
        model_info = {
            'tiny': {'size_mb': 39, 'vram_gb': 1},
            'base': {'size_mb': 74, 'vram_gb': 1},
            'small': {'size_mb': 244, 'vram_gb': 2},
            'medium': {'size_mb': 769, 'vram_gb': 5},
            'large': {'size_mb': 1550, 'vram_gb': 10}
        }
        info = model_info.get(model_size, model_info['base'])
        
        # Check if model exists locally
        model_file = self.models_dir / f"{model_size}.pt"
        model_exists = model_file.exists()
        
        print(f"\n📋 Configuration:")
        print(f"   Model: {model_size}")
        print(f"   Size: ~{info['size_mb']} MB")
        print(f"   VRAM: ~{info['vram_gb']} GB")
        print(f"   Device: {self.device.upper()}")
        
        if self.device == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        print(f"\n📁 Local Model Storage:")
        print(f"   Directory: {self.models_dir.absolute()}")
        
        if model_exists:
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"   Status: ✅ Model EXISTS in project")
            print(f"   File: {model_file.name}")
            print(f"   Size: {size_mb:.1f} MB")
            print(f"   ⚡ Loading from local storage...")
        else:
            print(f"   Status: ❌ Model NOT in project yet")
            print(f"   📥 Will download to: {model_file.absolute()}")
            print(f"   💾 Size: ~{info['size_mb']} MB")
            print(f"   ⏳ This will take 30-60 seconds...")
            print(f"   💡 Model will be saved in YOUR PROJECT for reuse!")
        
        print(f"\n🔄 Loading Whisper model...")
        
        # Load model with custom download location
        try:
            # IMPORTANT: download_root tells Whisper where to save/load models
            self.model = whisper.load_model(
                model_size, 
                device=self.device,
                download_root=str(self.models_dir)  # Save in project folder!
            )
            
            print(f"✅ Model loaded successfully!")
            
            # Confirm location if just downloaded
            if not model_exists and model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"💾 Model saved to project:")
                print(f"   {model_file.absolute()}")
                print(f"   Size: {size_mb:.1f} MB")
            
            print("="*70)
            print("✨ PROCESSOR READY! (Using LOCAL models)")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR loading model: {e}")
            print(f"💡 Troubleshooting:")
            print(f"   1. Check internet connection")
            print(f"   2. Check disk space (~{info['size_mb']} MB needed)")
            print(f"   3. Check write permissions for: {self.models_dir.absolute()}")
            print(f"   4. Try smaller model: model_size='tiny' or 'base'")
            raise
        
    def download_youtube(self, url: str) -> str:
        """
        Download audio from YouTube video
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded audio file
        """
        output_path = self.output_dir / "youtube_audio.mp3"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.output_dir / 'youtube_audio'),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        return str(output_path)
    
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        audio_path = self.output_dir / "extracted_audio.wav"
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # Audio codec
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite
            str(audio_path)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return str(audio_path)
    
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def transcribe_full(self, audio_path: str) -> Dict:
        """
        Transcribe entire audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result with timestamps
        """
        print(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            language='en'  # Change to 'vi' for Vietnamese
        )
        return result
    
    def detect_sentence_boundaries(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect sentence boundaries from Whisper segments
        
        Args:
            segments: Whisper segments with word timestamps
            
        Returns:
            List of sentences with start/end times
        """
        sentences = []
        current_sentence = {
            'text': '',
            'start': None,
            'end': None,
            'words': []
        }
        
        # Sentence ending patterns
        sentence_endings = r'[.!?।။።]'
        
        for segment in segments:
            if 'words' in segment:
                for word_info in segment['words']:
                    word = word_info['word'].strip()
                    
                    if current_sentence['start'] is None:
                        current_sentence['start'] = word_info['start']
                    
                    current_sentence['text'] += ' ' + word
                    current_sentence['words'].append(word_info)
                    current_sentence['end'] = word_info['end']
                    
                    # Check if sentence ends
                    if re.search(sentence_endings, word):
                        sentences.append(current_sentence.copy())
                        current_sentence = {
                            'text': '',
                            'start': None,
                            'end': None,
                            'words': []
                        }
        
        # Add remaining text as last sentence
        if current_sentence['text'].strip():
            sentences.append(current_sentence)
        
        return sentences
    
    def split_audio_by_timestamps(self, audio_path: str, sentences: List[Dict], 
                                  padding_ms: int = 100) -> List[str]:
        """
        Split audio file into separate files based on sentence timestamps
        
        Args:
            audio_path: Path to source audio file
            sentences: List of sentences with timestamps
            padding_ms: Padding to add before/after each segment (ms)
            
        Returns:
            List of paths to split audio files
        """
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Create output directory for sentences
        sentences_dir = self.output_dir / "sentences"
        sentences_dir.mkdir(exist_ok=True)
        
        output_files = []
        
        for i, sentence in enumerate(sentences, 1):
            start_ms = max(0, int(sentence['start'] * 1000) - padding_ms)
            end_ms = min(len(audio), int(sentence['end'] * 1000) + padding_ms)
            
            # Extract segment
            segment = audio[start_ms:end_ms]
            
            # Save
            output_file = sentences_dir / f"sentence_{i:03d}.wav"
            segment.export(str(output_file), format="wav")
            output_files.append(str(output_file))
        
        return output_files
    
    def save_transcriptions(self, sentences: List[Dict]) -> str:
        """
        Save transcriptions to JSON file
        
        Args:
            sentences: List of sentences with timestamps
            
        Returns:
            Path to JSON file
        """
        output_file = self.output_dir / "transcriptions.json"
        
        # Format sentences
        formatted_sentences = []
        for i, sentence in enumerate(sentences, 1):
            formatted_sentences.append({
                'ID': str(uuid.uuid4()),
                "LessonId": "",
                "OrderIndex": i,
                'start_point': sentence['start'],
                'stop_point': sentence['end'],
                'script': sentence['text'].strip(),
                "translated_script": "",
                "transcription": "",
                "note": "",
                "media_url": ""
                # 'duration': sentence['end'] - sentence['start']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_sentences, f, indent=2, ensure_ascii=False)
        
        return str(output_file)
    
    def save_srt(self, sentences: List[Dict]) -> str:
        """
        Save transcriptions as SRT subtitle file
        
        Args:
            sentences: List of sentences with timestamps
            
        Returns:
            Path to SRT file
        """
        output_file = self.output_dir / "subtitles.srt"
        
        def format_timestamp(seconds: float) -> str:
            """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences, 1):
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(sentence['start'])} --> "
                       f"{format_timestamp(sentence['end'])}\n")
                f.write(f"{sentence['text'].strip()}\n\n")
        
        return str(output_file)
    
    def process(self, input_path: str, is_youtube: bool = False, 
                is_video: bool = False) -> Dict:
        """
        Process audio/video file with full pipeline
        
        Args:
            input_path: Path to audio/video file or YouTube URL
            is_youtube: Whether input is YouTube URL
            is_video: Whether input is video file (extract audio)
            
        Returns:
            Dict with processing results
        """
        print("="*60)
        print("PROCESSING")
        print("="*60)
        
        # Get audio
        if is_youtube:
            audio_path = self.download_youtube(input_path)
        elif is_video:
            audio_path = self.extract_audio_from_video(input_path)
        else:
            audio_path = input_path
        
        # Transcribe
        transcription = self.transcribe_full(audio_path)
        
        # Detect sentences
        sentences = self.detect_sentence_boundaries(transcription['segments'])
        
        print(f"\nDetected {len(sentences)} sentences")
        
        # Split audio
        audio_files = self.split_audio_by_timestamps(audio_path, sentences)
        
        # Save transcriptions
        json_file = self.save_transcriptions(sentences)
        srt_file = self.save_srt(sentences)
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE!")
        print("="*60)
        print(f"Total sentences: {len(sentences)}")
        print(f"Audio files: {len(audio_files)}")
        print(f"JSON: {json_file}")
        print(f"SRT: {srt_file}")
        print("="*60 + "\n")
        
        return {
            'sentences': sentences,
            'audio_files': audio_files,
            'transcription_json': json_file,
            'transcription_srt': srt_file,
            'total_sentences': len(sentences)
        }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Audio/Video Processor with Whisper (LOCAL models in project)'
    )
    
    parser.add_argument('input', help='Input audio/video file or YouTube URL')
    parser.add_argument('--model', '-m', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size (default: base)')
    parser.add_argument('--device', '-d', default=None,
                       choices=['cuda', 'cpu'],
                       help='Device: cuda/cpu (default: auto-detect)')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--models-dir', default='models',
                       help='Directory for Whisper models (default: models)')
    parser.add_argument('--youtube', '-y', action='store_true',
                       help='Input is YouTube URL')
    parser.add_argument('--video', '-v', action='store_true',
                       help='Input is video file (extract audio)')
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioVideoProcessor(
        output_dir=args.output,
        model_size=args.model,
        device=args.device,
        models_dir=args.models_dir
    )
    
    # Process
    result = processor.process(
        args.input,
        is_youtube=args.youtube,
        is_video=args.video
    )