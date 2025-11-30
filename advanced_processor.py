"""
Advanced Audio/Video Processor với Voice Activity Detection (VAD)
LOCAL MODEL VERSION - Models stored in project folder
Tính năng bổ sung:
- VAD để detect silence và tách câu chính xác hơn
- Audio normalization
- Xuất nhiều định dạng (JSON, SRT, TXT, CSV)
- Batch processing
- Local model storage in project
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from pydub.silence import detect_nonsilent, split_on_silence
from pydub.effects import normalize as pydub_normalize
import numpy as np
from audio_video_processor_local import AudioVideoProcessor


class AdvancedAudioProcessor(AudioVideoProcessor):
    """Extended processor với các tính năng nâng cao và local model storage"""
    
    def __init__(self, output_dir: str = "output", model_size: str = "base", 
                 device: str = None, models_dir: str = "models"):
        """
        Initialize advanced processor with local model storage
        
        Args:
            output_dir: Directory to save output files
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            models_dir: Directory to store Whisper models in project
        """
        super().__init__(output_dir, model_size, device, models_dir)
        
    def detect_silence_segments(self, audio_path: str, 
                               min_silence_len: int = 500,
                               silence_thresh: int = -40) -> List[tuple]:
        """
        Detect các đoạn có giọng nói (không phải silence)
        
        Args:
            audio_path: Path to audio file
            min_silence_len: Độ dài tối thiểu của silence (ms)
            silence_thresh: Ngưỡng âm lượng coi là silence (dBFS)
            
        Returns:
            List of (start_ms, end_ms) tuples
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Detect non-silent chunks
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        return nonsilent_ranges
    
    def split_on_silence_advanced(self, audio_path: str,
                                  min_silence_len: int = 700,
                                  silence_thresh: int = -40,
                                  keep_silence: int = 200) -> List[str]:
        """
        Tách audio dựa trên silence detection
        
        Args:
            audio_path: Path to audio file
            min_silence_len: Độ dài tối thiểu của khoảng lặng (ms)
            silence_thresh: Ngưỡng âm lượng coi là silence (dBFS)
            keep_silence: Giữ lại bao nhiêu ms silence ở đầu/cuối
            
        Returns:
            List of paths to split audio files
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Split on silence
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        
        # Save chunks
        output_files = []
        chunks_dir = self.output_dir / "vad_chunks"
        chunks_dir.mkdir(exist_ok=True)
        
        print(f"\nDetected {len(chunks)} voice segments")
        
        for i, chunk in enumerate(chunks, 1):
            output_file = chunks_dir / f"chunk_{i:03d}.wav"
            chunk.export(str(output_file), format="wav")
            output_files.append(str(output_file))
            print(f"  Chunk {i}: {len(chunk)/1000:.2f}s")
        
        return output_files
    
    def normalize_audio(self, audio_path: str, target_dBFS: float = -20.0) -> str:
        """
        Normalize audio để có âm lượng đồng nhất
        
        Args:
            audio_path: Path to audio file
            target_dBFS: Target loudness level
            
        Returns:
            Path to normalized audio
        """
        audio = AudioSegment.from_file(audio_path)
        
        # Calculate change in dBFS
        change_in_dBFS = target_dBFS - audio.dBFS
        
        # Apply gain
        normalized_audio = audio.apply_gain(change_in_dBFS)
        
        # Save
        output_path = self.output_dir / "normalized_audio.wav"
        normalized_audio.export(str(output_path), format="wav")
        
        print(f"Audio normalized: {audio.dBFS:.1f} dBFS → {target_dBFS:.1f} dBFS")
        
        return str(output_path)
    
    def export_to_txt(self, sentences: List[Dict]) -> str:
        """
        Export transcriptions to plain text file
        
        Args:
            sentences: List of sentences with timestamps
            
        Returns:
            Path to TXT file
        """
        output_file = self.output_dir / "transcriptions.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences, 1):
                f.write(f"[{i}] {sentence['text'].strip()}\n")
        
        return str(output_file)
    
    def export_to_csv(self, sentences: List[Dict]) -> str:
        """
        Export transcriptions to CSV file
        
        Args:
            sentences: List of sentences with timestamps
            
        Returns:
            Path to CSV file
        """
        output_file = self.output_dir / "transcriptions.csv"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header
            f.write("id,text,start_time,end_time,duration\n")
            
            # Data
            for i, sentence in enumerate(sentences, 1):
                text = sentence['text'].strip().replace('"', '""')  # Escape quotes
                f.write(f'{i},"{text}",{sentence["start"]:.2f},'
                       f'{sentence["end"]:.2f},'
                       f'{sentence["end"] - sentence["start"]:.2f}\n')
        
        return str(output_file)
    
    def get_statistics(self, sentences: List[Dict]) -> Dict:
        """
        Calculate statistics about the transcription
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dict with statistics
        """
        if not sentences:
            return {
                'total_sentences': 0,
                'total_duration': 0,
                'total_words': 0
            }
        
        durations = [s['end'] - s['start'] for s in sentences]
        word_counts = [len(s['text'].split()) for s in sentences]
        
        stats = {
            'total_sentences': len(sentences),
            'total_duration': sum(durations),
            'average_sentence_duration': np.mean(durations),
            'min_sentence_duration': min(durations),
            'max_sentence_duration': max(durations),
            'total_words': sum(word_counts),
            'average_words_per_sentence': np.mean(word_counts),
            'min_words': min(word_counts),
            'max_words': max(word_counts)
        }
        
        return stats
    
    def process_advanced(self, input_path: str, 
                        use_vad: bool = False,
                        normalize: bool = False,
                        vad_params: Dict = None,
                        export_formats: List[str] = None) -> Dict:
        """
        Process với các tính năng nâng cao
        
        Args:
            input_path: Path to audio file
            use_vad: Use VAD to detect voice segments
            normalize: Normalize audio volume
            vad_params: Parameters for VAD (min_silence_len, silence_thresh, keep_silence)
            export_formats: List of formats to export ('json', 'srt', 'txt', 'csv')
            
        Returns:
            Dict with processing results
        """
        print("="*70)
        print("ADVANCED PROCESSING (LOCAL MODELS)")
        print("="*70)
        
        audio_path = input_path
        
        # Step 1: Normalize if requested
        if normalize:
            print("\n[1/4] Normalizing audio...")
            audio_path = self.normalize_audio(audio_path)
        else:
            print("\n[1/4] Skipping normalization")
        
        # Step 2: VAD if requested
        vad_chunks = None
        if use_vad:
            print("\n[2/4] Applying Voice Activity Detection...")
            params = vad_params or {}
            vad_chunks = self.split_on_silence_advanced(
                audio_path,
                min_silence_len=params.get('min_silence_len', 700),
                silence_thresh=params.get('silence_thresh', -40),
                keep_silence=params.get('keep_silence', 200)
            )
        else:
            print("\n[2/4] Skipping VAD")
        
        # Step 3: Transcribe
        print("\n[3/4] Transcribing with Whisper...")
        transcription = self.transcribe_full(audio_path)
        
        # Detect sentences
        sentences = self.detect_sentence_boundaries(transcription['segments'])
        print(f"Detected {len(sentences)} sentences")
        
        # Step 4: Export
        print("\n[4/4] Exporting results...")
        formats = export_formats or ['json', 'srt']
        export_files = {}
        
        if 'json' in formats:
            export_files['json'] = self.save_transcriptions(sentences)
            print(f"  ✅ JSON: {export_files['json']}")
        
        if 'srt' in formats:
            export_files['srt'] = self.save_srt(sentences)
            print(f"  ✅ SRT: {export_files['srt']}")
        
        if 'txt' in formats:
            export_files['txt'] = self.export_to_txt(sentences)
            print(f"  ✅ TXT: {export_files['txt']}")
        
        if 'csv' in formats:
            export_files['csv'] = self.export_to_csv(sentences)
            print(f"  ✅ CSV: {export_files['csv']}")
        
        # Statistics
        stats = self.get_statistics(sentences)
        
        print("\n" + "="*70)
        print("✅ ADVANCED PROCESSING COMPLETE!")
        print("="*70)
        print(f"📊 Statistics:")
        print(f"   Sentences: {stats['total_sentences']}")
        print(f"   Total duration: {stats['total_duration']:.1f}s")
        print(f"   Avg sentence: {stats['average_sentence_duration']:.2f}s")
        print(f"   Total words: {stats['total_words']}")
        print(f"   Avg words/sentence: {stats['average_words_per_sentence']:.1f}")
        print("="*70 + "\n")
        
        return {
            'sentences': sentences,
            'vad_chunks': vad_chunks,
            'export_files': export_files,
            'statistics': stats,
            'normalized_audio': audio_path if normalize else None
        }


def batch_process_folder(input_folder: str, output_base: str = "batch_output",
                        model_size: str = "base", device: str = None,
                        models_dir: str = "models",
                        use_vad: bool = False, normalize: bool = False,
                        extensions: List[str] = None):
    """
    Process tất cả audio files trong folder
    
    Args:
        input_folder: Folder chứa audio files
        output_base: Base directory cho outputs
        model_size: Whisper model size
        device: cuda/cpu
        models_dir: Directory for Whisper models
        use_vad: Use VAD
        normalize: Normalize audio
        extensions: File extensions to process (default: mp3, wav, m4a)
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        raise ValueError(f"Input folder not found: {input_folder}")
    
    # Get all audio files
    if extensions is None:
        extensions = ['mp3', 'wav', 'm4a', 'flac', 'ogg']
    
    audio_files = []
    for ext in extensions:
        audio_files.extend(input_path.glob(f"*.{ext}"))
        audio_files.extend(input_path.glob(f"*.{ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Create processor once (reuse for all files)
    processor = AdvancedAudioProcessor(
        model_size=model_size,
        device=device,
        models_dir=models_dir
    )
    
    results = []
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n{'='*70}")
        print(f"Processing [{i}/{len(audio_files)}]: {audio_file.name}")
        print(f"{'='*70}")
        
        # Create output directory for this file
        output_dir = Path(output_base) / audio_file.stem
        processor.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            result = processor.process_advanced(
                str(audio_file),
                use_vad=use_vad,
                normalize=normalize,
                export_formats=['json', 'srt', 'txt', 'csv']
            )
            results.append({
                'file': audio_file.name,
                'status': 'success',
                'sentences': result['statistics']['total_sentences'],
                'duration': result['statistics']['total_duration']
            })
        except Exception as e:
            print(f"❌ Error processing {audio_file.name}: {e}")
            results.append({
                'file': audio_file.name,
                'status': 'error',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    print(f"✅ Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(results)}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Advanced Audio Processor with VAD (LOCAL models)'
    )
    
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('--model', '-m', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large'],
                       help='Whisper model size')
    parser.add_argument('--device', '-d', default=None,
                       choices=['cuda', 'cpu'],
                       help='Device: cuda/cpu (auto-detect if not specified)')
    parser.add_argument('--models-dir', default='models',
                       help='Directory for Whisper models in project')
    parser.add_argument('--output', '-o', default='output',
                       help='Output directory')
    parser.add_argument('--vad', action='store_true',
                       help='Use Voice Activity Detection')
    parser.add_argument('--normalize', '-n', action='store_true',
                       help='Normalize audio volume')
    
    args = parser.parse_args()
    
    # Create processor
    processor = AdvancedAudioProcessor(
        output_dir=args.output,
        model_size=args.model,
        device=args.device,
        models_dir=args.models_dir
    )
    
    # Process
    result = processor.process_advanced(
        args.input,
        use_vad=args.vad,
        normalize=args.normalize,
        export_formats=['json', 'srt', 'txt', 'csv']
    )