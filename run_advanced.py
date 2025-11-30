"""
CLI Wrapper cho Advanced Audio/Video Processor
Cho phép chạy advanced_processor.py từ terminal với arguments

Usage:
    python run_advanced.py audio.mp3
    python run_advanced.py audio.mp3 --vad --normalize
    python run_advanced.py audio.mp3 --model small --device cuda --vad
    python run_advanced.py audio_folder --batch --model small --device cuda
"""

import argparse
import sys
from pathlib import Path
from advanced_processor import AdvancedAudioProcessor, batch_process_folder


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description='Advanced Audio/Video Processor with Voice Activity Detection (VAD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic processing
  python run_advanced.py audio.mp3
  
  # With VAD and normalization
  python run_advanced.py audio.mp3 --vad --normalize
  
  # All features
  python run_advanced.py audio.mp3 --model small --device cuda --vad -n --formats json srt txt csv
  
  # Video file
  python run_advanced.py video.mp4 --video --vad
  
  # YouTube video
  python run_advanced.py "https://youtube.com/watch?v=xxx" --youtube --vad
  
  # Batch processing
  python run_advanced.py audio_folder --batch --model small --device cuda
        """
    )
    
    # Positional argument
    parser.add_argument(
        'input',
        help='Path to audio/video file, YouTube URL, or folder (for batch)'
    )
    
    # Model and device options
    parser.add_argument(
        '--model', '-m',
        default='small',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: small)'
    )
    
    parser.add_argument(
        '--device', '-d',
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to use: cuda/cpu (default: auto-detect)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='advanced_output',
        help='Output directory (default: advanced_output)'
    )
    
    # Advanced features
    parser.add_argument(
        '--vad',
        action='store_true',
        help='Enable Voice Activity Detection (VAD)'
    )
    
    parser.add_argument(
        '--normalize', '-n',
        action='store_true',
        help='Normalize audio volume to -20 dBFS'
    )
    
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        default=['json', 'srt'],
        choices=['json', 'srt', 'txt', 'csv'],
        help='Export formats (default: json srt)'
    )
    
    # Input type flags
    parser.add_argument(
        '--video', '-v',
        action='store_true',
        help='Input is a video file (will extract audio)'
    )
    
    parser.add_argument(
        '--youtube', '-y',
        action='store_true',
        help='Input is a YouTube URL'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Batch process all audio files in folder'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=['.mp3', '.wav', '.m4a'],
        help='File extensions for batch processing (default: .mp3 .wav .m4a)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("ADVANCED AUDIO/VIDEO PROCESSOR WITH VAD")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device if args.device else 'auto-detect'}")
    print(f"Output: {args.output}")
    print(f"VAD: {'Enabled' if args.vad else 'Disabled'}")
    print(f"Normalize: {'Yes' if args.normalize else 'No'}")
    print(f"Formats: {', '.join(args.formats)}")
    print("="*60)
    
    try:
        # Batch processing mode
        if args.batch:
            print(f"\n🔄 Batch processing folder: {args.input}")
            
            if not Path(args.input).is_dir():
                print(f"❌ Error: {args.input} is not a directory")
                sys.exit(1)
            
            batch_process_folder(
                folder_path=args.input,
                output_base=args.output,
                file_extensions=args.extensions,
                model_size=args.model,
                device=args.device
            )
            
        else:
            # Single file processing
            print(f"\n🎵 Processing single file/URL")
            
            # Create processor
            processor = AdvancedAudioProcessor(
                output_dir=args.output,
                model_size=args.model,
                device=args.device
            )
            
            # Process
            result = processor.process_advanced(
                args.input,
                is_youtube=args.youtube,
                is_video=args.video,
                use_vad=args.vad,
                normalize=args.normalize,
                export_formats=args.formats
            )
            
            # Print results
            print("\n" + "="*60)
            print("✅ PROCESSING COMPLETE!")
            print("="*60)
            
            stats = result['statistics']
            print(f"\n📊 Statistics:")
            print(f"   Total sentences: {stats['total_sentences']}")
            print(f"   Total duration: {stats['total_duration']:.2f}s")
            print(f"   Average sentence: {stats['avg_sentence_duration']:.2f}s")
            print(f"   Total words: {stats['total_words']}")
            
            print(f"\n📁 Export files:")
            for format_type, filepath in result['export_files'].items():
                print(f"   {format_type.upper()}: {filepath}")
            
            print(f"\n🎵 Audio files: {len(result['audio_files'])} sentence files")
            print(f"   Location: {args.output}/sentences/")
            
            print("\n✨ Done! Check the output directory for results.")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found - {e}")
        sys.exit(1)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"\n❌ GPU out of memory!")
            print(f"💡 Try:")
            print(f"   1. Smaller model: --model base")
            print(f"   2. Use CPU: --device cpu")
        else:
            print(f"\n❌ Runtime error: {e}")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Processing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()