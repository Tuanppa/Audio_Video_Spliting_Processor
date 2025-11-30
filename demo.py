"""
Demo script đơn giản để test Audio/Video Processor
"""

from audio_video_processor import AudioVideoProcessor


def demo_audio_file():
    """Demo xử lý audio file"""
    print("\n" + "="*60)
    print("DEMO: Xử lý Audio File")
    print("="*60)
    
    processor = AudioVideoProcessor(
        output_dir="demo_output",
        model_size="base"
    )
    
    # Thay đổi đường dẫn file của bạn ở đây
    audio_file = "your_audio.mp3"  # Thay bằng file audio của bạn
    
    try:
        result = processor.process(audio_file)
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Tổng số câu: {result['total_sentences']}")
        print(f"🎵 Files audio: {len(result['audio_files'])}")
        print(f"📝 Transcription: {result['transcription_json']}")
        
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {audio_file}")
        print("Vui lòng thay đổi đường dẫn file trong demo.py")


def demo_video_file():
    """Demo xử lý video file"""
    print("\n" + "="*60)
    print("DEMO: Xử lý Video File")
    print("="*60)
    
    processor = AudioVideoProcessor(
        output_dir="demo_output",
        model_size="base"
    )
    
    # Thay đổi đường dẫn file của bạn ở đây
    video_file = "your_video.mp4"  # Thay bằng file video của bạn
    
    try:
        result = processor.process(video_file, is_video=True)
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Tổng số câu: {result['total_sentences']}")
        print(f"🎵 Files audio: {len(result['audio_files'])}")
        print(f"📝 Transcription: {result['transcription_json']}")
        
    except FileNotFoundError:
        print(f"\n❌ Không tìm thấy file: {video_file}")
        print("Vui lòng thay đổi đường dẫn file trong demo.py")


def demo_youtube():
    """Demo xử lý YouTube video"""
    print("\n" + "="*60)
    print("DEMO: Xử lý YouTube Video")
    print("="*60)
    
    processor = AudioVideoProcessor(
        output_dir="demo_output",
        model_size="base"
    )
    
    # Thay đổi YouTube URL của bạn ở đây
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        result = processor.process(youtube_url, is_youtube=True)
        print(f"\n✅ Hoàn thành!")
        print(f"📊 Tổng số câu: {result['total_sentences']}")
        print(f"🎵 Files audio: {len(result['audio_files'])}")
        print(f"📝 Transcription: {result['transcription_json']}")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("Vui lòng kiểm tra URL YouTube và kết nối internet")


def interactive_demo():
    """Demo tương tác"""
    print("\n" + "="*60)
    print("AUDIO/VIDEO PROCESSOR - INTERACTIVE DEMO")
    print("="*60)
    
    print("\nChọn loại input:")
    print("1. Audio file (MP3, WAV, etc.)")
    print("2. Video file (MP4, AVI, MOV, etc.)")
    print("3. YouTube URL")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn của bạn (0-3): ").strip()
    
    if choice == "0":
        print("Tạm biệt!")
        return
    
    if choice == "1":
        path = input("Nhập đường dẫn audio file: ").strip()
        processor = AudioVideoProcessor(output_dir="demo_output", model_size="base")
        try:
            result = processor.process(path)
            print_result(result)
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    elif choice == "2":
        path = input("Nhập đường dẫn video file: ").strip()
        processor = AudioVideoProcessor(output_dir="demo_output", model_size="base")
        try:
            result = processor.process(path, is_video=True)
            print_result(result)
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    elif choice == "3":
        url = input("Nhập YouTube URL: ").strip()
        processor = AudioVideoProcessor(output_dir="demo_output", model_size="base")
        try:
            result = processor.process(url, is_youtube=True)
            print_result(result)
        except Exception as e:
            print(f"❌ Lỗi: {e}")
    
    else:
        print("❌ Lựa chọn không hợp lệ!")


def print_result(result):
    """In kết quả"""
    print("\n" + "="*60)
    print("✅ XỬ LÝ HOÀN THÀNH!")
    print("="*60)
    print(f"\n📊 Thống kê:")
    print(f"   - Tổng số câu: {result['total_sentences']}")
    print(f"   - Files audio tạo ra: {len(result['audio_files'])}")
    print(f"\n📁 Files kết quả:")
    print(f"   - JSON: {result['transcription_json']}")
    print(f"   - SRT: {result['transcription_srt']}")
    print(f"\n💡 Xem chi tiết trong thư mục: demo_output/")
    
    # In preview 3 câu đầu
    import json
    with open(result['transcription_json'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📝 Preview 3 câu đầu:")
    for item in data[:3]:
        print(f"\n   Câu {item['id']}: [{item['start_time']}s - {item['end_time']}s]")
        print(f"   '{item['text']}'")


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║   AUDIO/VIDEO PROCESSOR - DEMO                            ║
║   Tách câu và Transcription tự động                       ║
╚═══════════════════════════════════════════════════════════╝

Chọn demo bạn muốn chạy:
    """)
    
    print("1. Demo Audio File")
    print("2. Demo Video File")
    print("3. Demo YouTube")
    print("4. Interactive Mode (khuyến nghị)")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn (0-4): ").strip()
    
    if choice == "1":
        demo_audio_file()
    elif choice == "2":
        demo_video_file()
    elif choice == "3":
        demo_youtube()
    elif choice == "4":
        interactive_demo()
    elif choice == "0":
        print("Tạm biệt!")
    else:
        print("❌ Lựa chọn không hợp lệ!")