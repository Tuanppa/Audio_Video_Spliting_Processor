"""
GPU Availability Test Script
Kiểm tra GPU có sẵn sàng cho Whisper không
Test với file audio_video_processor.py
"""

import sys

print("="*60)
print("GPU AVAILABILITY CHECK FOR WHISPER")
print("="*60)

# Check PyTorch
try:
    import torch
    print("\n✅ PyTorch installed")
    print(f"   Version: {torch.__version__}")
except ImportError:
    print("\n❌ PyTorch not installed!")
    print("   Install: pip install torch torchaudio")
    sys.exit(1)

# Check CUDA availability
print(f"\n🔍 CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    
    # GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n📊 GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"   Compute capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\n🧪 Testing GPU computation...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("   ✅ GPU computation works!")
    except Exception as e:
        print(f"   ❌ GPU computation failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 GPU IS READY TO USE!")
    print("="*60)
    print("\nRun: python audio_video_processor.py audio.mp3")
    
else:
    print("\n" + "="*60)
    print("❌ CUDA NOT AVAILABLE - WILL USE CPU")
    print("="*60)
    
    print("\n🔧 To enable GPU:")
    print("1. Check: nvidia-smi")
    print("2. Install: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("3. Test: python test_gpu.py")

print("\n" + "="*60)