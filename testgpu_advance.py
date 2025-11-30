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
result_cpu = cpu_processor.process_advanced("actors.mp3")
cpu_time = time.time() - start

# Test GPU
print("\n🚀 Testing GPU...")
start = time.time()
gpu_processor = AdvancedAudioProcessor(
    model_size="small",
    device="cuda"
)
result_gpu = gpu_processor.process_advanced("actors.mp3")
gpu_time = time.time() - start

# Compare
print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"CPU time: {cpu_time:.2f}s")
print(f"GPU time: {gpu_time:.2f}s")
print(f"Speed up: {cpu_time/gpu_time:.1f}x faster with GPU")
print(f"Time saved: {cpu_time - gpu_time:.2f}s")