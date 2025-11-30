# 🚀 GPU Setup Guide - Tăng tốc xử lý 5-10x

## 📊 CPU vs GPU Performance

| Model  | CPU Time | GPU Time | Speed Up |
|--------|----------|----------|----------|
| tiny   | 10s      | 2s       | 5x       |
| base   | 20s      | 3s       | 6.7x     |
| small  | 60s      | 8s       | 7.5x     |
| medium | 180s     | 20s      | 9x       |
| large  | 360s     | 35s      | 10x      |

*Thời gian xử lý 1 phút audio*

---

## ✅ BƯỚC 1: KIỂM TRA GPU

```powershell
nvidia-smi
```

**Kết quả mong muốn:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
```

**Nếu lỗi → Cài NVIDIA Driver:**
1. https://www.nvidia.com/Download/index.aspx
2. Chọn GPU của bạn
3. Download và cài
4. Restart máy

---

## 🔧 BƯỚC 2: CÀI PYTORCH VỚI CUDA

### Cách 1: Conda (Khuyến nghị)

```powershell
conda activate DicTool

# Uninstall CPU version
pip uninstall torch torchaudio -y

# Install GPU version
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Cách 2: Pip

```powershell
conda activate DicTool

pip uninstall torch torchaudio -y

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Hoặc CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🧪 BƯỚC 3: TEST GPU

Tạo file `test_gpu.py`:

```python
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("✅ GPU ready!")
else:
    print("❌ GPU not available")
```

Chạy:
```powershell
python test_gpu.py
```

---

## 🚀 BƯỚC 4: DÙNG GPU

### Auto-detect (Khuyến nghị)

```powershell
# Tự động dùng GPU nếu có
python audio_video_processor.py actors.mp3
```

Output sẽ hiện:
```
Loading Whisper model: base...
Using device: CUDA
GPU: NVIDIA GeForce RTX 3060
GPU Memory: 12.00 GB
Model loaded successfully!
```

### Force GPU hoặc CPU

```powershell
# Force GPU
python audio_video_processor.py actors.mp3 --device cuda

# Force CPU (để so sánh)
python audio_video_processor.py actors.mp3 --device cpu
```

---

## 💡 MEMORY REQUIREMENTS

| Model  | VRAM | Recommended GPU |
|--------|------|-----------------|
| tiny   | 1 GB | GTX 1050+       |
| base   | 1 GB | GTX 1050+       |
| small  | 2 GB | GTX 1060+       |
| medium | 5 GB | RTX 2060+       |
| large  | 10GB | RTX 3080+       |

---

## 🐛 TROUBLESHOOTING

### "CUDA not available"

```powershell
# Check driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "CUDA out of memory"

```powershell
# Use smaller model
python audio_video_processor.py audio.mp3 --model small

# Or use CPU
python audio_video_processor.py audio.mp3 --device cpu
```

---

## ✅ CHECKLIST

- [ ] `nvidia-smi` works
- [ ] PyTorch with CUDA installed
- [ ] `python test_gpu.py` shows GPU ready
- [ ] Audio processor shows "Using device: CUDA"

---

**Download files:**
- **[audio_video_processor.py](computer:///mnt/user-data/outputs/audio_video_processor.py)** (GPU-enabled)
- **[test_gpu.py](computer:///mnt/user-data/outputs/test_gpu.py)** (Test script)
