# ⚡ GPU Quick Start - 4 Bước Đơn Giản

## 🎯 Tăng tốc 5-10x với GPU

---

## BƯỚC 1: Kiểm tra GPU

```powershell
nvidia-smi
```

✅ Thấy thông tin GPU → Tiếp tục  
❌ Lỗi → [Cài driver](https://www.nvidia.com/Download/index.aspx) và restart

---

## BƯỚC 2: Cài PyTorch với CUDA

```powershell
conda activate DicTool
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## BƯỚC 3: Test GPU

Download và chạy **[test_gpu.py](computer:///mnt/user-data/outputs/test_gpu.py)**:

```powershell
python test_gpu.py
```

Phải thấy: `🎉 GPU IS READY TO USE!`

---

## BƯỚC 4: Chạy với GPU

Download **[audio_video_processor.py](computer:///mnt/user-data/outputs/audio_video_processor.py)** (phiên bản mới):

```powershell
# Auto dùng GPU
python audio_video_processor.py actors.mp3

# Output sẽ hiện:
# Using device: CUDA
# GPU: NVIDIA GeForce RTX 3060
```

---

## 📊 So sánh tốc độ

```powershell
# CPU (chậm)
python audio_video_processor.py audio.mp3 --device cpu

# GPU (nhanh 5-10x)
python audio_video_processor.py audio.mp3 --device cuda
```

---

## 🐛 Lỗi thường gặp

### "CUDA not available"

```powershell
# Cài lại PyTorch GPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### "Out of memory"

```powershell
# Dùng model nhỏ hơn
python audio_video_processor.py audio.mp3 --model small
```

---

## ✅ Checklist

- [ ] `nvidia-smi` hiện GPU info
- [ ] `python test_gpu.py` → GPU ready
- [ ] Code hiện "Using device: CUDA"
- [ ] Xử lý nhanh hơn trước 5-10x

---

**Chi tiết đầy đủ:** [GPU_SETUP.md](computer:///mnt/user-data/outputs/GPU_SETUP.md)
