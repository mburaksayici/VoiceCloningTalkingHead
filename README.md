# 🎙️🧠 TalkingHead-VoiceCloner

Generate realistic **AI-talking head videos** using just:
- a **single input image**
- a **30-second voice sample**

This project combines **MEMO** (for video generation) and **StyleTTS 2** (for voice cloning) into a streamlined pipeline.

** Still not ready to use due to stability issues, will share a dockerhub link **
---

## 🎬 Demo

You can find a demo output in:

```
output-i.mp4
```



https://github.com/user-attachments/assets/280e7aed-6e5d-4a5b-9b15-e4a7dd99bc13



---

## 🧠 Models Used

### 🗣️ [StyleTTS 2](https://arxiv.org/abs/2310.10320): Human-Level Text-to-Speech
> Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani  
StyleTTS 2 leverages style diffusion and adversarial training with large speech language models (SLMs) for highly expressive, human-level text-to-speech. Supports zero-shot speaker cloning.

### 🎥 [MEMO](https://arxiv.org/abs/2403.13034): Memory-Guided Diffusion for Talking Head Generation
> Longtao Zheng, Yifan Zhang, et al.  
MEMO is a state-of-the-art diffusion-based model that generates identity-consistent, lip-synced, and emotionally expressive talking head videos from a still image and audio.

---

## 🧪 Hardware Requirements

- **GPU with at least 24 GB VRAM**
- ✅ Recommended: A100, H100, RTX 6000 Ada, RTX 3090, L40/L40S
- ✅ **Tested on [Runpod.io](https://runpod.io)**:
  - **PyTorch 2.8**
  - **Container size ≥ 50 GB**
  - **CUDA 12.4 compatible**

---

## 🧾 Python Usage

```python
from src.models.audio_driven_talking_head_models.MeMoTalkingHead.MeMoTalkingHead import MeMoTalkingHead
from src.models.voice_cloning_models.StyleTTS2.StyleTTS2Cloning import StyleTTS2Cloning
from src.pipelines.TalkingHeadVoiceCloningPipeline import TalkingHeadPipeline

def main():
    # Initialize models
    voice_cloning = StyleTTS2Cloning(device="cpu")
    talking_head = MeMoTalkingHead(checkpoint_dir="content/memo/checkpoints/", device="cuda")

    # Create pipeline
    pipeline = TalkingHeadPipeline(talking_head, voice_cloning)
    
    # Generate video
    output = pipeline.generate(
        reference_image="image.jpg",
        reference_audio="reference.wav",
        prompt="Hi"
    )

    print(f"Generated video saved to: {output}")
    return output

if __name__ == "__main__":
    main()
```

---

## 💻 CLI Usage

You can also use a command-line interface version:

```bash
python generate.py \
  --image image.jpg \
  --audio reference.wav \
  --text "Hi. How are you? "
```


---

## 📄 Citation

### MEMO:
```
@article{zheng2024memo,
  title={MEMO: Memory-Guided Diffusion for Expressive Talking Video Generation},
  author={Zheng, Longtao and Zhang, Yifan and Guo, Hanzhong and Pan, Jiachun and Tan, Zhenxiong and Lu, Jiahao and Tang, Chuanxin and An, Bo and Yan, Shuicheng},
  journal={arXiv preprint arXiv:2403.13034},
  year={2024}
}
```

### StyleTTS 2:
```
@article{li2023styletts2,
  title={StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models},
  author={Li, Yinghao Aaron and Han, Cong and Raghavan, Vinay S. and Mischler, Gavin and Mesgarani, Nima},
  journal={arXiv preprint arXiv:2310.10320},
  year={2023}
}
```

---

## 🔒 License

This project is built for educational and research purposes only. All rights for the underlying models belong to their respective authors.
