
from pathlib import Path
import sys

from src.base.AudioDrivenTalkingHead import BaseTalkingHead, TalkingHeadInput

import torch
import numpy as np


# Add StyleTTS2 to Python path
MEMO_PATH = Path(__file__).parent / "memo"
if str(MEMO_PATH) not in sys.path:
    sys.path.append(str(MEMO_PATH))





class MeMoTalkingHead(BaseTalkingHead):
    """Implementation of MeMo talking head model"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.weight_dtype = torch.bfloat16
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Initialize models
        self.vae = self._load_vae()
        self.reference_net = self._load_reference_net()
        self.diffusion_net = self._load_diffusion_net()
        self.image_proj = self._load_image_proj()
        self.audio_proj = self._load_audio_proj()
        
        # Set models to eval mode
        for model in [self.vae, self.reference_net, self.diffusion_net, 
                     self.image_proj, self.audio_proj]:
            model.requires_grad_(False)
            model.eval()
        
        # Enable memory efficient attention
        # self.reference_net.enable_xformers_memory_efficient_attention()
        # self.diffusion_net.enable_xformers_memory_efficient_attention()
        
        # Initialize pipeline
        self.pipeline = self._setup_pipeline()
    
    def _load_vae(self):
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(
            self.checkpoint_dir / "vae"
        ).to(device=self.device, dtype=self.weight_dtype)
    
    def _load_reference_net(self):
        from .memo.memo.models.unet_2d_condition import UNet2DConditionModel
        return UNet2DConditionModel.from_pretrained(
            self.checkpoint_dir, subfolder="reference_net", 
            use_safetensors=True
        ).to(device=self.device, dtype=self.weight_dtype)
    
    def _load_diffusion_net(self):
        from .memo.memo.models.unet_3d import UNet3DConditionModel
        return UNet3DConditionModel.from_pretrained(
            self.checkpoint_dir, subfolder="diffusion_net", 
            use_safetensors=True
        ).to(device=self.device, dtype=self.weight_dtype)
    
    def _load_image_proj(self):
        from .memo.memo.models.image_proj import ImageProjModel
        return ImageProjModel.from_pretrained(
            self.checkpoint_dir, subfolder="image_proj", 
            use_safetensors=True
        ).to(device=self.device, dtype=self.weight_dtype)
    
    def _load_audio_proj(self):
        from .memo.memo.models.audio_proj import AudioProjModel
        return AudioProjModel.from_pretrained(
            self.checkpoint_dir, subfolder="audio_proj", 
            use_safetensors=True
        ).to(device=self.device, dtype=self.weight_dtype)
    
    def _setup_pipeline(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from .memo.memo.pipelines.video_pipeline import VideoPipeline
        
        scheduler = FlowMatchEulerDiscreteScheduler()
        pipeline = VideoPipeline(
            vae=self.vae,
            reference_net=self.reference_net,
            diffusion_net=self.diffusion_net,
            scheduler=scheduler,
            image_proj=self.image_proj
        )
        return pipeline.to(device=self.device, dtype=self.weight_dtype)
    
    def generate_video(self, reference_audio,reference_image, generated_video ) -> str:
        from .memo.memo.utils.audio_utils import (
            extract_audio_emotion_labels, 
            preprocess_audio, 
            resample_audio
        )
        from .memo.memo.utils.vision_utils import preprocess_image, tensor_to_video
        
        # Configuration
        resolution = 512
        num_generated_frames_per_clip = 16
        fps = 30
        num_init_past_frames = 2
        num_past_frames = 16
        inference_steps = 20
        cfg_scale = 3.5
        seed = 42
        generator = torch.manual_seed(seed)
        
        # Process image
        pixel_values, face_emb = preprocess_image(
            face_analysis_model=str(self.checkpoint_dir / "misc/face_analysis"),
            image_path=reference_image,
            image_size=resolution
        )
        
        # Process audio
        audio_path = reference_audio
        audio_path = resample_audio(
            audio_path, 
            str(Path(audio_path).with_suffix('.wav'))
        )
        cache_dir = "cache/"

        cache_dir = os.path.join(output_dir, "audio_preprocess")
        audio_emb, audio_length = preprocess_audio(
            wav_path=audio_path,
            num_generated_frames_per_clip=num_generated_frames_per_clip,
            fps=fps,
            wav2vec_model=str(self.checkpoint_dir / "wav2vec2"),
            vocal_separator_model=str(self.checkpoint_dir / "misc/vocal_separator/Kim_Vocal_2.onnx"),
            device=self.device
        )
        
        audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
            model=str(self.checkpoint_dir),
            wav_path=audio_path,
            emotion2vec_model=str(self.checkpoint_dir / "emotion2vec_plus_large"),
            audio_length=audio_length,
            device=self.device
        )
        
        # Generate video frames
        video_frames = self._generate_video_frames(
            pixel_values=pixel_values,
            face_emb=face_emb,
            audio_emb=audio_emb,
            audio_emotion=audio_emotion,
            num_emotion_classes=num_emotion_classes,
            resolution=resolution,
            num_generated_frames_per_clip=num_generated_frames_per_clip,
            num_init_past_frames=num_init_past_frames,
            num_past_frames=num_past_frames,
            inference_steps=inference_steps,
            cfg_scale=cfg_scale,
            generator=generator
        )
        
        # Create final video
        output_path = f"output-{seed}.mp4"
        tensor_to_video(
            video_frames[:, :audio_length],
            output_path,
            audio_path,
            fps=fps
        )
        
        return output_path