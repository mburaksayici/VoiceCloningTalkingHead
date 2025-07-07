
from pathlib import Path
import sys
import os 
import random 
import string 

from src.base.AudioDrivenTalkingHead import BaseTalkingHead, TalkingHeadInput

import torch
import numpy as np

import argparse
import logging
import os

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from packaging import version
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.abspath("src/models/audio_driven_talking_head_models/MeMoTalkingHead/memotalkinghead_submodule"))

from memo.models.audio_proj import AudioProjModel
from memo.models.image_proj import ImageProjModel
from memo.models.unet_2d_condition import UNet2DConditionModel
from memo.models.unet_3d import UNet3DConditionModel
from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video




# Add StyleTTS2 to Python path
MEMO_PATH = Path(__file__).parent / "memo"
if str(MEMO_PATH) not in sys.path:
    sys.path.append(str(MEMO_PATH))



def get_random_device():
    # Check if there are at least two CUDA devices available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print("MULTIPLE GPUS, YEEEEY")
        return torch.device(f"cuda:{random.choice([0, 1])}")
    elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")  # Default to CPU if no CUDA device is available




class MeMoTalkingHead(BaseTalkingHead):
    """Implementation of MeMo talking head model"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print("MULTIPLE GPUS, YEEEEY"*100)
            device_0="cuda:0"
            device_1="cuda:1"

        else:
            device_0="cuda:0"
            device_1="cuda:0"

        self.device_0 = torch.device(device_0)
        self.device_1 = torch.device(device_1)
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
        self.reference_net.enable_xformers_memory_efficient_attention()
        self.diffusion_net.enable_xformers_memory_efficient_attention()
        
        # Initialize pipeline
        self.pipeline = self._setup_pipeline()
    
    def _load_vae(self):
        from diffusers import AutoencoderKL
        return AutoencoderKL.from_pretrained(
            self.checkpoint_dir / "vae"
        ).to(device=self.device_0, dtype=self.weight_dtype)
    
    def _load_reference_net(self):
        from .memo.memo.models.unet_2d_condition import UNet2DConditionModel
        return UNet2DConditionModel.from_pretrained(
            self.checkpoint_dir, subfolder="reference_net", 
            use_safetensors=True
        ).to(device=self.device_1, dtype=self.weight_dtype)
    
    def _load_diffusion_net(self):
        from .memo.memo.models.unet_3d import UNet3DConditionModel
        return UNet3DConditionModel.from_pretrained(
            self.checkpoint_dir, subfolder="diffusion_net", 
            use_safetensors=True
        ).to(device=self.device_0, dtype=self.weight_dtype)
    
    def _load_image_proj(self):
        from .memo.memo.models.image_proj import ImageProjModel
        return ImageProjModel.from_pretrained(
            self.checkpoint_dir, subfolder="image_proj", 
            use_safetensors=True
        ).to(device=self.device_1, dtype=self.weight_dtype)
    
    def _load_audio_proj(self):
        from .memo.memo.models.audio_proj import AudioProjModel
        return AudioProjModel.from_pretrained(
            self.checkpoint_dir, subfolder="audio_proj", 
            use_safetensors=True
        ).to(device=self.device_0, dtype=self.weight_dtype)
    
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
        return pipeline.to(device=self.device_1, dtype=self.weight_dtype)
    
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
        num_past_frames = 5
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
            str(Path(audio_path).with_suffix('.wav')))

        # Generate a random directory name with 10 characters
        random_dir_name = ''.join(random.choices(string.ascii_letters + string.digits, k=10))

        # Create the directory
        os.makedirs(random_dir_name, exist_ok=True)
        


        cache_dir = os.path.join(random_dir_name, "audio_preprocess")

        audio_emb, audio_length = preprocess_audio(
            wav_path=audio_path,
            num_generated_frames_per_clip=num_generated_frames_per_clip,
            fps=fps,
            wav2vec_model=str(self.checkpoint_dir / "wav2vec2"),
            vocal_separator_model=str(self.checkpoint_dir / "misc/vocal_separator/Kim_Vocal_2.onnx"),
            device=self.device_0,
            cache_dir = cache_dir
        )
        
        audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
            model=str(self.checkpoint_dir),
            wav_path=audio_path,
            emotion2vec_model=str(self.checkpoint_dir / "emotion2vec_plus_large"),
            audio_length=audio_length,
            device=self.device_1
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


    def _generate_video_frames(self,pixel_values, face_emb, audio_emb, audio_emotion, num_emotion_classes,
                resolution, num_generated_frames_per_clip, num_init_past_frames, 
                num_past_frames, inference_steps, cfg_scale, generator):


        print("Loading models")

        self.vae.requires_grad_(False).eval()
        self.reference_net.requires_grad_(False).eval()
        self.diffusion_net.requires_grad_(False).eval()
        self.image_proj.requires_grad_(False).eval()
        self.audio_proj.requires_grad_(False).eval()

        # Enable memory-efficient attention for xFormers
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            #reference_net.enable_xformers_memory_efficient_attention()
            #diffusion_net.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Create inference pipeline
       

        video_frames = []
        num_clips = audio_emb.shape[0] // num_generated_frames_per_clip
        for t in tqdm(range(num_clips), desc="Generating video clips"):
            if len(video_frames) == 0:
                # Initialize the first past frames with reference image
                past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1)
                past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
            else:
                past_frames = video_frames[-1][0]
                past_frames = past_frames.permute(1, 0, 2, 3)
                past_frames = past_frames[0 - num_past_frames :]
                past_frames = past_frames * 2.0 - 1.0
                past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)

            audio_tensor = (
                audio_emb[
                    t
                    * num_generated_frames_per_clip : min(
                        (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                    )
                ]
                .unsqueeze(0)
                .to(device=self.audio_proj.device, dtype=self.audio_proj.dtype)
            )
            audio_tensor = self.audio_proj(audio_tensor)

            audio_emotion_tensor = audio_emotion[
                t
                * num_generated_frames_per_clip : min(
                    (t + 1) * num_generated_frames_per_clip, audio_emb.shape[0]
                )
            ]
            pipeline_output = self.pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                audio_emotion=audio_emotion_tensor,
                emotion_class_num=num_emotion_classes,
                face_emb=face_emb,
                width=pixel_values.shape[2],
                height=pixel_values.shape[3],
                video_length=num_generated_frames_per_clip,
                num_inference_steps=inference_steps,
                guidance_scale=cfg_scale,
                generator=generator,
                is_new_audio=t == 0,
            )

            video_frames.append(pipeline_output.videos)

            torch.cuda.empty_cache()

        video_frames = torch.cat(video_frames, dim=2)
        video_frames = video_frames.squeeze(0)
        video_frames = video_frames[:, :audio_length]

        return video_frames
