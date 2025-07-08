# voice_cloning_models/StyleTTS2/StyleTTS2Cloning.py

import os
import sys
from pathlib import Path
from typing import Union, Optional
import yaml
import traceback 

import torch
import numpy as np
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import phonemizer
import nltk
nltk.download('punkt_tab')


# Add StyleTTS2 to Python path
STYLETTS2_PATH = Path(__file__).parent / "StyleTTS2"
if str(STYLETTS2_PATH) not in sys.path:
    sys.path.append(str(STYLETTS2_PATH))



from src.base.VoiceCloning import BaseVoiceCloning
from .StyleTTS2.models import build_model, load_ASR_models, load_F0_models
from .StyleTTS2.Utils.PLBERT.util import load_plbert
from .StyleTTS2.utils import recursive_munch
from .StyleTTS2.text_utils import TextCleaner
from .StyleTTS2.Modules.diffusion.sampler import (
    DiffusionSampler, 
    ADPM2Sampler, 
    KarrasSchedule
)


class StyleTTS2Cloning(BaseVoiceCloning):
    """StyleTTS2 implementation of voice cloning"""
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path]="src/models/voice_cloning_models/StyleTTS2/StyleTTS2/Configs",
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        self.config_dir = Path(checkpoint_dir)
        self.checkpoint_path = "src/models/voice_cloning_models/StyleTTS2/StyleTTS2/StyleTTS2-LibriTTS/Models/LibriTTS/epochs_2nd_00020.pth"

        
        
        # Initialize components
        self._init_transforms()
        self._init_models()
        self._init_phonemizer()
        self.text_cleaner = TextCleaner()

    def _init_transforms(self):
        """Initialize audio transforms"""
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, 
            n_fft=2048, 
            win_length=1200, 
            hop_length=300
        )
        self.mean, self.std = -4, 4

    def _init_phonemizer(self):
        """Initialize phonemizer"""
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True
        )

    def _init_models(self):
        """Initialize all required models"""
        # Load config
        config_path = self.config_dir / "config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Load ASR model
        ASR_config = config.get('ASR_config')
        ASR_config = os.path.join ( "src/models/voice_cloning_models/StyleTTS2/StyleTTS2" , config.get('ASR_config'),  ) 

        ASR_path = os.path.join ( "src/models/voice_cloning_models/StyleTTS2/StyleTTS2" , config.get('ASR_path'),  ) 
        self.text_aligner = load_ASR_models(ASR_path, ASR_config)
        
        # Load F0 model
        F0_path = config.get('F0_path', False)
        F0_path = os.path.join ( "src/models/voice_cloning_models/StyleTTS2/StyleTTS2" , config.get('F0_path'),  ) 
        self.pitch_extractor = load_F0_models(F0_path)
        
        # Load BERT model
        BERT_path = config.get('PLBERT_dir', False)
        BERT_path = os.path.join ( "src/models/voice_cloning_models/StyleTTS2/StyleTTS2" , config.get('PLBERT_dir'),  ) 

        self.plbert = load_plbert(BERT_path)
        
        # Build main model
        model_params = recursive_munch(config['model_params'])
        self.model_params = model_params
        self.model = build_model(
            model_params,
            self.text_aligner,
            self.pitch_extractor,
            self.plbert
        )

        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        # Load checkpoints

        params_whole = torch.load(self.checkpoint_path, map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except Exception as exc:
                    exception_string = traceback.format_exc()

                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()

                    for k, v in state_dict.items():
                        # Remove prefix if present
                        if k.startswith("module.generator."):
                            name = k[len("module.generator."):]
                        elif k.startswith("module."):
                            name = k[7:]
                        else:
                            name = k
                        new_state_dict[name] = v

                    try:
                        self.model[key].load_state_dict(new_state_dict, strict=False)
                    except Exception as exc2:
                        print(f"hell no check here. key : {key}  original exception : {exception_string}")
                        print(f"second exception: {traceback.format_exc()}")
                        pass
        
        # Set models to eval mode and move to device
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
            
        # Initialize diffusion sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001,
                sigma_max=3.0,
                rho=9.0
            ),
            clamp=False
        )

    def _preprocess_audio(self, wave: np.ndarray) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        print("Mel shape:", mel_tensor.shape, "Mean:", mel_tensor.mean().item(), "Std:", mel_tensor.std().item())

        return mel_tensor

    def _compute_style(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Compute style embedding from reference audio"""
        wave, sr = librosa.load(str(audio_path), sr=24000)
        audio, _ = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        
        mel_tensor = self._preprocess_audio(audio).to(self.device)
        
        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        
        return torch.cat([ref_s, ref_p], dim=1)

    def _length_to_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        """Convert lengths to mask"""
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(
            lengths.shape[0], -1
        ).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def clone_voice(
        self,
        reference_audio: Union[str, Path],
        prompt: str,
        output_path: Optional[Union[str, Path]] = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0
    ) -> str:
        """
        Clone voice and generate speech
        
        Args:
            reference_audio: Path to reference audio file
            prompt: Text to be synthesized
            output_path: Path for output audio
            alpha: Style mixing parameter for reference
            beta: Style mixing parameter for prediction
            diffusion_steps: Number of diffusion steps
            embedding_scale: Scale for BERT embedding
            
        Returns:
            str: Path to generated audio file
        """
        # Compute style from reference audio
        ref_s = self._compute_style(reference_audio)
        
        # Prepare text
        text = prompt.strip()
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        print("Phonemes:", ps)


        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Process text
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)
            
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
            
            # Generate style
            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps
            ).squeeze(1)
            
            # Style mixing
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            
            # Generate durations
            d = self.model.predictor.text_encoder(
                d_en, s, input_lengths, text_mask
            )
            
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            # Create alignment
            pred_aln_trg = torch.zeros(
                input_lengths, int(pred_dur.sum().data)
            )
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            
            # Generate audio
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new
            
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new
            
            out = self.model.decoder(
                asr,
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0)
            )
            
            # Convert to waveform
            waveform = out.squeeze().cpu().numpy()[..., :-50]
        
        # Save audio
        if output_path is None:
            output_path = f"output_{hash(prompt)}.wav"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save using scipy
        from scipy.io import wavfile
        wavfile.write(str(output_path), 24000, waveform)
        
        return str(output_path)