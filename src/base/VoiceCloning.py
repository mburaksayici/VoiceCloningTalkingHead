from abc import ABC, abstractmethod
from typing import * 
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VoiceCloningInput:
    """Input data for voice cloning"""
    reference_audio: Union[str, Path]
    prompt: str


class BaseVoiceCloning(ABC):
    """Base class for voice cloning models"""
    
    @abstractmethod
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize voice cloning model"""
        pass
    
    @abstractmethod
    def clone_voice(self, input_data: VoiceCloningInput) -> str:
        """
        Clone voice and generate speech
        
        Args:
            input_data: VoiceCloningInput containing reference audio and prompt
            
        Returns:
            str: Path to generated audio file
        """
        pass
