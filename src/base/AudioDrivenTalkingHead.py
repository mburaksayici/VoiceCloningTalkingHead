from abc import ABC, abstractmethod
from typing import * 
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TalkingHeadInput:
    """Input data for talking head generation"""
    reference_image: Union[str, Path]
    prompt: str
    reference_audio: Optional[Union[str, Path]] = None
    generated_video: Optional[Union[str, Path]] = None


class BaseTalkingHead(ABC):
    """Base class for talking head models"""
    
    @abstractmethod
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        """Initialize talking head model"""
        pass
    
    @abstractmethod
    def generate_video(self, input_data: TalkingHeadInput) -> str:
        """
        Generate talking head video
        
        Args:
            input_data: TalkingHeadInput containing required data
            
        Returns:
            str: Path to generated video file
        """
        pass

