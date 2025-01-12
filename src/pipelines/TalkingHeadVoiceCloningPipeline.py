from src.base.VoiceCloning import BaseVoiceCloning, VoiceCloningInput
from src.base.AudioDrivenTalkingHead import BaseTalkingHead, TalkingHeadInput


class TalkingHeadPipeline:
    """Pipeline combining voice cloning and talking head generation"""
    
    def __init__(
        self,
        talking_head: BaseTalkingHead,
        voice_cloning: BaseVoiceCloning
    ):
        self.talking_head = talking_head
        self.voice_cloning = voice_cloning
    
    def generate(self, input_data: TalkingHeadInput) -> str:
        """
        Generate talking head video with cloned voice
        
        Args:
            input_data: TalkingHeadInput containing all required data
            
        Returns:
            str: Path to generated video file
        """
        # First clone the voice if reference audio is provided
        if input_data.reference_audio:
            voice_input = VoiceCloningInput(
                reference_audio=input_data.reference_audio,
                prompt=input_data.prompt
            )
            input_data.generated_audio = self.voice_cloning.clone_voice(voice_input)
        
        # Then generate the video
        return self.talking_head.generate_video(input_data)