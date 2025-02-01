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
    
    def generate(self, reference_audio, reference_image, prompt) -> str: # TalkingHeadInput
        """
        Generate talking head video with cloned voice
        
        Args:
            input_data: TalkingHeadInput containing all required data
            
        Returns:
            str: Path to generated video file
        """
        # First clone the voice if reference audio is provided
        breakpoint()
        if reference_audio:
            """voice_input = VoiceCloningInput(
                reference_audio=reference_audio,
                prompt=prompt
            )
            """
        #    generated_audio = self.voice_cloning.clone_voice(reference_audio=reference_audio,prompt=prompt)
        reference_audio = "reference.mp3" # always test with mp3. no wav for now.
        # Then generate the video
        generated_video = "video.mp4"
        return self.talking_head.generate_video(reference_audio,reference_image, generated_video)