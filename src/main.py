from src.models.audio_driven_talking_head_models.MeMoTalkingHead import MeMoTalkingHead
from src.models.voice_cloning_models import DummyVoiceCloning
from src.pipelines.TalkingHeadVoiceCloningPipeline import TalkingHeadPipeline

def main():
    # Initialize models
    talking_head = MeMoTalkingHead(checkpoint_dir="path/to/checkpoints")
    voice_cloning = DummyVoiceCloning()
    
    # Create pipeline
    pipeline = TalkingHeadPipeline(talking_head, voice_cloning)
    
    # Generate video
    output = pipeline.generate(
        image_path="examples/assets/image.jpg",
        reference_audio="examples/assets/reference.wav",
        prompt="Hello, this is a test"
    )
    print(f"Generated video saved to: {output}")

if __name__ == "__main__":
    main()
