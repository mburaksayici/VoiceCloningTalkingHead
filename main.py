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
        reference_audio="audio.mp3",
        prompt="Hi. How are you? Is everything alright?"
    )

    print(f"Generated video saved to: {output}")
    return output


if __name__ == "__main__":
    main()
