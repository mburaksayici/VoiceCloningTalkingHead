
# Base directory
mkdir -p content/memo/checkpoints

# Main model directories
mkdir -p content/memo/checkpoints/audio_proj
mkdir -p content/memo/checkpoints/diffusion_net
mkdir -p content/memo/checkpoints/image_proj
mkdir -p content/memo/checkpoints/reference_net
mkdir -p content/memo/checkpoints/vae
mkdir -p content/memo/checkpoints/wav2vec2
mkdir -p content/memo/checkpoints/emotion2vec_plus_large

# Misc directories
mkdir -p content/memo/checkpoints/misc/audio_emotion_classifier
mkdir -p content/memo/checkpoints/misc/face_analysis/models
mkdir -p content/memo/checkpoints/misc/vocal_separator


	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/audio_proj/config.json -d content/memo/checkpoints/audio_proj -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/audio_proj/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/audio_proj -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/diffusion_net/config.json -d content/memo/checkpoints/diffusion_net -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/diffusion_net/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/diffusion_net -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/image_proj/config.json -d content/memo/checkpoints/image_proj -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/image_proj/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/image_proj -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/audio_emotion_classifier/config.json -d content/memo/checkpoints/misc/audio_emotion_classifier -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/misc/audio_emotion_classifier -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/1k3d68.onnx -d content/memo/checkpoints/misc/face_analysis/models -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/2d106det.onnx -d content/memo/checkpoints/misc/face_analysis/models -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/face_landmarker_v2_with_blendshapes.task -d content/memo/checkpoints/misc/face_analysis/models -o face_landmarker_v2_with_blendshapes.task && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/genderage.onnx -d content/memo/checkpoints/misc/face_analysis/models -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/glintr100.onnx -d content/memo/checkpoints/misc/face_analysis/models -o glintr100.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/face_analysis/models/scrfd_10g_bnkps.onnx -d content/memo/checkpoints/misc/face_analysis/models -o scrfd_10g_bnkps.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/misc/vocal_separator/Kim_Vocal_2.onnx -d content/memo/checkpoints/misc/vocal_separator -o Kim_Vocal_2.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/download_checks.json -d content/memo/checkpoints/misc/vocal_separator -o download_checks.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/mdx_model_data.json -d content/memo/checkpoints/misc/vocal_separator -o mdx_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/misc/vocal_separator/vr_model_data.json -d content/memo/checkpoints/misc/vocal_separator -o vr_model_data.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/raw/main/reference_net/config.json -d content/memo/checkpoints/reference_net -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/memoavatar/memo/resolve/main/reference_net/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/reference_net -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors -d content/memo/checkpoints/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/sd-vae-ft-mse/raw/main/config.json -d content/memo/checkpoints/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors -d content/memo/checkpoints/wav2vec2 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/config.json -d content/memo/checkpoints/wav2vec2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/facebook/wav2vec2-base-960h/raw/main/preprocessor_config.json -d content/memo/checkpoints/wav2vec2 -o preprocessor_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/resolve/main/model.pt -d content/memo/checkpoints/emotion2vec_plus_large -o model.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/config.yaml -d content/memo/checkpoints/emotion2vec_plus_large -o config.yaml && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/configuration.json -d content/memo/checkpoints/emotion2vec_plus_large -o configuration.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/emotion2vec/emotion2vec_plus_large/raw/main/tokens.txt -d content/memo/checkpoints/emotion2vec_plus_large -o tokens.txt

