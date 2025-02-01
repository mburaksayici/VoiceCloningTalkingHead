wget https://huggingface.co/yl4579/StyleTTS2-LibriTTS/resolve/main/Models/LibriTTS/epochs_2nd_00020.pth  -P src/models/voice_cloning_models/StyleTTS2/StyleTTS2/StyleTTS2-LibriTTS/Models/LibriTTS/
apt-get install espeak -y
wget https://www.wavsource.com/snds_2020-10-01_3728627494378403/people/comedians/miller_larry.wav -O reference.wav
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y