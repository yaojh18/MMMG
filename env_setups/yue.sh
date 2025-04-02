## set up yue environments
conda create -n yue python=3.8 -y && conda activate yue
conda config --set channel_priority false
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)
##pip install flash-attn --no-build-isolation

## download infer code 
sudo yum update
sudo yum install git-lfs
git lfs install
git clone https://github.com/multimodal-art-projection/YuE.git
cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer