## set up tango2 environments
conda create -n tango python=3.10 -y && conda activate tango
git clone https://github.com/declare-lab/tango.git
cd tango
pip install -r requirements.txt