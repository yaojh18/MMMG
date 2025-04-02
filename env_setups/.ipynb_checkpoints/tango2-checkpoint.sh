## set up tango2 environments
conda create -n tango2 python=3.10 && conda activate tango2
git clone https://github.com/declare-lab/tango.git
cd tango
pip install -r requirements.txt