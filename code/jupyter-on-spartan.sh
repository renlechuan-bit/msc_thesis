# How to run Jupyter on Spartan

# 1) create a screen
screen -S jupyter

# 2) start an sinteractive job from inside the screen
sinteractive -p cascade,interactive,sapphire --time=12:00:00 --cpus-per-task=1 --mem=16G --account=punim2504

# 3) start jupyter, set IP to hostname (current node)
jupyter-lab --no-browser --port 9000 --ip $(hostname)

# 4) copy the URL for the jupyter server, should be something like http://spartan-bm163.hpc.unimelb.edu.au:9000/lab

# 5) on VS Code, open an ipynb file, on top right corner, go on select kernel, paste your jupyterlab-server URL

# Note: if using environments, make sure that the environment has ipykernel installed (mamba install -n <ENV-NAME> ipykernel)