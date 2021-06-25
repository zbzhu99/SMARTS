# Instructions to run SMARTS with XingTian

## Build manually and run locally
Currently, do not install `smarts` directly from pypi. Instead, install `smarts` from the `xtsmarts` branch of `huaweinoah/smarts` repository. 

The steps to install and run Smarts with XingTian is as follows.

```bash
# Install Smarts and build scenarios
$ cd /path/to/SMARTS/
$ git clone https://github.com/huawei-noah/SMARTS.git 
$ git checkout xtsmarts
# Follow prompts for setting up sumo and SUMO_HOME environment variable
$ ./install_deps.sh
# Verify sumo is >= 1.5.0
$ sumo
$ python3.7 -m venv .venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install .
$ scl scenario build-all ./scenarios
$ deactivate

# Install XingTian
$ cd /path/to/xingtian
$ git clone https://github.com/Adaickalavan/xingtian.git
$ git checkout xtsmarts
$ python3.7 -m venv .venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install git+https://github.com/huawei-noah/SMARTS.git@xtsmarts
$ pip install -r ./examples/smarts/requirements.txt
$ pip install -r ./requirements.txt
$ pip install -e .

# Note: Modify the `scenarios` path in `/path/to/xingtian/examples/smarts/smarts.yaml` file to point to the desired scenario, e.g., `/path/to/SMARTS/scenarios/loop`.

# Run the code 
$ python3.7 xt/main.py -f examples/smarts/smarts.yaml
```

## Build and run in docker container
```bash
# In host terminal
$ cd /path/to/xingtian
$ export XTSMARTS_VERSION=v0.1
$ docker build --network=host -f ./docker/Dockerfile.smarts -t xtsmarts:${XTSMARTS_VERSION} .
$ docker run --rm -it --network=host xtsmarts:${XTSMARTS_VERSION}

# In interactive docker container bash 
# To train
$ python3.7 xt/main.py -f examples/smarts/smarts.yaml
# To evaluate
$ python3.7 xt/main.py -f examples/smarts/smarts.yaml -t evaluate
```