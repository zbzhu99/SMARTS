# Instructions to run SMARTS with XingTian

## Build manually and run locally
Currently, do not install `smarts` directly from pypi. Instead, install `smarts` from the `xtsmarts` branch of `huaweinoah/smarts` repository. 

The steps to install and run xtSmarts is as follows.

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
$ pip install -e .[camera-obs]
$ pip install -r ./examples/gameOfTag/requirements.txt
$ scl scenario build-all ./scenarios

# Run the code 
$ python3.7 ./examples/gameOfTag/train.py
```

## Change the following
1. Change the following in `./examples/gameOfTag/smarts.yaml` file

    ```yaml
    env_para:
        headless: True # False : to view output in Envision
        scenarios: # Set full path to scenario
            - /home/kyber/workspaces/SMARTS/scenarios/cloverleaf

    benchmark:
        base_path: "/home/kyber/workspaces/" # Set to a directory where you have read/write permission, for storage of trained models.
    ```

## To visualise in Envision
1. set `headless: False` in `./examples/gameOfTag/smarts.yaml` file
2. Run `scl` in a separate terminal
    ```bash
    $ scl envision start -s ./scenarios -p 8081
    ```
3. Navigate to localhost:6061

## Build and run in docker container
```bash
# In host terminal
$ cd /path/to/smarts
$ export XTSMARTS_VERSION=v0.1
$ docker build --network=host -f ./utils/docker/Dockerfile.got -t xtsmarts:${XTSMARTS_VERSION} .
$ docker run --rm -it --network=host xtsmarts:${XTSMARTS_VERSION}

# In interactive docker container bash 
# To train
$ python3.7 ./examples/gameOfTag/train.py
# To evaluate
$ python3.7 ./examples/gameOfTag/evaluate.py
```



# Useful commands
```bash
$ export VERSION=v0.4.18

$ docker build --network=host -f ./utils/docker/Dockerfile.tensorflow -t adaickalavan/smarts:$VERSION-tensorflow .   

$ docker login
$ docker push adaickalavan/smarts:$VERSION-tensorflow

$ docker run --rm -it --gpus=all --network=host --volume=/home/kyber/workspaces/SMARTS/:/src/ adaickalavan/smarts:v0.4.18-tensorflow

$ cd /src
$ source ./examples/gameOfTag/.venv/bin/activate
$ python3.7 ./examples/gameOfTag/train.py
```
