# Instructions to run Game of Tag example

## Build manually and run locally
Currently, do not install `smarts` directly from pypi. Instead, install `smarts` from the `smarts-got` branch of `huaweinoah/smarts` repository. 

The steps to install and run smarts-got is as follows.

```bash
# Install Smarts and build scenarios
$ cd /path/to/SMARTS/
$ git clone https://github.com/huawei-noah/SMARTS.git 
$ git checkout smarts-got
# Follow prompts for setting up sumo and SUMO_HOME environment variable
$ ./install_deps.sh
# Verify sumo is >= 1.5.0
$ sumo
$ python3.7 -m venv .venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .[train,test,dev,camera-obs,got]
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
1. set `headless: False` in `./examples/gameOfTag/got.yaml` file
2. Run `scl` in a separate terminal
    ```bash
    $ scl envision start -s ./scenarios -p 8081
    ```
3. Navigate to localhost:8081

## Build and run in docker container
```bash
# In host terminal
$ cd /path/to/SMARTS
$ export VERSION=v0.4.19
$ docker build --network=host -f ./utils/docker/Dockerfile.tensorflow -t adaickalavan/smarts:$VERSION-tensorflow .
$ docker run --rm -it --gpus=all --network=host --volume=/home/kyber/workspaces/SMARTS/:/src/ adaickalavan/smarts:$VERSION-tensorflow

# In interactive docker container bash 
$ cd /src
# To train
$ python3.7 ./examples/gameOfTag/train.py
# To evaluate
$ python3.7 ./examples/gameOfTag/evaluate.py
```

## Tensorboard
```
$ tensorboard --logdir=/home/kyber/workspaces/SMARTS/examples/gameOfTag/logs
```