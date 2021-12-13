# Left Turn in Intersection
This example illustrates the training of an ego agent to make an uprotected left turn in an intersection in traffic using DreamerV2 (https://github.com/danijar/dreamerv2) reinforcement-learning algorithm.

Ego agent earns rewards based on the distance travelled and on succesfully turning left in the intersection. It is penalised for colliding with other vehicles, for going off-road, and for going off-route.

## Trained agent navigating intersection
![](./docs/_static/intersection.gif)

## Setup
```bash
$ git clone https://github.com/huawei-noah/SMARTS.git
$ cd <path>/SMARTS/examples/intersection
$ python3.7 -m venv ./.venv
$ source ./.venv/bin/activate
$ pip install --upgrade pip
$ pip install -e .
$ scl scenario build-all --clean ./scenarios/
```

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/examples/intersection
    $ python3.7 run.py 
    ```
1. Trained model is saved into `<path>/SMARTS/examples/intersection/logs/<folder_name>` folder.

## Evaluate
1. Evaluate
    ```bash
    $ cd <path>/SMARTS/examples/intersection
    $ scl envision start -s ./scenarios &
    $ python3.7 run.py --mode=evaluate --logdir="<path>/SMARTS/examples/intersection/logs/<folder_name>" --head
    ```
1. Go to `localhost:8081` to view the simulation in Envision.

## Docker
1. Build and train inside docker container
    ```bash
    $ cd <path>/SMARTS
    $ docker build --file=<path>/SMARTS/examples/intersection/Dockerfile --network=host --tag=intersection <path>/SMARTS
    $ docker run --rm -it --network=host --gpus=all intersection
    (container) $ cd /src/examples/intersection
    (container) $ python3.7 run.py
    ```