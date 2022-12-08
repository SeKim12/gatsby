# GATSby
Four-year plan generator for Stanford CS using Genetic Algorithms augmented with Tabu Search. 

## Requirements
- python >= 3.8.9

## Using Gatsby
First, install the required modules:

```console
$ pip install -r requirements.txt 
```

You can run a single instance of Gatsby by running the following:

```console
# this will default to AI
$ python run_gatsby.py

# to run with different track (e.g. AI, HCI, Systems), try
$ python run_gatsby.py <track_name>
```

## Experiments
You can collect data across different parameters using the following:
```console
$ python collect_data.py 
```
You can change parameters using `config.yaml` and set multiple parameter ranges using `telemetry/wrapper.py`.
Refer to documentation in `telemetry/wrapper.py` for more information.
## Directory
```angular2html
.
├── data
│   ├── sample_courses.json              # modified json file from CS 221 scheduling assignment
├── gats                    
│   ├── algorithm              
│   │   ├── gatsby.py                    # main algorithm and interfaces
│   │   ├── genops.py                    # impl. of different genetic operators
│   ├── data              
│   │   ├── util.py                      # data collection/processing pipeline
│   ├── model              
│   │   ├── scheduler.py                 # Stanford CS course scheduling problem model
│   ├── telemetry              
│   │   ├── wrapper.py                   # run Gatsby across multiple parameters
...
```

## Resources
 * [Final Paper](https://drive.google.com/file/d/1Ot7TVMxMdfXPOKcorhY-jWCaw5eyFCnw/view?usp=sharing)
 * [Project Video](https://www.youtube.com/watch?v=LmFLmpgDGbE)
 * [Project Slides](https://drive.google.com/file/d/19_kvqYC-SHbyIMhpfD0Ii-_MpqAC6vCe/view?usp=sharing)