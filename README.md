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

# to run with different track, try
$ python run_gatsby.py HCI
```

You can collect data across different parameters using the following:

```console
$ python collect_data.py 
```