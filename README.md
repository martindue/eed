# EED — Eye Event Detection

Train and use models for detection of events in eye tracking data

## Setup

Clone the project  

~~~bash  
  git clone https://github.com/martindue/eed.git
~~~

Go to the project directory  

~~~bash  
  cd eed
~~~

Install dependencies (using conda)

~~~bash  
conda env create -f environment.yaml
~~~

Activate the environment

~~~bash  
conda activate eed
~~~


## Data
The code currently works with the 6 .npy files in this folder:  [https://github.com/r-zemblys/irf/tree/master/etdata/lookAtPoint_EL]( https://github.com/r-zemblys/irf/tree/master/etdata/lookAtPoint_EL)

They have to be placed into the folders *.data/raw/train_data* and *.data/raw/test_data/* which both need to be created manually by the user. 

I plan to add support for more [publicly available data](https://github.com/r-zemblys/EM-event-detection-evaluation?tab=readme-ov-file#list-of-publicly-available-annotated-eye-movement-datasets). 

## Usage
The program uses the lightningCLI and jsonargparse command line interfaces. 

Running training:
~~~bash  
python src/ml/scripts/main.py fit --config src/ml/config/main.yaml
~~~

Many options and settings can be set in the config file.

Hyperparamter search for sklearn with Optuna:
~~~bash  
python src/ml/scripts/tune_sklearn.py --config src/ml/config/sklearn_tune.yaml
~~~
## License  

[MIT](https://choosealicense.com/licenses/mit/)
