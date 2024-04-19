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

They have to be placed into the folders *.data/raw/train_data* and *.data/raw/test_data/* which both needs to be created. 

I plan to add support for more [publicly available data](https://github.com/r-zemblys/EM-event-detection-evaluation?tab=readme-ov-file#list-of-publicly-available-annotated-eye-movement-datasets). 
## License  

[MIT](https://choosealicense.com/licenses/mit/)
