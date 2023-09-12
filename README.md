# TrackingBert
---------------
BERT Model for Track Reconstruction in HEP Analysis

## Setup
### Install TrackingBert
```bash
git clone https://github.com/xju2/TrackingBert
cd TrackingBert
pip install -e .
```
### Install Related Packages
1) Keras BERT, credit to CyberZHG (https://github.com/CyberZHG)
```bash
git clone https://github.com/CyberZHG/keras-bert.git
cd keras-bert
pip install -e .
```
**IMPORTANT**: Replace the `bert.py` file in `keras-bert` with `TrackingBert/bert.py`, due to application-specific adjustments made.

2) Track data processing package
```bash
git clone https://github.com/xju2/visual_tracks.git
cd visual_tracks
pip install -e .
```

## Dataset
* Use internal pre-processed dataset: `/global/cfs/cdirs/m3443/data/trackml-kaggle/train_all`
* Download dataset manually: https://www.kaggle.com/competitions/trackml-particle-identification/data

## Code Structure
* `process_data.py`: processing data into `.npz` files, code can be run in parallel and hence creating multiple files simultaneously
* `merge_inputs.py`: merge the `.npz` files produced by `process_data.py` into big files used for training, validation, and testing
* `train.py`: train a model given parameters and inputs
* `bert.py`: the adjusted BERT file to replace the `keras-bert/bert.py`
* `inference.ipynb`: a notebook containing util functions for conducting inference and plottings, with examples of running inferences
* `tensorboard.ipynb`: a notebook to run the tensorboard for checking training progress
* `detectors.csv`: the data file that contains all detector elements info


## Usage
### Processing Data
Adjust parameters in `process_data.py` and run
```bash
python process_data.py
```
Note that this may take a long time based on how many events there are. It may require several 4h runs on CPU.

After the previous step is done, adjust the directory in `merge_inputs.py` to ensure that this is the folder where all processed data is saved. Then run
```bash
python merge_inputs.py
```

### Training
Adjust parameters in `train.py` and run
```bash
python train.py
```
To check the training progress, run all cells in `tensorboard.ipynb` after replacing the model path, and click the link at the end of the notebook, this will lead you to the tensorboard in a new tab. **NOTE**: Some metrics may be ill-defined and hence not meaningful, depending on the models and tasks. Training/validation loss is usually the best ones to monitor.

Typical training for a 1M-parameter model may take 12-16h on a single GPU.

To submit a job to server through slrum scripts, go into the folder `slurm_scripts` and run `sh submit.sh`. Change the code in `run.sh` as needed. **NOTE**: make sure to *copy* all codes in `lib.sh` and *run them in terminal* before running `sh submit.sh`, to ensure all required Cuda libraries are loaded.


### Inference
Open `inference.ipynb`, run all the util functions (there are a lot!). To get the results for a particular model, run in a cell
```python
get_results(MODEL_PATH, n_sample=N_SAMPLE, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, **kwargs)
```
See the `Inference results` section for examples.

To load a model for other applications, run
```python
model = load_model(MODEL_PATH, custom_objects=get_custom_objects())
```
Predictions on a given input can be obtained using `model.predict(INPUT)`, see the function `pred` for an example.
