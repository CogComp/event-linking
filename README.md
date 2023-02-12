Here is the repo of our paper: [Event Linking: Grounding Event Mentions to Wikipedia](http://cogcomp.org/page/publication_view/996)

EveLINK is built based on the code of [BLINK](https://github.com/facebookresearch/BLINK). When using our code, please give enough credits to [BLINK](https://github.com/facebookresearch/BLINK).

## Create environment

```
conda create -n evelink python=3.8
conda activate evelink
pip install -r requirements.txt
``` 
You may need to change torch version based on your CUDA version.

## Data & models

You can download the data and the model [here].

## Run EveLINK

To run our model, you need to first process all the data by running:
```
python data_processing.py
```
Then, to link event mentions to Wikipedia, you need to run:
```
python event_linking_main.py --predict --topk 100 --mode test
```
The output file will be saved in [out] folder, and you can evaluate the accuracy by running:
```
python event_linking_main.py --evaluate --topk 100 --mode test
```
You can choose ``mode`` from [test, valid, train]. 
