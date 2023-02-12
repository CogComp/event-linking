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
You can choose ``mode`` from [test, valid, train]. If you want to evaluate on New York Times data, just add ``--nyt``.

## Train EveLINK
If you want to train a new model, please follow the listed steps:

You first need to get the data ready by running:
```
python data_processing.py
```
If you want to use different data, please convert it to the same format of the data we provide. 

### Train Bi-Encoder
To train the bi-encoder, just run:
```
python blink/biencoder/train_biencoder.py --output_path "models/biencoder" --data_path "out/" --num_train_epochs 10 --learning_rate 1e-5 --bert_model "bert-large-uncased" --train_batch_size 64 --max_context_length 256 --max_cand_length 256 --data_parallel --print_interval 10000 --eval_interval 10000000
```

### Train Cross-Encoder
Before you train cross-encoder, you first need to use your trained Bi-Encoder to generate the training data. Please run:

```
python blink/biencoder/eval_biencoder.py --path_to_model "models/biencoder/pytorch_model.bin" --data_path "out/" --mode "valid" --bert_model "bert-large-uncased" --eval_batch_size 128 --encode_batch_size 200 --max_context_length 256 --max_cand_length 256 --output_path "out/" --cand_encode_path "models/all_entities.encoding" --cand_pool_path "models/cand_pool.t7" --save_topk_result --data_parallel
python event_linking_main.py --predict --top_k 30 --fast --mode train --save_topk_result
python event_linking_main.py --predict --top_k 30 --fast --mode valid --save_topk_result
```

After generating all the encodings and data, run:
```
python blink/crossencoder/train_cross.py --output_path "models/crossencoder" --data_path "out/" --num_train_epochs 10 --learning_rate 1e-5 --bert_model "bert-large-uncased" --train_batch_size 2 --eval_batch_size 64 --max_context_length 256 --max_cand_length 256 --max_seq_length 512 --add_linear --data_parallel --print_interval 10000 --eval_interval 10000000
```



