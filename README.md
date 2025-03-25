# ST-PLD
The code of paper "Discovering New Intents via Spatio-Temporal Pseudo-Label Denoising"

## Get Started
### Requirement
Install all required libraries:
```
pip install -r requirements.txt
```

### Model Preparation
Down the pytorch bert model, or convert tensorflow param yourself as follow:
```
export BERT_BASE_DIR=/users/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
Set BERT path in the ./configs/config_STPLD.py

### Run
Run the experiments by:
```
bash run_STPLD.sh
```
You can change the parameters in the script. The selected parameters are as follows:
```
dataset: banking｜clinc | stackoverflow
known_class_ratio: 0.25 | 0.5 | 0.75
```
