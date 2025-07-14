# GPT-QAOA Inference

## Prerequisites

Create Python (>=3.10+) virtual environment and install all requirements:
```bash
$ python -m venv venv
$ . ./venv/bin/activate
$ python -m pip install -r requirements.txt
```

## Train
GPT model based on [nanoGPT](https://github.com/karpathy/nanoGPT) repository, but with significantly changes:
- add custom padding attention,
- add caching KV for fast inference.

Example of training data you can find in `train/data/graph_qaoa`. Config you can find in `config`.

Run training process:
```bash
python train_qaoa.py config/train_graph_qaoa_tiny_50m.py --device=cuda --compile=True --wandb_log=True
```

## Inference

Run the following script to generate a circuit:
```bash
python inference.py
```

It can take up to 30 second for GPU and 300 seconds for CPU.

## Results

After the script finishes running, you will find two files:
- `generated_circuit_tokens_amount_<the_number_of_circuits>.pkl` - contains the generated circuit,
- `generation_times.pkl` - contains the generation time for each circuit.
