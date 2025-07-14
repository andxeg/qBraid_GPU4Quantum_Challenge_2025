out_dir = "out-graph-qaoa-20m"
eval_interval = 500            # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10               # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False  # False

wandb_log = False              # override via command line if you like
wandb_project = "gpt-qaoa"
wandb_run_name = "graph-qaoa-20m"

dataset = "graph_qaoa"
pad_token_id = 0                 # matches token_to_id["<pad>"]

# Training batch config
gradient_accumulation_steps = 8  # Scale gradient_accumulation_steps up/down to fit GPU memory. Use 10-12 only if training is unstable or too noisy.
batch_size = 32                  # try 1 for large model, 2, 4, 8 or 16 if training remains stable
block_size = 1024                # must match T used during window generation

# Model architecture ~19M params
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Optimizer settings
learning_rate = 3e-4
max_iters = 90_000
lr_decay_iters = 90_000 # make equal to max_iters usually
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 2000

# Hardware
device = "cuda"                  # run on mps
compile = True                 # do not torch compile the model
