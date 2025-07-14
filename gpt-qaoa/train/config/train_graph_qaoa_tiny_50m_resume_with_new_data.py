out_dir = "out-graph-qaoa-50m-ft-nasdaq"
eval_interval = 250            # keep frequent because we'll overfit
eval_iters = 200
log_interval = 25               # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True  # False

wandb_log = False
wandb_project = "gpt-qaoa"
wandb_run_name = "out-graph-qaoa-50m-ft-nasdaq"

dataset = "graph_qaoa"
pad_token_id = 0

# Training batch config
gradient_accumulation_steps = 16 # try 32 or 64
batch_size = 16
block_size = 1024

# baby GPT model :)
# Model architecture ~50M params
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

# Optimizer
learning_rate = 3e-4
max_iters = 15_000
lr_decay_iters = 15_000
min_lr = 3e-5
beta2 = 0.95
warmup_iters = 2000

# Hardware
device = "cuda"
compile = True
