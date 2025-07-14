out_dir = "out-graph-qaoa-50m"
eval_interval = 500            # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10               # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False  # False

wandb_log = False              # override via command line if you like
wandb_project = "gpt-qaoa"
wandb_run_name = "graph-qaoa-50m"

dataset = "graph_qaoa"
pad_token_id = 0               # matches token_to_id["<pad>"]

# Training batch config
gradient_accumulation_steps = 16 # try 4 also
batch_size = 16                 # try 1 for large model, 2, 4, 8 or 16 if training remains stable
block_size = 1024              # must match T used during window generation

# baby GPT model :)
# Model architecture ~50M params
n_layer = 8   # 4 or 6 - increase if model stable      # more layers to model long dependencies
n_head = 8    # 4 or 6      # should divide n_embd evenly
n_embd = 512   # 256 or 384 - increase if model stable     # larger embedding space for more tokens
dropout = 0.2

learning_rate = 3e-4 # 3e-4 # 5e-4            # with baby networks can afford to go a bit higher
max_iters = 150_000
lr_decay_iters = 150_000           # make equal to max_iters usually
min_lr = 3e-5 # 3e-5 # small model # 5e-5 # large model                   # learning_rate / 10 usually
beta2 = 0.95                    # make a bit bigger because number of tokens per iter is small

warmup_iters = 6000              # not super necessary potentially

# Hardware
device = "cuda"                  # run on mps
compile = True                 # do not torch compile the model
