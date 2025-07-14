"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import random
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_qaoa import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# device = "mps" # for mac only
dtype = "float32" # for mac only
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join("data", dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data_npz = np.load(os.path.join(data_dir, "train.npz"), mmap_mode="r")
    else:
        data_npz = np.load(os.path.join(data_dir, "val.npz"), mmap_mode="r")

    data = data_npz["x"]    # memmap array of shape (N, T)
    targets = data_npz["y"] # memmap array of shape (N, T)

    N = data.shape[0]
    ix = np.random.randint(0, N, size=batch_size)

    # Extract only the batch from memmap, cast just this to tensors
    x = torch.tensor(data[ix], dtype=torch.long)
    y = torch.tensor(targets[ix], dtype=torch.long)

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True) # NOTE! maybe set pin_memory=True
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


# dataloader for large dataset
class NPZDataset:
    def __init__(self, data_dir, device="cpu", split="train", batch_size=1024):
        self.data_dir = data_dir
        self.device = device
        self.split = split
        self.batch_size = batch_size
        self.files = sorted([f for f in os.listdir(data_dir) if f.startswith(split) and f.endswith(".npz")])
        self.data = None
        self.targets = None
        self.current_file_idx = 0
        self.load_next_chunk()

    def update_current_file_idx(self):
        self.current_file_idx  = (self.current_file_idx + 1) % len(self.files)

    def load_random_chunk(self):
        path = os.path.join(self.data_dir, random.choice(self.files))
        self.load_chunk(path)

    def load_next_chunk(self):
        path = os.path.join(self.data_dir, self.files[self.current_file_idx])
        self.load_chunk(path)
        self.update_current_file_idx()

    def load_chunk(self, path: str):
        print(f"[{self.split}] Loading chunk: {path}")
        data_npz = np.load(path, mmap_mode='r')
        self.data = data_npz["x"]    # memmap array of shape (N, T)
        self.targets = data_npz["y"] # memmap array of shape (N, T)

    def get_batch(self):
        N = self.data.shape[0]
        ix = np.random.randint(0, N, size=self.batch_size)

        # Extract only the batch from memmap, cast just this to tensors
        x = torch.tensor(self.data[ix], dtype=torch.long)    # load data eagerly from the memmap file into RAM
        y = torch.tensor(self.targets[ix], dtype=torch.long) # load data eagerly from the memmap file into RAM
        
        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(self.device, non_blocking=True) # NOTE! maybe set pin_memory=True
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
pad_token_id = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    pad_token_id = meta.get("pad_token_id", 0)
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    model_args['pad_token_id'] = pad_token_id
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line

    # NOTE!
    # new code if resume with different vocab size
    # Overwrite vocab_size with new size
    model_args['vocab_size'] = meta_vocab_size  # e.g., 55045
    model_args['pad_token_id'] = pad_token_id

    # old code
    # for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    #     model_args[k] = checkpoint_model_args[k]

    # new code
    # Force other attributes to match
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        model_args[k] = checkpoint_model_args[k]
    
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # NOTE!
    # new code if resume with different vocab size
    # --- Filter incompatible keys ---
    # due to the change in meta.pkl: vocab size and etc
    new_state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k not in new_state_dict or state_dict[k].shape != new_state_dict[k].shape:
            print(f"Skipping loading weight for: {k} (shape mismatch)")
            del state_dict[k]

    # --- Patch embedding and output head weights by merging old and new ---

    emb_name = 'transformer.wte.weight'   # change if your embedding layer key is different
    lm_head_name = 'lm_head.weight'       # change if your output head key is different

    if emb_name in checkpoint['model'] and emb_name not in state_dict:
        old_emb = checkpoint['model'][emb_name]
        new_emb = model.state_dict()[emb_name].clone()

        n_old_tokens = old_emb.shape[0]
        new_emb[:n_old_tokens, :] = old_emb

        state_dict[emb_name] = new_emb
        print(f"Patched embedding weights: copied {n_old_tokens} old tokens embeddings")

    if lm_head_name in checkpoint['model'] and lm_head_name not in state_dict:
        old_head = checkpoint['model'][lm_head_name]
        new_head = model.state_dict()[lm_head_name].clone()

        n_old_tokens = old_head.shape[0]
        new_head[:n_old_tokens, :] = old_head

        state_dict[lm_head_name] = new_head
        print(f"Patched lm_head weights: copied {n_old_tokens} old tokens weights")

    model.load_state_dict(state_dict, strict=False)
    
    # model.load_state_dict(state_dict) # leave this command if vocab size wasn't changed
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    # optimizer.load_state_dict(checkpoint['optimizer'])

    # NOTE!
    # new code if resume with different vocab size
    # Create a new optimizer fresh, do NOT load old optimizer state
    # In case if vocab size was changed
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(train_data, val_data):
    out = {}
    model.eval()
    for dataset, name in zip([train_data, val_data], ["train", "val"]):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
# if want to start new wandb run
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


# if want to resume, first find wanddb run id - see run url
# if wandb_log and master_process and init_from == 'resume':
#     import wandb
#     wandb_run_id = "jegc8hck"  # Replace this with your actual run ID
#     wandb.init(
#         project=wandb_project,
#         name=wandb_run_name,
#         id=wandb_run_id,
#         resume="must",
#         config=config
#     )

# training loop
# X, Y = get_batch("train") # fetch the very first batch NOTE! this is old get_batch

print("Before read train and val datasets.")

train_data = NPZDataset(data_dir, device=device, split="train", batch_size=batch_size) # NOTE! this is new get_batch
val_data = NPZDataset(data_dir, device=device, split="val", batch_size=batch_size)     # NOTE! this is new get_batch
X, Y = train_data.get_batch()
print(f"Train datasets: {train_data.files}")
print(f"Val datasets: {val_data.files}")


t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

print("Before train loop.")
# chunk_switch_interval = 10000 # should be at least 5000
# chunk_switch_interval = 5000 # use for fine-tuning
chunk_switch_interval = 3000 # use for fast fine-tuning
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss(train_data, val_data)
        train_chunk_index = train_data.current_file_idx
        val_chunk_index = val_data.current_file_idx
        print(f"step {iter_num}: train loss {losses['train']:.4f}, train chunk {train_chunk_index}, val loss {losses['val']:.4f}, val chunk {val_chunk_index}")
        if wandb_log:
            # wandb.log({
            #     "iter": iter_num,
            #     "train/loss": losses['train'],
            #     "val/loss": losses['val'],
            #     "lr": lr,
            #     "mfu": running_mfu*100, # convert to percentage
            # })

            val_chunk_key = f"val/loss_chunk_{val_chunk_index}"
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                val_chunk_key: losses['val'],
                "lr": lr,
                "mfu": running_mfu * 100,
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f"ckpt_it_{iter_num}_t_{int(time.time())}_val_loss_{losses['val']}.pt"))
                # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        # X, Y = get_batch("train") # NOTE! this is old get_batch
        X, Y = train_data.get_batch() # NOTE! this is new get_batch
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

    if iter_num != 0 and iter_num % chunk_switch_interval == 0:
        # train_data.load_random_chunk()
        # val_data.load_random_chunk()
        train_data.load_next_chunk()
        val_data.load_next_chunk()

if ddp:
    destroy_process_group()
