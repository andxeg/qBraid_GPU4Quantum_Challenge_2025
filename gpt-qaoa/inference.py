from contextlib import nullcontext
import importlib
import os
import pathlib
import pickle
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import networkx
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

# Must be called early, before model creation or torch.compile
torch.set_float32_matmul_precision('high')

# import model_qaoa
from model_qaoa import GPT, GPTConfig

# import model_qaoa_cached
from model_qaoa_cached import GPT as GPT_cached, GPTConfig as GPTConfig_cached
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# Dict with paths to checkpoints and meta files
MODELS_INFO = {
    "20m_new": {
        "ckpt_path": "./checkpoints/ckpt_20m_new.pt",
        "meta_path": "./data/train/meta_20_new.pkl",
        "graph_tokenizer": "graph_to_tokens_v1",
    },
    "50m_old": {
        "ckpt_path": "./checkpoints/ckpt_50m_old.pt",
        "meta_path": "./data/train/meta_50m_old.pkl",
        "graph_tokenizer": "graph_to_tokens_old_format",
    },
    "50m_new": {
        "ckpt_path": os.path.join(THIS_DIR, "./checkpoints/ckpt_50m_new.pt"),
        "meta_path": os.path.join(THIS_DIR, "./data/train/meta_50m_new.pkl"),
        "graph_tokenizer": "graph_to_tokens_v1",
    },
}


def load_graphs(file: pathlib.Path) -> List[nx.Graph]:
    with open(file, "rb") as f:
        graphs = pickle.load(f)
    return graphs


def graph_to_tokens_old_format(graph: nx.Graph) -> List[str]:
    """
        Compound tokens like '(0, 1)' and '[0 1]'.
        And 2-decimal precision for float numbers.
        Graphs only with edges weights.
    """
    bos_token = "<bos>"
    end_of_graph_token = "<end_of_graph>"
    graph_tokens = []
    graph_tokens.append(bos_token)
    for u, v in graph.edges:
        graph_tokens.append(f"({u},{v})")
        graph_tokens.append(f"{graph.edges[u, v]['weight']:.2f}") # 2-decimal precision
    graph_tokens.append(end_of_graph_token)    
    return graph_tokens


def graph_to_tokens_v1(graph: nx.Graph) -> List[str]:
    """
        Compound tokens like '(0, 1)' and '[0 1]'.
        And 2-decimal precision for float numbers.
        Graphs with nodes and edges weights.
    """
    bos_token = "<bos>"
    end_of_graph_token = "<end_of_graph>"
    nodes_weight_start = "<node_weights_start>"
    nodes_weight_end = "<node_weights_end>"
    graph_tokens = []
    graph_tokens.append(bos_token)
    graph_tokens.append("<format_v1>")

    # nodes
    graph_tokens.append(nodes_weight_start)
    for u in graph.nodes:
        if "return_" in graph.nodes[u]:
            graph_tokens.append(f"{graph.nodes[u]['return_']:.2f}") # 2-decimal precision
        elif "mu" in graph.nodes[u]:
            graph_tokens.append(f"{graph.nodes[u]['mu']:.2f}") # 2-decimal precision
        else:
            raise Exception(f"Cannot find return value using key 'mu' and 'return_'.")
    
    graph_tokens.append(nodes_weight_end)
    
    # edges
    for u, v in graph.edges:
        graph_tokens.append(f"({u},{v})")
        graph_tokens.append(f"{graph.edges[u, v]['weight']:.2f}") # 2-decimal precision
    graph_tokens.append(end_of_graph_token)    
    return graph_tokens


def generate_long_circuit_cpu(
    model,
    config,
    graph_tokens: List[str],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_total_tokens: int = 6000,
    temperature: float = 1.0,
    top_k: int = 10
) -> List[str]:
    model.eval()
    pad_id = config.pad_token_id
    stop_id = config.stop_token_id
    block_size = config.block_size
    device = next(model.parameters()).device

    # === Encode input graph ===
    input_ids = [stoi.get(tok, stoi["<unk>"]) for tok in graph_tokens]
    generated_ids = input_ids[:]

    while len(generated_ids) < max_total_tokens:
        # Prepare input window: last block_size tokens
        idx_cond = torch.tensor([generated_ids[-block_size:]], dtype=torch.long).to(device)
        if (idx_cond == pad_id).all():
            print("Context is only padding. Stopping early.")
            break

        with torch.no_grad():
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)

                # Apply top-k mask, but fallback if all logits would be -inf
                threshold = v[:, [-1]]  # shape: (B, 1)
                logits_masked = logits.clone()
                logits_masked[logits < threshold] = -float("Inf")

                if torch.isinf(logits_masked).all():
                    # fallback: don't mask at all
                    print("All logits would be masked. Falling back to unfiltered logits.")
                else:
                    logits = logits_masked

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("NaN or Inf detected in logits. Dumping logits:")
                print(logits)
                raise RuntimeError("NaNs or Infs in logits")

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError("Invalid probabilities: contains NaN or Inf after softmax.")
            next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)

        # Stop if <end_of_circuit> is generated
        if stop_id is not None and next_id == stop_id:
            break

    # === Decode circuit tokens (excluding graph) ===
    generated_tokens = [itos[i] for i in generated_ids]
    for stop_token in ("<end_of_circuit>", "<pad>"):
        if stop_token in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(stop_token)]
    generated_circuit_tokens = generated_tokens[len(input_ids):]
    return generated_circuit_tokens


def generate_long_circuit_cpu_cached(
    model,
    config,
    graph_tokens: List[str],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_total_tokens: int = 6000,
    temperature: float = 1.0,
    top_k: int = 10
) -> List[str]:
    pad_id = config.pad_token_id
    stop_id = config.stop_token_id
    block_size = config.block_size
    device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    print(f"device: {device}")

    # Encode input graph
    input_ids = [stoi.get(tok, stoi["<unk>"]) for tok in graph_tokens]
    generated_ids = input_ids[:]

    # Track state across tokens
    past_key_values = None
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Prime the cache with the full input graph
    with torch.no_grad():
        logits, _, past_key_values = model(input_tensor, use_cache=True)

    while len(generated_ids) < max_total_tokens:
        idx_cond = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _, past_key_values = model(
                idx_cond, past_key_values=past_key_values, use_cache=True
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0].item()

        generated_ids.append(next_id)

        if stop_id is not None and next_id == stop_id:
            break

    # === Decode output ===
    generated_tokens = [itos[i] for i in generated_ids]
    for stop_token in ("<end_of_circuit>", "<pad>"):
        if stop_token in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(stop_token)]

    return generated_tokens[len(input_ids):]


def generate_long_circuit_compiled(
    model,
    config,
    graph_tokens: List[str],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_total_tokens: int = 6000,
    temperature: float = 1.0,
    top_k: int = 10
) -> List[str]:
    pad_id = config.pad_token_id
    stop_id = config.stop_token_id
    block_size = config.block_size
    device = next(model.parameters()).device

    print(f"device: {device}")

    # Encode input graph
    input_ids = [stoi.get(tok, stoi["<unk>"]) for tok in graph_tokens]
    generated_ids = input_ids[:]

    while len(generated_ids) < max_total_tokens:
        # Take last `block_size` tokens as context
        idx_cond = torch.tensor([generated_ids[-block_size:]], dtype=torch.long, device=device)

        # Optional: stop early if all pad tokens (unlikely with real data)
        if torch.all(idx_cond == pad_id):
            break

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]  # only last token
        logits /= temperature

        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            threshold = v[:, [-1]]
            logits_masked = logits.clone()
            logits_masked[logits < threshold] = -float("Inf")

            # Avoid fallback logic for performance; assume model is healthy
            logits = logits_masked

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)[0, 0].item()
        generated_ids.append(next_id)

        # Stop if end token generated
        if stop_id is not None and next_id == stop_id:
            break

    # Decode tokens, skip graph
    generated_tokens = [itos[i] for i in generated_ids]
    for stop_token in ("<end_of_circuit>", "<pad>"):
        if stop_token in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(stop_token)]

    return generated_tokens[len(input_ids):]


def generate_long_circuit_compiled_cached(
    model,
    config,
    graph_tokens: List[str],
    stoi: Dict[str, int],
    itos: Dict[int, str],
    max_total_tokens: int = 6000,
    temperature: float = 1.0,
    top_k: int = 10
) -> List[str]:
    pad_id = config.pad_token_id
    stop_id = config.stop_token_id
    block_size = config.block_size
    device = next(model.parameters()).device

    print(f"device: {device}")

    # Encode input graph
    input_ids = [stoi.get(tok, stoi["<unk>"]) for tok in graph_tokens]
    generated_ids = input_ids[:]

    # Track state across tokens
    past_key_values = None
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Prime the cache with the full input graph
    with torch.no_grad():
        logits, _, past_key_values = model(input_tensor, use_cache=True)

    while len(generated_ids) < max_total_tokens:
        idx_cond = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, _, past_key_values = model(
                idx_cond, past_key_values=past_key_values, use_cache=True
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)[0, 0].item()

        generated_ids.append(next_id)

        if stop_id is not None and next_id == stop_id:
            break

    # === Decode output ===
    generated_tokens = [itos[i] for i in generated_ids]
    for stop_token in ("<end_of_circuit>", "<pad>"):
        if stop_token in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(stop_token)]

    return generated_tokens[len(input_ids):]


def warmup_model(model, config, input_len: int, device: str = "cuda"):
    """Run a warm-up pass with fixed input length to compile the graph once."""
    dummy_input = torch.randint(0, config.vocab_size, (1, input_len), device=device)
    with torch.inference_mode():
        _ = model(dummy_input, use_cache=True)


def generate_batch(
    graphs_batch,
    model,
    config,
    stoi,
    itos,
    graph_tokenizer,
    max_total_tokens: int = 7000,
    cached: bool = True,
    device: str = "cuda"
):
    if cached:
        print("Warm-up model")
        start_t = time.time()
        warmup_model(model, config, input_len=len(graph_tokenizer(graphs_batch[0])), device=device)
        end_t = time.time()
        print(f"Elapsed time for warm-up: {end_t - start_t} secs")

    # choose inference function according to parameters
    generate_long_circuit_func = None
    if device == "cpu" and not cached:
        generate_long_circuit_func = generate_long_circuit_cpu
    elif device == "cpu" and cached:
        generate_long_circuit_func = generate_long_circuit_cpu_cached
    elif device == "cuda" and not cached:
        generate_long_circuit_func = generate_long_circuit_compiled
    elif device == "cuda" and cached:
        generate_long_circuit_func = generate_long_circuit_compiled_cached

    times = []
    results = []
    for graph in tqdm.tqdm(graphs_batch):
        graph_tokens = graph_tokenizer(graph)
        start_t = time.time()
        generated_circuit_tokens = generate_long_circuit_func(
            model,
            config,
            graph_tokens,
            stoi,
            itos,
            max_total_tokens=max_total_tokens,
            temperature=0.5,
            top_k=None
        )
        end_t = time.time()
        gen_t = end_t - start_t
        times.append(gen_t)
        results.append(generated_circuit_tokens)
        print(f"Generation time: {gen_t} secs, generated QAOA Circuit length: {len(generated_circuit_tokens)}")

    return results, times


def load_meta_info(meta_path: str) -> Tuple[Dict[Any, Any], Dict[str, int], Dict[int, str]]:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    return meta, stoi, itos 


def load_model(
    ckpt_path: str,
    meta: Dict[Any, Any],
    device: str,
    compile: bool,
    cached: bool
):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if cached:        
        gptconf = GPTConfig_cached(**checkpoint["model_args"])
        gptconf.stop_token_id = meta["stop_token_id"]
        model = GPT_cached(gptconf)
    else:
        gptconf = GPTConfig(**checkpoint["model_args"])
        gptconf.stop_token_id = meta["stop_token_id"]
        model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    if device == "cuda":
        print("Model on GPU")
        model = model.to(torch.device("cuda"))
        model = model.to(dtype=torch.bfloat16) # optional, for speed
        if compile:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)
            model.eval()
            print("model was compiled successfully")
    else:
        print("Model on CPU")
        model.eval()
        model.to(device)
    return model, gptconf


def inference(
    graphs_batch: List[nx.Graph],
    model_id: str,
    cached: bool = True,
    device: str = "cuda"
):
    assert model_id in MODELS_INFO, f"Invalid {model_id=}"
    model_params = MODELS_INFO[model_id]
    
    # Parameters
    ckpt_path = model_params["ckpt_path"]
    meta_path = model_params["meta_path"]
    graph_tokenizer = eval(model_params["graph_tokenizer"])
    init_from = "resume"
    seed = 1337

    if device == "cuda":
        dtype = "float32"
        compile_model = True
    elif device == "cpu":
        dtype = "float32"
        compile_model = False
    else:
        raise Exception(f"Invalid {device=}")

    # Prepare torch and device
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu" # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    # Load meta information: vocabulary, token to id dict, id to token dict, etc
    print(f"Loading meta from {meta_path}...")
    meta, stoi, itos = load_meta_info(meta_path)
    print(f'found vocab_size = {len(meta["stoi"])}')

    # Load model
    print(f"Loading model from {ckpt_path}...")
    model, config = load_model(ckpt_path, meta, device, compile_model, cached)
    print(f"Model with {cached=} was successfully loaded:\n{model}")

    # Run generation
    generated_circuits, times = generate_batch(
        graphs_batch,
        model,
        config,
        stoi,
        itos,
        graph_tokenizer,
        cached=cached,
        device=device
    )

    return generated_circuits, times


def save_seq_tokens_to_file(tokens: List[str], filename: str) -> None:
    with open(filename, "wb") as file:
        pickle.dump(tokens, file)


def load_seq_tokens_from_file(filename: str) -> List[str]:
    with open(filename, "rb") as file:
        loaded_list = pickle.load(file)
    return loaded_list


def save_generated_circuits(generated_circuits: List[List[str]], debug: bool = False) -> None:
    filename_gen = f"generated_circuit_tokens_amount_{len(generated_circuits)}.pkl"
    save_seq_tokens_to_file(generated_circuits, filename_gen)
    generated_circuit_loaded = load_seq_tokens_from_file(filename_gen)

    if debug:
        for idx, circuit in enumerate(generated_circuits):
            print("-" * 40)
            print(f"Circuit #{idx + 1}. First 20 tokens: {circuit[:20]}")
            print("-" * 40)

    # Check if everything saved correctly
    for c_gen, c_gen_loaded in zip(generated_circuits, generated_circuit_loaded):
        if not any([e1 == e2 for e1, e2 in zip(c_gen, c_gen_loaded)]):
            raise Exception("Something was wrong during saving")


def save_generation_times(generation_times: List[float], debug: bool = False) -> None:
    times_numpy_array = np.array(generation_times)

    if debug:
        print(f"Generation times: {generation_times}")

    filename = "generation_times.pkl"
    with open(filename, "wb") as file:
        pickle.dump(times_numpy_array, file)


def save_results(
    generated_circuits: List[List[str]],
    generation_times: List[float],
    debug: bool = False
) -> None:
    save_generated_circuits(generated_circuits, debug=debug)
    save_generation_times(generation_times, debug=debug)


if __name__ == "__main__":
    model_id = "50m_old"
    device = "cuda"
    cached = True
    limit = 2
    debug = True # partially print each generated circuit
    graphs_filename = "./data/graph_and_circuits/random_graphs_for_testing.pkl"

    print("Starting to load graphs")
    graphs_batch = load_graphs(graphs_filename)
    print("Graphs were successfully loaded")

    graphs_batch = graphs_batch[:limit]
    generated_circuits, generation_times = inference(graphs_batch, model_id, cached=cached, device=device)

    save_results(generated_circuits, generation_times, debug=debug)
