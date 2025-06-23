import os
import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from concurrent.futures import ProcessPoolExecutor


INPUT_FILE = "bookcorpus_cleaned.txt"
OUTPUT_FILE = "bookcorpus_tokens.bin"
BATCH_SIZE = 1000
NUM_WORKERS = os.cpu_count() 


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def batched_iterator(file_path, batch_size):
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                batch.append(line)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch


def encode_lines(lines):
    token_ids = []
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        token_ids.extend(ids)
    return token_ids


if __name__ == "__main__":
    with open(OUTPUT_FILE, "wb") as fout:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []

            print(f"Launching parallel tokenization using {NUM_WORKERS} workers")
            for batch in batched_iterator(INPUT_FILE, BATCH_SIZE):
                futures.append(executor.submit(encode_lines, batch))

            for i, future in enumerate(tqdm(futures, desc="Tokenizing", unit="batch")):
                token_ids = future.result()
                np.array(token_ids, dtype=np.uint16).tofile(fout)

    print(f"\n Tokenization complete. Output written to: {OUTPUT_FILE}")
