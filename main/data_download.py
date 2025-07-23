from datasets import load_dataset

dataset = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1", 
                       split="train", cache_dir="./my_cache")