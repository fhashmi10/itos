"""Model config in json format"""

CFG = {
    "data": {
        "image_path": "data/Images/",
        "captions_file": "data/captions.txt"
    },
    "tokenize": {
        "vocab_size": 5000
    },
    "train": {
        "batch_size": 64,
        "buffer_size": 1000,
        "epoches": 10
    },
    "model": {
        "embedding_dim":256,
        "units":512
    }
}
