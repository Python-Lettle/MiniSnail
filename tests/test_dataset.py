import numpy as np
from minisnail.config import SnailConfig
from minisnail.tokenizer import get_tokenizer
from minisnail.util import read_memmap_data

try:
    data = read_memmap_data('./data/valid_dataset.npy', dtype=np.int32)
    print("✅ 有效的数据文件")
    print(f"shape: {data.shape}, dtype: {data.dtype}")

    config = SnailConfig.from_json('./config.json')
    tokenizer = get_tokenizer(config)

    text = tokenizer.decode(data[:40])
    print(text)
    print(tokenizer.decode(data[-40:]))

except Exception as e:
    print(f"❌ 不是有效的数据文件: {e}")
