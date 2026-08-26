import os
from typing import IO, Any, BinaryIO
import torch
import argparse

from minisnail.debug import console

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
) -> dict[str, Any]:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the checkpoint state.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
    Returns:
        dict[str, any]: A dictionary of the checkpoint state.
    """
    # Load the checkpoint from the file or object
    if isinstance(src, str) or isinstance(src, os.PathLike):
        src = open(src, 'rb')
    # Load the model state from the checkpoint
    # 注意: checkpoint 是从 GPU 模型保存的, 默认 map_location 会把张量加载回原设备(GPU),
    # 导致续训时同时占用一份 checkpoint 权重+优化器状态和新建模型, 显存翻倍直至 OOM。
    # 显式映射到 CPU, load_state_dict 会按需拷贝到模型所在设备。
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)

    return checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load pretrain model from checkpoint.')
    parser.add_argument('--checkpoint', type=str, default='./output/new_pretrain/checkpoint.pt', help='Path to the checkpoint file.')
    parser.add_argument('--output', type=str, default='./model/new_pretrain', help='Path to the output file.')
    args = parser.parse_args()
    
    console.print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    args.output = os.path.join(args.output, 'pretrain_lm_' + str(checkpoint['global_step']) + '.pt')
    torch.save(checkpoint['model_state_dict'], args.output)
    console.print(f"Pretrain model saved to {args.output}")