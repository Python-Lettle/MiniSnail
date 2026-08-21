import argparse
from minisnail.dataset import PretrainDataset, get_dataloader
from minisnail.tokenizer import get_tokenizer
from minisnail.config import SnailConfig
from minisnail.util import setup_seed
from minisnail.debug import console

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.json')
    parser.add_argument('--data_path', type=str, default='tests/data/pretrain.jsonl')
    args = parser.parse_args()

    config = SnailConfig.from_json(args.config)
    tokenizer = get_tokenizer(config)
    setup_seed(config.system.seed)

    dataset = PretrainDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config.model.context_length,
    )

    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    console.print(dataset[0])