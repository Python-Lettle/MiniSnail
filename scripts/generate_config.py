from minisnail.config import SnailConfig
from minisnail.util import console

if __name__ == '__main__':
    config = SnailConfig()
    config.to_json("./config.json")
    console.print(f"Config saved to ./config.json")