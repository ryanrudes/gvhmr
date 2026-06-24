from gvhmr.configs import parse_args_to_cfg, register_store_gvhmr
from gvhmr.utils.vis.rich_logger import print_cfg

if __name__ == "__main__":
    register_store_gvhmr()
    cfg = parse_args_to_cfg()
    print_cfg(cfg, use_rich=True)
