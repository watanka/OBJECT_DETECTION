import hydra
from omegaconf import DictConfig

@hydra.main(config_path ='config', config_name = 'config')
def my_app(cfg: DictConfig) -> None :
    print(cfg)

if __name__ == '__main__' :
    my_app()