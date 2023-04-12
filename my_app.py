import hydra
from omegaconf import DictConfig

@hydra.main(config_path ='config', config_name = 'config')
def my_app(cfg: DictConfig) -> None :
    print(cfg)
    print(hydra.utils.get_original_cwd())
    '''
    ref : https://hydra.cc/docs/advanced/instantiate_objects/overview/
    instantiate

    model
    datamodule
    optimizer
    trainer

    '''

if __name__ == '__main__' :
    my_app()