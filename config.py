
def get_config():
    sweep_config = {
        'method': 'random'
        }

    metric = {
        'name': 'best_val_loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
    parameters_dict = {
        'epochs': {
            'values': [10]
            },
        'lr': {
            'values': [0.0001, 0.0003, 0.0007, 0.001]
            },
        'wd': {
            'values': [ 0, 0.000001, 0.00001, 0.0001, 0.001]
            },
        'model_arch': {
            'values': [ 'rexnet1_0x', 'rexnet1_3x', 'darknet19']
            },
        'image_size': {
            'value': 448
            }

            
        }

    sweep_config['parameters'] = parameters_dict
    parameters_dict.update({
        'classes': {
            'value': 1}
        ,
        'batch_size': {
            'value': 32}
        ,
        'freeze': {
            'value': None}
        ,
        'checkpoint': {
            'value': 'checkpoint.pth'}
    }) 

    return sweep_config
