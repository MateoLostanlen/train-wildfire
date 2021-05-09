
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
            'values': [5, 10, 15]
            },
        'lr': {
            'values': [0.0004, 0.0006, 0.0008, 0.001, 0.003]
            },
        'wd': {
            'values': [ 0, 0.000001, 0.00001, 0.0001, 0.001]
            },
        'model_arch': {
            'values': [ 'rexnet1_3x', 'rexnet1_5x', 'rexnet2_0x']
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
            'value': '1'}
        ,
        'checkpoint': {
            'value': 'checkpoint.pth'}
    }) 

    return sweep_config