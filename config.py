
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
            'values': [10, 20, 30, 40, 50]
            },
        'lr': {
            'values': [ 0.000001, 0.000003, 0.000005, 0.000007, 0.00001, 0.00003, 0.00007, 0.0001]
            },
        'wd': {
            'values': [ 0, 0.000001, 0.00001, 0.0001, 0.001]
            },
        'image_size': {
            'value': 448
            },
        'checkpointTF': {
            'values': [None]
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
        'architecture': {
            'value': 'rexnet1_0x'}
        ,
        'freeze': {
            'value': None}
        ,
        'train_ratio': {
            'value': 0.8}
        ,
        'checkpoint': {
            'value': 'checkpoint.pth'}
    }) 