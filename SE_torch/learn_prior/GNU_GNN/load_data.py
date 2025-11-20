from SE_torch.net_preprocess.process_net_data import parse_ieee_mat, System


def load_data(config, num_samples):
    file = config.get('file')
    if file is None:
        raise(ValueError('Please specify a net file'))

    data = parse_ieee_mat(file)
    system_data = data['data']['system']
    sys = System(system_data)