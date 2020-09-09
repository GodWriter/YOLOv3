def parse_model_config(path):
    # 打开yolov3.config
    file = open(path, 'r')

    # 读取每一行，去除#，空格等字符
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]

    # 定义module_defs用于保存网络结构，由词典构建
    module_defs = []
    for line in lines:
        # 若是以[为首，说明网络某个子结构初始
        if line.startswith('['):
            # 每遇到一个[，就定一个词典加入module_defs
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()

            # 由于某些网络子结构不包括BN，为保证后续网络构建统一，故BN都初始化为0
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            # 将每行操作由=划分，并去除空格后加入module_defs中
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    options = dict()

    options['gpus'] = '0, 1, 2, 3'
    options['num_workers'] = '10'

    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options
