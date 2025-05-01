import json


def save_results(device, res_dict):
    file_name = './results/{}.json'.format(device)

    with open(file_name, 'w') as f:
        json.dump(res_dict, f)

