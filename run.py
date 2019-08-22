import subprocess
import random


# run tasks on background
# write output to different files

cuda_cnt = 0
process = []

for noise_labels in [30, 40, 50, 60, 70]:
    for remove_n in [15000, 20000, 25000, 30000, 35000]:
        for strategy in ['normal', 'random', 'unforgettable']:
            log_fname = '_'.join([str(noise_labels), str(remove_n), strategy]) + '.log'
            with open('./logs/' + log_fname, 'w+') as fout:
                # sleep_time = random.randint(0, 5)
                # p = subprocess.Popen(['sleep', str(sleep_time)])
                # subprocess.Popen(['echo', str(noise_labels), str(remove_n), strategy], stdout=fout)
                p = subprocess.Popen(['python', 'main.py',
                                    '--output_dir', 'online_results',
                                    '--dataset', 'cifar10',
                                    '--epochs', '200',
                                    '--data_augmentation',
                                    '--mode', 'online',
                                    '--burn_in_epochs', '50',
                                    '--noise_percent_labels', str(noise_labels),
                                    '--remove_n', str(remove_n),
                                    '--remove_strategy', strategy,
                                    '--device', 'cuda:{0}'.format(cuda_cnt % 8),
                                    '--seed', '1',
                                    ], stdout=fout)
                cuda_cnt += 1
                process.append(p)
                if cuda_cnt % 16 == 0:
                    # wait process
                    print('use all gpus, wait tasks to be finished')
                    exit_codes = [p.wait() for p in process]
                    print(exit_codes)
                    # reset
                    cuda_cnt == 0
                    process = []
