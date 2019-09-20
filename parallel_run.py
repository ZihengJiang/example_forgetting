import subprocess
import random


# run tasks on background
# write output to different files

cuda_cnt = 0
process = []
dataset = 'svhn'

for noise_labels in [30, 40, 50]:
    for remove_percent in [30, 40, 50]:
        for strategy in ['random',  'unforgettable', 'low-acc']:
            log_fname = '_'.join([dataset, str(noise_labels), str(remove_percent), strategy]) + '.log'
            with open('./logs/' + log_fname, 'w+') as fout:
                # sleep_time = random.randint(0, 5)
                # p = subprocess.Popen(['sleep', str(sleep_time)])
                # subprocess.Popen(['echo', str(noise_labels), str(remove_n), strategy], stdout=fout)
                p = subprocess.Popen(['python', 'main.py',
                                    '--output_dir', 'online_results',
                                    '--dataset', dataset,
                                    '--epochs', '200',
                                    '--data_augmentation',
                                    '--burn_in_epochs', '50',
                                    '--noise_percent_labels', str(noise_labels),
                                    '--remove_percent', str(remove_percent),
                                    '--remove_strategy', strategy,
                                    '--device', 'cuda:{0}'.format(cuda_cnt % 8),
                                    '--seed', '1',
                                    ], stdout=fout)
                cuda_cnt += 1
                process.append(p)
                if cuda_cnt % 24 == 0:
                    # wait process
                    print('use all gpus, wait tasks to be finished')
                    exit_codes = [p.wait() for p in process]
                    print(exit_codes)
                    # reset
                    cuda_cnt == 0
                    process = []

exit_codes = [p.wait() for p in process]
print(exit_codes)
