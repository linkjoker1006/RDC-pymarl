# Project Overview
The core source code is located in the `src` directory. After successful execution, a `result` directory will be generated in the root folder to store experiment outputs. Our code is developed based on [pymarl3](https://github.com/tjuHaoXiaotian/pymarl3) and [epymarl](https://github.com/uoe-agents/epymarl) code bases, and the installation of dependencies can refer to the tutorials provided by them.

## Training Scripts
- `train_mpe_qmix.sh`, ​`train_smac_qmix.sh`, `train_mpe_vdn.sh`, `train_smac_vdn.sh`:  
  Scripts for training ​**QMIX** and ​**VDN** agents on [MPE](https://github.com/Farama-Foundation/PettingZoo) and [SMAC](https://github.com/oxwhirl/smac) environments.  
  *Note: Baseline algorithms (e.g., MAPPO) and the benchmark [SMACv2](https://github.com/oxwhirl/smacv2) are also supported but not discussed in the paper.*

## Testing Scripts
- ​`test_f.sh`: Batch testing with ​**fixed delays**.  
- ​`test_uf.sh`: Batch testing with ​**unfixed delays**.  
- ​`test_history_length.sh`: Tests varying **input history lengths** under **fixed delays**.  
  *Note: Modify parameters in these scripts as needed for your use case.*  

- ​**Other `test_*.sh` files**:  
  Contain pre-configured test commands. ​Replace the `checkpoint_path` in these scripts with your trained model checkpoints before execution. If you need to print the detailed results during the test, please remove the note on the corresponding position in file `RDC-pymarl\src\runners\parallel_runner.py`.  
  ```python
  cur_returns.extend(episode_returns)

  # test with chunksize
  # chunksize = 32
  # if cur_stats['n_episodes'] % chunksize == 0:
  #     print(len(self.test_returns))
  #     if cur_stats['n_episodes'] / chunksize == 1:
  #         if self.args.env in ["sc2", "sc2_v2"]:
  #             print("test_battle_won: {} return_mean: {}".format(cur_stats['battle_won'] / chunksize, np.mean(cur_returns)))
  #         elif self.args.env == "gymma":
  #             print("return_mean: {}".format(np.mean(cur_returns)))
  #         self.last_test_stats = copy.deepcopy(cur_stats)
  #     else:
  #         if self.args.env in ["sc2", "sc2_v2"]:
  #             print("test_battle_won: {} return_mean: {}".format((cur_stats['battle_won'] - self.last_test_stats['battle_won']) / chunksize, np.mean(cur_returns[-chunksize:])))
  #         elif self.args.env == "gymma":
  #             print("return_mean: {}".format(np.mean(cur_returns[-chunksize:])))
  #         self.last_test_stats = copy.deepcopy(cur_stats)

  n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size

  ...other code...

  stats.clear()
  # print(self.logger.stats)
  ```

## Configurations
- `src/config` contains all the environment and algorithm super parameter configurations. Please note that the Echo and Flash compensators in the paper correspond to the situation where the **predictor_mode** is **single_step** and **multi_step**, respectively.

## Results Directory
The `result` directory (auto-generated after training/testing) includes:
- `model`: Model checkpoints
- `sacred`: Sacred experiment outputs
- `tb_logs`: Tensorboard logs

## Usage
1. Train a model:
   ```bash
   CUDA_VISIBLE_DEVICES="0" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" delay_aware=True cheating_start_value=1.0 cheating_end_value=1.0 exp_name="TAG-QMIX-B" >> tag_qmix_b.log 2>&1 &
   ```
2. Test a model:
   ```bash
   sh ./test_f.sh 'CUDA_VISIBLE_DEVICES="0" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_b.log'
   sh ./test_uf.sh 'CUDA_VISIBLE_DEVICES="0" nohup python -u src/main.py --config=pd_qmix_gru4mpe --env-config=gymma with env_args.key="pz-mpe-simple-tag-v3" env_args.pretrained_wrapper="PretrainedTag" predictor_mode="none" exp_name="TAG-QMIX-B" test_nepisode=1280 evaluate=True checkpoint_path=""' 'tag_test_qmix_b.log'
   ```
