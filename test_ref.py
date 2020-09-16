from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from models.networks.pose_dla_dcn import get_ref_net as get_dla_ref_net


def main(opt):
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  #train_dataset = Dataset(opt, 'train')
  val_dataset = Dataset(opt, 'val')

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
 )


  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = get_dla_ref_net(num_layers=opt.arch.split("_")[-1], heads=opt.heads, vocab_size=val_dataset.vocab_size, head_conv=opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  _, preds = trainer.val(0, val_loader)
  val_loader.dataset.run_eval_ref(preds)

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
