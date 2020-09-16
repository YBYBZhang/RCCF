from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import sys
#sys.path.insert(0, "src/lib")
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
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  Dataset = get_dataset(opt.dataset, opt.task)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)

  train_dataset = Dataset(opt, 'train')
  val_dataset = Dataset(opt, 'val')

  print('Setting up data...')
  val_loader = torch.utils.data.DataLoader(
      val_dataset, 
      batch_size=1, 
      shuffle=False,
      num_workers=1,
      pin_memory=True
 )

  train_loader = torch.utils.data.DataLoader(
      train_dataset, 
      batch_size=opt.batch_size, 
      shuffle=True,
      num_workers=opt.num_workers,
      pin_memory=True,
      drop_last=True
  )

  logger = Logger(opt)

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  print('Creating model...')
  model = get_dla_ref_net(num_layers=opt.arch.split("_")[-1], heads=opt.heads, vocab_size=train_dataset.vocab_size, head_conv=opt.head_conv)
#   model = create_model(opt.arch, opt.heads, opt.head_conv)
  optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  start_epoch = 0
  best =  0
  if opt.load_model != '':
    model, optimizer, start_epoch, best = load_model(
      model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)

  Trainer = train_factory[opt.task]
  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

  if opt.test:
    _, preds = trainer.val(0, val_loader)
    val_loader.dataset.run_eval(preds, opt.save_dir)
    return
  

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
      save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                 epoch, model, optimizer)
      with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
        acc = val_loader.dataset.run_eval_ref(preds)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if acc >  best:
        #best = log_dict_val[opt.metric]
        best = acc
        save_model(os.path.join(opt.save_dir, 'model_best.pth'), 
                   epoch, model, best)

    save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                 epoch, model, optimizer, best)
    logger.write('\n')
    if epoch in opt.lr_step:
      #save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
      #           epoch, model, optimizer)
      lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
