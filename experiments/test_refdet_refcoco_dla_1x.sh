#cd src
# train
python test_ref.py refdet --resume --test --keep_res  --exp_id coco_dla_1x --batch_size 1 --master_batch 16 --lr 5e-4 --gpus 1 --num_workers 2 --dense_wh --data_json data/coco/refcoco_unc/data.json --data_h5 data/coco/refcoco_unc/data.h5 --coco_json data/coco/annotations/instances_train2014.json
#python main_ref.py refdet --resume --test --exp_id coco_dla_1x --batch_size 32 --master_batch 16 --lr 5e-4 --gpus 0,1 --num_workers 4 --dense_wh --data_json data/coco/refcoco_unc/data.json --data_h5 data/coco/refcoco_unc/data.h5 --coco_json data/coco/annotations/instances_train2014.json
# test
##python test.py refdet --exp_id coco_dla_1x --keep_res --resume
# # flip test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test 
# # multi scale test
# python test.py ctdet --exp_id coco_dla_1x --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
#cd ..
