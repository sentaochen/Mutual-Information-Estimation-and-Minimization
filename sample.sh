python demo.py  --net resnet  --dataset PACS --num_class 7 --source art_painting cartoon photo --target sketch --gpu 0 --seed 0 | tee resnet[art_painting_cartoon_photo]Tosketch_seed0.log
python demo.py  --net resnet  --dataset PACS --num_class 7 --source art_painting cartoon sketch --target photo --gpu 0 --seed 0 | tee resnet[art_painting_cartoon_sketch]Tophoto_seed0.log
python demo.py  --net resnet  --dataset PACS --num_class 7 --source cartoon photo sketch --target art_painting --gpu 0 --seed 0 | tee resnet[cartoon_photo_sketch]Toart_painting_seed0.log
python demo.py  --net resnet  --dataset PACS --num_class 7 --source art_painting photo sketch --target cartoon --gpu 0 --seed 0 | tee resnet[art_painting_photo_sketch]Tocartoon_seed0.log
