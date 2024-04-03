# AD_Detection
AD Detection Research


- AD Rearch requirements
- Usage: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

1. Install
cd YOLOX
pip3 install -v -e .  # or  python3 setup.py develop

ln -s /home/glint/xzwang/data/raw_data/VOCdevkitOutStripe ./datasets/VOCdevkit
ln -s /home/glint/xzwang/data/stripe_voc/VOCdevkitOutStripe ./datasets/VOCdevkit

pip install torch==1.13.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

pip uninstall nvidia_cublas_cu11
pip freeze > requirements.txt



I removed the miniconda3/lib/python3.10/site-packages/nvidia/cublas and miniconda3/envs/cai/lib/python3.8/site-packages/torch/lib/libcublas*, and the error goes away.


python tools/train.py  -expn 'voc_data' -e 10 --resume


change epoch, warmup, data augumentation 
- /home/glint/xzwang/code/YOLOX/yolox/exp/yolox_base.py

change model
/home/glint/xzwang/code/YOLOX/exps/example/yolox_voc/yolox_voc_s.py

python 裁剪_按照指定step裁剪图像为指定大小.py --images_path /home/glint/xzwang/AnalysisData/2-10_3.28 --labels_path /home/glint/xzwang/AnalysisData/2-10_3.28 --output_path /home/glint/xzwang/AnalysisData/





