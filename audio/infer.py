import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments



parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   '配置文件')
add_arg('use_gpu',          bool,   True,                       '是否使用GPU预测')
add_arg('audio_path',       str,    '/data/cx/AudioClassification-Pytorch-master/test.txt','音频路径')
add_arg('model_path',       str,    'models/EcapaTdnn_MelSpectrogram/best_model/', '导出的预测模型文件路径')
args = parser.parse_args()
print_arguments(args=args)



# 获取识别器
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

text,labels,scores = predictor.predict_batch(audios_data=args.audio_path)



print(f'{text}\n'
       f'预测结果标签为：{labels}\n'
       f' 得分：{scores}\n')
#print(f'音频预测结果：{results}')