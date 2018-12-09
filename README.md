# Sketch-Classifier-retrained-from-Inception
the course lab of HCI, UCAS

对Google TensorFlow Inception v3进行retrain后，得到的针对漫画草图的分类器
根目录下的inception文件夹内为原始的分类模型，执行命令为
>Python classifier.py --image_file file_path_to_image --num_top_predictions number_of_top_results

其中最后的参数为获取top N分类，非必须

Pokemon文件夹为retrain时使用的训练集，包括7个分类

retrain命令为
>python retrain.py --model_dir .\inception --image_dir .\Pokemon --how_many_training_steps 5000

最后的迭代次数参数非必须，默认4000

本实验中

step=200时，train accuracy为95%左右，validation accuracy 88%，cross entropy 0.60

step=1000,train accuracy 98%, validation accuracy 98%, cross entropy 0.2

step=5000,train accuracy 100%, validation accuracy 98%, cross entropy 0.04

retrain后得到的模型在new_model下，其中的classifier.py与根目录下的不同，进行分类的命令为
>python classifier.py file_path_to_image

