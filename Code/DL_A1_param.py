import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalization", default='inbuilt', help='provide normalization type among BN, IN, BIN, LN, GN, NN, torch_bn')
    parser.add_argument("--n", type= int,default= 2, help='no of layers with feature map size 256,128 and 64')
    parser.add_argument("--r", type= int,default= 25, help='no of classes in dataset')
    parser.add_argument("--batch_size", type= int,default= 32)
    parser.add_argument("--graph", help='no of classes in dataset')
    parser.add_argument("--test_data_file", default= '/home/mikshu/Desktop/DL_A1/birds_test',help='input data path')
    parser.add_argument("--output_file", default= '/home/mikshu/Desktop/DL_A1/op_r.csv',help='output model and csv path')
    parser.add_argument("--optim", help='type of optimizer')
    parser.add_argument("--model_type", help='type of model_type')
    parser.add_argument("--beam_size", help='size of beam')
    parser.add_argument("--model_file", default = '/home/mikshu/Desktop/DL_A1/resnet/all_model/part_1.1.pth', help='path to saved model')

    args = parser.parse_args()

    return args

args = parse_args()




