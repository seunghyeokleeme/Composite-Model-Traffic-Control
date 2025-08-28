import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from attention_layer import AttentionLayer
from traffic_dataset import TrafficDataLoader

def main(model_path, data_path):
    print("Loading test data...")
    data_loader = TrafficDataLoader(data_path=data_path)
    test_X = data_loader.test_X

    print(f"Loading model from '{model_path}'...")
    best_model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    # best_model = load_model(model_path)

    print("Generating predictions...")
    print(f"test_X: {type(test_X)}")
    print(f"test_X: {test_X.shape}")
    print(f"test_X: {test_X.ndim}")
    print(f"test_X: {test_X.dtype}")    

    # For debugging with a single sample  
    # test_X_first = test_X[0]
    # test_X_first = test_X_first[np.newaxis]

    # test_X_first = np.array([[
    # [149., 185., 302., 162., 117.,  85., 272.,  66., 171.,  97.,  18., 219.,  89., 135., 181., 103.,   0.],
    # [140.,	204.,	237.,	153.,	100.,	74.,	192.,	29.,	151.,	68.,	21.,	170.,	52.,	108.,	202.,	67., 0.],
    # [99.,	161.,	192.,	109.,	72.,	62.,	164.,	28.,	104.,	64.,	12.,	144.,	43.,	86.,	139.,	43., 0.],
    # [75.,	128.,	159.,	92.,	58.,	57.,	159.,	24.,	79.,	50.,	7.,	113.,	32.,	81.,	124.,	41., 0.],
    # [69.,	106.,	153.,	80.,	56.,	43.,	129.,	19.,	67.,	39.,	8.,	112.,	26.,	65.,	103.,	37., 0.],
    # [53.,	94.,	147.,	93.,	50.,	42.,	129.,	21.,	86.,	48.,	3.,	151.,	37.,	68.,	100.,	30., 0.],
    # [89.,	115.,	212.,	141.,	67.,	61.,	161.,	22.,	181.,	85.,	10.,	272.,	60.,	104.,	132.,	24., 0.],
    # [70.,	336.,	40.,	47.,	640.,	110.,	273.,	33.,	318.,	147.,	18.,	474.,	105.,	177.,	234.,	37., 0.],
    # [155.,	206.,	323.,	219.,	106.,	145.,	199.,	35.,	282.,	133.,	17.,	435.,	92.,	165.,	222.,	32., 0.],
    # [250.,	371.,	496.,	305.,	223.,	170.,	429.,	52.,	410.,	231.,	22.,	552.,	94.,	275.,	343.,	43., 0.],
    # [305.,	521.,	567.,	400.,	329.,	200.,	588.,	64.,	455.,	312.,	39.,	590.,	128.,	287.,	349.,	48., 0.],
    # [286.,	506.,	537.,	363.,	300.,	210.,	599.,	67.,	441.,	304.,	47.,	538.,	143.,	285.,	352.,	51., 0.],
    # [296.,	505.,	569.,	356.,	288.,	220.,	597.,	65.,	461.,	285.,	57.,	513.,	168.,	282.,	332.,	57., 0.],
    # [287.,	462.,	584.,	362.,	273.,	199.,	481.,	55.,	422.,	254.,	60.,	503.,	168.,	277.,	321.,	59., 0.],
    # [310.,	461.,	235.,	183.,	122.,	207.,	526.,	59.,	500.,	310.,	80.,	637.,	204.,	287.,	346.,	61., 0.],
    # [326.,	478.,	582.,	441.,	307.,	231.,	610.,	65.,	428.,	255.,	54.,	415.,	137.,	222.,	286.,	51., 0.],
    # [331.,	492.,	516.,	412.,	277.,	243.,	644.,	61.,	438.,	250.,	54.,	418.,	134.,	230.,	298.,	53., 0.],
    # [334.,	502.,	505.,	417.,	279.,	239.,	597.,	59.,	473.,	275.,	52.,	417.,	129.,	233.,	298.,	54., 0.],
    # [347.,	519.,	601.,	528.,	345.,	232.,	659.,	51.,	479.,	260.,	42.,	431.,	128.,	211.,	292.,	51., 0.],
    # [356.,	548.,	557.,	505.,	341.,	234.,	689.,	44.,	500.,	269.,	44.,	430.,	119.,	221.,	301.,	53., 0.],
    # [337.,	476.,	557.,	450.,	309.,	186.,	574.,	50.,	408.,	224.,	32.,	426.,	113.,	196.,	273.,	53., 0.],
    # [265.,	373.,	524.,	371.,	271.,	172.,	544.,	63.,	353.,	192.,	32.,	411.,	119.,	207.,	290.,	57., 0.],
    # [301.,	340.,	495.,	316.,	237.,	168.,	355.,	69.,	288.,	153.,	24.,	397.,	127.,	215.,	286.,	60., 0.],
    # [258.,	272.,	391.,	252.,	161.,	162.,	346.,	68.,	250.,	137.,	23.,	322.,	102.,	182.,	252.,	58., 0.]
    # ]], dtype=np.float64)

    # print(f"test_X_first: {test_X_first}")
    # print(f"test_X_first: {type(test_X_first)}")
    # print(f"test_X_first: {test_X_first.shape}")
    # print(f"test_X_first: {test_X_first.ndim}")
    # print(f"test_X_first: {test_X_first.dtype}")
    # predictions = best_model.predict(test_X_first)

    predictions = best_model.predict(test_X)
    print(predictions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained traffic prediction model.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the trained .keras model file.')
    parser.add_argument('--data_dir', default='./datasets', type=str, help='Dataset directory.')
    args = parser.parse_args()

    main(args.model_path, args.data_dir)