from datasets.basic_dataset_scaffold import BaseDataset
import os, numpy as np
import pandas as pd


def Give(opt, datapath):
    image_sourcepath  = opt.source_path+'/Img'
    # image_name item_id evaluation_status
    dataset_df = pd.read_table(opt.source_path+'/Eval/train_test_partition.txt', header=0, delimiter=' ')
    train_class = set()
    test_class = set()
    train_image_dict = {}
    train_conversion = {}
    test_image_dict = {}
    test_conversion = {}
    for i, (image_name, item_id, status) in enumerate(zip(dataset_df["image_name"], dataset_df["item_id"], dataset_df["evaluation_status"])):
        image_path = image_sourcepath + "/" + image_name
        class_id = int(item_id[-4:])
        class_str = image_name.split("/")
        class_name = image_name[1]+"_"+image_name[2]
        if status == "train":
            if class_id not in train_class:
                train_class.add(class_id)
                train_conversion[class_id] = class_name
                train_image_dict[class_id] = []
            train_image_dict[class_id].append(image_path)
        else:
            if class_id not in test_class:
                test_class.add(class_id)
                test_conversion[class_id] = class_name
                test_image_dict[class_id] = []
            test_image_dict[class_id].append(image_path)

    train_dataset = BaseDataset(train_image_dict, opt)
    test_dataset = BaseDataset(test_image_dict, opt, is_validation=True)
    eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion = train_conversion
    test_dataset.conversion = test_conversion
    eval_dataset.conversion = test_conversion
    eval_train_dataset.conversion = train_conversion
    val_dataset = None

    return {'training': train_dataset, 'validation': val_dataset, 'testing': test_dataset, 'evaluation': eval_dataset,
            'evaluation_train': eval_train_dataset}
