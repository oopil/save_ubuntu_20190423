import os
import sys
import time
import argparse
from class_metabot import *

"""parsing and configuration"""
def parse_args()->argparse:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default='0', help='the index of class from 0 to 3')
    return parser.parse_args()

def extr_meta_data(class_name)->list:
    print('start to extract meta data from dataset folder')
    # base_folder_path = '/home/sp/Datasets/MRI_chosun/ADAI_MRI_Result_V1_0'
    base_folder_path = '/home/sp/Datasets/MRI_chosun/test_sample_2'

    bot = MetaBot(base_folder_path)
    meta_list = bot.MRI_chosun(class_name)
    print(class_name, len(meta_list), meta_list)
    del bot
    return meta_list

def chosun_MRI_pipeline(index)->None:
    bot = MetaBot('.')
    result_file_name = 'chosun_MRI_pipeline_finish_list_' + str(index)
    contents = []
    if bot.is_exist(result_file_name):
        file = open(result_file_name, 'rt')
        contents = file.readlines()
        print(contents)
        file.close()

    file = open(result_file_name, 'a+t')
    class_name = ['aAD', 'ADD', 'mAD', 'NC']
    meta_data_list = extr_meta_data(class_name[index])
    total_num = len(meta_data_list)
    for i, subj in enumerate(meta_data_list):
        subj_name = subj[1]
        name = subj_name + '\n'
        print('<<< Index : ', i+1, '/', total_num ,' >>>')
        if name in contents:
            print('this subj is already processed.', subj_name)
            continue
        chosun_MRI_preprocess(subj[0], subj[1], subj[2])
        file.writelines(subj_name+'\n')
    file.close()
    file = open(result_file_name, 'rt')
    print(file.readlines())
    file.close()
    del bot
    return

def chosun_MRI_preprocess(folder_path, subj_name, input_image)->None:
    print('run the pipeline for chosun univ MRI data.')
    print('the pipeline contains autorecon1 and autorecon-inflate1')
    result_folder_name = 'freesurfer'
    subj_dir_path = os.path.join(folder_path, subj_name)
    options = ['-autorecon1','-autorecon2-inflate1']
    is_run = [False, False]

    command1 = 'recon-all '+'-i '+ input_image + ' -s ' + result_folder_name + ' -sd '+ subj_dir_path + ' ' + options[0]
    command2 = 'recon-all '+' -s ' + result_folder_name + ' -sd '+ subj_dir_path + ' ' + options[1]
    export_command = 'export SUBJECT_DIR='+folder_path

    print('\n' + export_command)
    print(command1)
    print(command2, '\n')

    if is_run[0]:
        start_time_1 = time.strftime("20%y:%m:%d %H:%M")
        # assert False
        os.system(export_command)
        os.system(command1)
        start_time_2 = time.strftime("20%y:%m:%d %H:%M")

    if is_run[1]:
        print(command2, '\n')
        os.system(command2)

    end_time = time.strftime("20%y:%m:%d %H:%M")
    print('processing pipeline is done.')
    if is_run[0]:
        print('start time 1 : ',start_time_1)
    if is_run[1]:
        print('start time 2 : ',start_time_2)
    print('end time : ',end_time)

    # if succeed to run all pipeline,
    # write the name in the finished list file
    return
#

def test():
    test = ['a','b','c','d']
    file = open('test_file', 'a+t')
    for i in test:
        file.writelines(i+'\n')
    file.close()
    file = open('test_file', 'r')
    contents = file.readlines()
    print(contents)
    file.close()

def main()->None:
    # test()
    args = parse_args()
    index = args.index
    # extr_meta_data('aAD')
    chosun_MRI_pipeline(index)
    # chosun_MRI_preprocess()
    return

if __name__ == '__main__':
    main()

# def chosun_MRI_preprocess()->None:
#     print('run the pipeline for chosun univ MRI data.')
#     print('the pipeline contains autorecon1 and autorecon-inflate1')
#     folder_path = '/home/sp/Datasets/MRI_chosun/test_sample'
#     subj_name = 'mAD'
#     result_folder_name = 'freesurfer'
#     subj_dir_path = os.path.join(folder_path, subj_name)
#     input_image = os.path.join(subj_dir_path, 'T1.nii.gz')
#     options = ['-autorecon1','-autorecon2-inflate1']
#
#     command1 = 'recon-all '+'-i '+ input_image + ' -s ' + result_folder_name + ' -sd '+ subj_dir_path + ' ' + options[0]
#     command2 = 'recon-all '+' -s ' + result_folder_name + ' -sd '+ subj_dir_path + ' ' + options[1]
#     export_command = 'export SUBJECT_DIR='+folder_path
#
#     print('\n' + export_command)
#     print(command1)
#     print(command2, '\n')
#
#     start_time_1 = time.strftime("20%y:%m:%d %H:%M")
#     # assert False
#     os.system(export_command)
#     os.system(command1)
#     start_time_2 = time.strftime("20%y:%m:%d %H:%M")
#     print(command2, '\n')
#     os.system(command2)
#     end_time = time.strftime("20%y:%m:%d %H:%M")
#
#     print('processing pipeline is done.')
#     print('start time 1 : ',start_time_1)
#     print('start time 2 : ',start_time_2)
#     print('end time : ',end_time)
#
#     # if succeed to run all pipeline,
#     # write the name in the finished list file
#     return
