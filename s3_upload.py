import os
import boto3
import datetime
import argparse
from common import is_image
from botocore.exceptions import ClientError
access_key_id = ''
access_pswd = ''
bucket = 'wivisiontest'

def upload2s3(src_pth, file_name, dest_path):
    res = boto3.resource('s3')
    clt = boto3.client('s3')
    try:
        clt.upload_file(src_pth, bucket, str(dest_path + '/' + file_name))
    except ClientError as e:
        print(e)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', help='Provide Directory path')
    parser.add_argument('--key_id', help='Provide Directory path')
    parser.add_argument('--pswd', help='Provide Directory path')
    args = parser.parse_args()
    src_path = args.src_path
    access_key_id = args.key_id
    access_pswd = args.pswd
    return src_path

def creat_dt_on_s3():
    res = boto3.resource('s3')
    clt = boto3.client('s3')
    now = datetime.datetime.now()
    dt_str = now.strftime("%d-%m-%y-%H-%M-%S")
    clt.put_object(Bucket = bucket, Key=(dt_str+'/'))
    return dt_str

if __name__ == "__main__":
    acc_id = ''
    psw_ = ''
    dir_ = creat_dt_on_s3()
    src_path = parse_opt()
    if len(os.listdir(src_path)) > 0:
        for file_ in os.listdir(src_path):
            if is_image(file_):
                upload2s3(src_path + file_, file_, dir_)