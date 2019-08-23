# -*- coding:utf-8 -*-
import yaml


def main():
    pass


def load_conf(path):
    return yaml.load(open(path))


def some_test():
    conf = load_conf("./conf/gen_pretrain_data.yaml")
    print(conf)
    print(conf["input_files"])
    # x = "xxx,yyy"
    # y = x.strip().split(",")
    # print(y)


if __name__ == '__main__':
    some_test()
