# -*- coding:utf-8 -*-
import yaml


def main():
    conf = load_conf("./conf/gen_pretrain_data.yaml")
    print(conf["input_file"])
    print(conf["output_file"])


def load_conf(path):
    return yaml.load(open(path))


def get_docs():
    docs = [[]]
    for line in open("./data/sample_text.txt"):
        line = line.strip()
        if line:
            docs[-1].append(line.split(" "))
            # docs[-1].append(line)
        else:
            docs.append([])

    return docs


def gen_sample():
    import random
    all_documents = get_docs()
    document_index = 1
    max_seq_length = 128
    short_seq_prob = 0.1
    rng = random.Random(12345)

    max_num_tokens = max_seq_length - 3
    target_seq_length = rng.randint(2, max_num_tokens) if rng.random() < short_seq_prob else max_num_tokens

    document = all_documents[document_index]
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while -len(document) < i < len(document):
        print("i={}".format(i if i >= 0 else i + len(document)))
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        print("len(current_chunk)={}, current_length={}".format(len(current_chunk), current_length))
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    random_document_index = 0
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                print("is_random_next={}, len_a={}, len_b={}, tokens_a={}, tokens_b={}"
                      .format(is_random_next, len(tokens_a), len(tokens_b), tokens_a, tokens_b))

        i += 1


def gen_json_sample():
    import json
    documents = get_docs()
    for d in documents:
        print(json.dumps(d, ensure_ascii=False))


def gen_tfrecords_sample():
    import collections
    import tensorflow as tf
    output_file = "./data/train.tfrecords"

    if tf.gfile.Exists(output_file):
        tf.gfile.Remove(output_file)
    writer = tf.python_io.TFRecordWriter(output_file)

    input_ids = [101, 2559, 2012, 2009, 2062, 2012, 103, 103, 1010, 2002, 2387, 2008, 2009, 8501, 1996, 9315, 1010,
                 1000, 2089, 2000, 16220, 1012, 1000, 102, 2066, 2087, 1997, 2010, 3507, 103, 103, 24071, 1010, 103,
                 2001, 3565, 16643, 9047, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    input_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    masked_lm_positions = [6, 7, 29, 30, 33, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    masked_lm_ids = [6528, 25499, 2751, 1011, 16220, 20771, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    masked_lm_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    next_sentence_label = 0

    features = collections.OrderedDict()
    features["input_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_ids)))
    features["input_mask"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(input_mask)))
    features["segment_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(segment_ids)))
    features["masked_lm_positions"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(masked_lm_positions)))
    features["masked_lm_ids"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(masked_lm_ids)))
    features["masked_lm_weights"] = tf.train.Feature(float_list=tf.train.FloatList(value=list(masked_lm_weights)))
    features["next_sentence_labels"] = tf.train.Feature(int64_list=tf.train.Int64List(value=list([next_sentence_label])))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(tf_example.SerializeToString())


def load_tfrecords_sample():
    import tensorflow as tf
    max_seq_length = 128
    max_predictions_per_seq = 20  # max_seq_length*0.15
    features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.FixedLenFeature([1], tf.int64),
    }

    input_file = "./data/train.tfrecords"
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(lambda record: tf.parse_single_example(record, features))
    iteration = dataset.make_one_shot_iterator()

    example_iter = tf.python_io.tf_record_iterator(input_file)
    example = next(example_iter)
    p = tf.parse_single_example(example, features)

    dataset = tf.data.TFRecordDataset([input_file])
    dataset = dataset.map(lambda record: tf.parse_single_example(record, features))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(p))
        print("-" * 36)
        try:
            while True:
                data_record = sess.run(next_element)
                print(data_record)
        except Exception as e:
            print("end! e={}".format(e))


def some_test():
    gen_json_sample()


if __name__ == '__main__':
    main()
