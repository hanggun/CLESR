import os
import re
from gensim.models import word2vec
import collections
import json
from tqdm import tqdm
from datasets import load_dataset
import jieba


def train_word2vec(train_file, output_model, iter=5):
    num_features = 300  # Word vector dimensionality
    min_word_count = 5  # Minimum word count
    num_workers = 12  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    sentences = word2vec.Text8Corpus(train_file)

    model = word2vec.Word2Vec(sentences, workers=num_workers,
                              vector_size=num_features, min_count=min_word_count,
                              window=context, sg=1, hs=1, epochs=iter)
    # 保存模型，供日後使用
    # model.save(output_model)
    model.wv.save_word2vec_format(output_model, binary=False)


def add_start_end(d):
    d = '[start] ' + d + ' [end]'
    return d


def get_genia_mrc():
    file_dir = '/home/zxa/ps/open_data/ner/genia/'
    filenames = ['train.jsonlines.txt', 'dev.jsonlines.txt', 'test.jsonlines.txt']

    categories = set()
    for file in filenames:
        file = os.path.join(file_dir, file)
        D = []
        with open(file, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                d = {'text': line['tokens'],
                     'labels': []}
                for mention in line['entity_mentions']:
                    d['labels'].append([mention['start'], mention['end']-1, mention['entity_type']])
                    categories.add(mention['entity_type'])
                D.append(d)

        # MRC
        mrc_out = []

        for line in D:
            for query in categories:
                labels = collections.defaultdict(list)
                if line['labels']:
                    for label in line['labels']:
                        labels[label[2]].append(label[:2])
                mrc_line = {'context': ' '.join(line['text']),
                            'start_position': [],
                            'end_position': [],
                            'entity_label': query,
                            'query': query,
                            'entity': []
                            }

                if query in labels:
                    for q in labels[query]:
                        mrc_line['start_position'].append(q[0])
                        mrc_line['end_position'].append(q[1])
                        mrc_line['entity'].append(line['text'][q[0]:q[1] + 1])
                mrc_out.append(mrc_line)

        out_file = os.path.join(file_dir, 'mrc-ner.' + file.split('/')[-1].split('.')[0])
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(mrc_out, ensure_ascii=False, indent=2))


def generate_genia_word2vec_context_label():
    """生成训练上下文标签向量的文本
    格式： Okadaic acid was found to be a potent inducer of [protein] .
    """

    file_dir = '/home/zxa/ps/open_data/ner/genia/'
    filenames = ['genia.dev', 'genia.test', 'genia.train']

    output = []
    context_label_name = set()
    for file in filenames:
        file = os.path.join(file_dir, file)
        text, label = None, None
        with open(file, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                if line_idx % 3 == 0:
                    original_text = line.strip()
                    text = line.strip().split()
                if line_idx % 3 == 1:
                    if line.strip():
                        label = line.strip().split('|')
                    else:
                        label = None
                if line_idx % 3 == 2:
                    if label:
                        for l in label:
                            new_label = '[' + l.split('#')[1] + ']' # [protein]
                            context_label_name.add(new_label)
                            start, end = list(map(int, l.split()[0].split(',')))
                            masked_text = text.copy()
                            for i in range(start, end):
                                if i == start:
                                    masked_text[i] = new_label
                                else:
                                    masked_text[i] = '' # 填充位置，否则前面位置变了，后面位置也会便
                            masked_text = ' '.join(masked_text)
                            masked_text = re.sub(' +', ' ', masked_text)
                            output.append(original_text + ' ' + masked_text)
                    else:
                        text = ' '.join(text)
                        output.append(text)

    with open(os.path.join(file_dir, 'genia_context_label_word2vec.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    with open(os.path.join(file_dir, 'context_label_name.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(context_label_name))
    print(f'data successfully saved in genia_context_label_word2vec.txt')


def get_conll_mrc_data():
    """将conll转化为mrc格式"""
    file_dir = '/home/zxa/ps/open_data/ner/conll2003/'
    filenames = ['train.txt', 'dev.txt', 'test.txt']

    label2query = {
        'ORG': 'organization',
        'PER': 'person',
        'LOC': 'location',
        'MISC': 'name of miscellaneous'
    }
    label2fullquery = {
        'ORG': 'Find organizations including companies, agencies and institutions:',
        'PER': 'Find persons including names and fictional characters:',
        'LOC': 'Find locations including countries, cities, towns, continents',
        'MISC': 'Find names of miscellaneous:'
    }
    categories = set()
    total_D = []
    for file in filenames:
        file = os.path.join(file_dir, file)
        data = open(file, encoding='utf8').read().split('\n\n')
        count = 0
        D = []
        maxlen = 0
        for line in tqdm(data, f'process {file}'):
            if 'DOCSTART' in line or not line:
                continue
            count += 1

            d = {'text': [],
                 'labels': []}
            sen_split = line.split('\n')
            for i, items in enumerate(sen_split):
                items = items.split()
                flag = items[-1]
                assert len(items) == 4
                d['text'].append(items[0])
                if flag[0] == 'B':
                    d['labels'].append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d['labels'][-1][1] = i
            D.append(d)
            maxlen = max(maxlen, len(d['text']))
        total_D.extend(D)

        # MRC
        mrc_out = []

        for line in D:
            for query in categories:
                labels = collections.defaultdict(list)
                if line['labels']:
                    for label in line['labels']:
                        labels[label[2]].append(label[:2])
                mrc_line = {'context': ' '.join(line['text']),
                            'start_position': [],
                            'end_position': [],
                            'entity_label': query,
                            'query': label2query[query],
                            'entity': []
                            }

                if query in labels:
                    for q in labels[query]:
                        mrc_line['start_position'].append(q[0])
                        mrc_line['end_position'].append(q[1])
                        mrc_line['entity'].append(line['text'][q[0]:q[1]+1])
                mrc_out.append(mrc_line)

        out_file = os.path.join(file_dir, 'mrc-ner.'+file.split('/')[-1].split('.')[0])
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(mrc_out, ensure_ascii=False, indent=2))

        print(count)
        print(maxlen)
    # word2vec context file
    generate_conll_word2vec_context_label(total_D)


def generate_conll_word2vec_context_label(D):
    """生成训练上下文标签向量的文本
    """

    output = []
    for d in D:
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                start = label[0]
                end = label[1]
                new_label = '[' + label[2] + ']'
                text = d['text'].copy()
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                text = ' '.join(text)
                text = re.sub(' +', ' ', text)
                output.append(original_text + ' ' + text)
        else:
            output.append(original_text)

    save_file = '/home/zxa/ps/open_data/ner/conll2003/conll_context_label_word2vec.txt'
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


def generate_msra_word2vec_context_label(is_cutword=False):
    """生成训练上下文标签向量的文本
    """
    categories = set()
    D = []
    file_dir = '/home/zxa/ps/open_data/ner/zh_msra/'
    for file in ['mrc-ner.train', 'mrc-ner.dev', 'mrc-ner.test']:
        with open(os.path.join(file_dir, file), encoding='utf8') as f:
            data = json.load(f)
        d = {}
        for line in tqdm(data, 'load text and label'):
            if line['span_position']:
                label = line['entity_label']
                categories.add(label)
                text = ''.join(line['context'].split())
                for span in line['span_position']:
                    start, end = map(int, span.split(';'))
                    if text in d:
                        d[text].append([start, end, label])
                    else:
                        d[text] = [[start, end, label]]
        for text, labels in d.items():
            D.append({'text': text, 'labels': labels})

    output = []
    for d in tqdm(D, 'process'):
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                text = list(d['text'])
                start = label[0]
                end = label[1]
                if not is_cutword:
                    new_label = '[' + label[2] + ']'
                else:
                    new_label = label[2]
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                if not is_cutword:
                    text = ' '.join(text)
                    text = re.sub(' +', ' ', text)
                    output.append(original_text + ' ' + text)
                else:
                    text.insert(start, ' ')
                    text.insert(end+2, ' ')
                    text = ''.join(text)
                    original_text = list(d['text'])
                    original_text.insert(start, ' ')
                    original_text.insert(end+2, ' ')
                    original_text = ''.join(original_text)
                    text = ' '.join(jieba.lcut(text))
                    text = text.replace(new_label, '['+new_label+']')
                    text = re.sub(' +', ' ', text)
                    original_text = re.sub(' +', ' ', ' '.join(jieba.lcut(original_text)))
                    output.append(original_text + ' ' + text)
        else:
            if not is_cutword:
                output.append(original_text)
            else:
                output.append(' '.join(jieba.lcut(d['text'])))

    if not is_cutword:
        save_file = '/home/zxa/ps/open_data/ner/zh_msra/msra_context_label_word2vec.txt'
    else:
        save_file = '/home/zxa/ps/open_data/ner/zh_msra/msra_cut_context_label_word2vec.txt'
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


def check_no_label_rate(file_dir):
    for file in ['mrc-ner.train', 'mrc-ner.dev', 'mrc-ner.test']:
        with open(os.path.join(file_dir, file), encoding='utf8') as f:
            data = json.load(f)
        sample_dict = {}
        for line in data:
            text = line['context']
            sample_dict[text] = sample_dict.get(text, False) | (line.get('span_position') != [])

        total, num_true, num_false = 0, 0, 0
        for key, value in sample_dict.items():
            if value:
                num_true += 1
            else:
                num_false += 1
            total += 1

        print(f"{file}: non label rate {num_false/total}")


def create_onenote_bmes(D, file_dir, mode='train'):
    named_entities = D.info.features['sentences'][0]['named_entities'].feature.names
    out = []
    for doc in tqdm(D, f'process {mode}'):
        for line in doc['sentences']:
            words = line['words']
            entities = line['named_entities']
            tmp = []
            for word, entity in zip(words, entities):
                tmp.append(word+'\t'+named_entities[entity]+'\n')
            tmp.append(' \n')
            out.append(''.join(tmp))
    with open(file_dir+f"{mode}.bi", 'w', encoding='utf8') as f:
        f.write(''.join(out))


def get_onenote5_mrc_data():
    """将conll转化为mrc格式"""
    file_dir = '/home/zxa/ps/open_data/ner/onenotesv5/'
    filenames = ['train.bi', 'dev.bi', 'test.bi']

    label2query = {
        'CARDINAL': 'cardinal value',
        'DATE': 'date value',
        'EVENT': 'event name',
        'FAC': 'building name',
        'GPE': 'geo-political entity',
        'LANGUAGE': 'language name',
        'LAW': 'law name',
        'LOC': 'location name',
        'MONEY': 'money name',
        'NORP': 'affiliation',
        'ORDINAL': 'ordinal value',
        'ORG': 'organization name',
        'PERCENT': 'percent value',
        'PERSON': 'person name',
        'PRODUCT': 'product name',
        'QUANTITY': 'quantity value',
        'TIME': 'time value',
        'WORK_OF_ART': 'name of work of art',
    }
    categories = set()
    total_D = []
    for file in filenames:
        file = os.path.join(file_dir, file)
        data = open(file, encoding='utf8').read().split('\n \n')
        count = 0
        D = []
        maxlen = 0
        for line in tqdm(data, f'process {file}'):
            if not line:
                continue
            count += 1

            d = {'text': [],
                 'labels': []}
            sen_split = line.split('\n')
            for i, items in enumerate(sen_split):
                items = items.strip().split('\t')
                flag = items[-1]
                assert len(items) == 2
                d['text'].append(items[0])
                if flag[0] == 'B':
                    d['labels'].append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d['labels'][-1][1] = i
            D.append(d)
            maxlen = max(maxlen, len(d['text']))
        total_D.extend(D)

        # MRC
        mrc_out = []

        for line in D:
            for query in categories:
                labels = collections.defaultdict(list)
                if line['labels']:
                    for label in line['labels']:
                        labels[label[2]].append(label[:2])
                mrc_line = {'context': ' '.join(line['text']),
                            'start_position': [],
                            'end_position': [],
                            'entity_label': query,
                            'query': label2query[query],
                            'entity': []
                            }

                if query in labels:
                    for q in labels[query]:
                        mrc_line['start_position'].append(q[0])
                        mrc_line['end_position'].append(q[1])
                        mrc_line['entity'].append(line['text'][q[0]:q[1]+1])
                mrc_out.append(mrc_line)

        out_file = os.path.join(file_dir, 'mrc-ner.'+file.split('/')[-1].split('.')[0])
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(mrc_out, ensure_ascii=False, indent=2))

        print(count)
        print(maxlen)
    # word2vec context file
    generate_onenote5_word2vec_context_label(total_D)


def generate_onenote5_word2vec_context_label(D):
    """生成训练上下文标签向量的文本
    """

    output = []
    for d in D:
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                text = list(d['text'])
                start = label[0]
                end = label[1]
                new_label = '[' + label[2] + ']'
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                text = ' '.join(text)
                text = re.sub(' +', ' ', text)
                output.append(original_text + ' ' + text)
        else:
            output.append(original_text)

    save_file = '/home/zxa/ps/open_data/ner/onenotesv5/onenotev5_context_label_word2vec.txt'
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


def split_ACE2005_documents():
    import glob
    import pandas as pd
    import numpy as np
    filedir = '/home/zxa/ps/open_data/ner/ace2005/data/English/'
    used_files = ['bc', 'bn', 'nw', 'wl']
    total_files = []
    for file in used_files:
        used_file_path = os.path.join(filedir, file, 'adj')
        files = glob.glob(used_file_path+'/*.sgm')
        files = [x[:-4] for x in files]
        total_files.extend(files)
    np.random.shuffle(total_files)
    file_nums = len(total_files)
    out_dict = {'type': [],
                'path': []}
    train_files = total_files[:int(file_nums*0.8)]
    dev_files = total_files[int(file_nums*0.8):int(file_nums*0.9)]
    test_files = total_files[int(file_nums*0.9):]
    split_names = ['train', 'dev', 'test']
    split_files = [train_files, dev_files, test_files]
    for name, files in zip(split_names, split_files):
        for file in files:
            out_dict['type'].append(name)
            out_dict['path'].append(file)
    dt = pd.DataFrame(out_dict)
    dt = dt.set_index('type')
    dt.to_csv('data_list.csv')


def split_ACE2004_documents():
    import glob
    import pandas as pd
    import numpy as np
    filedir = '/home/zxa/ps/open_data/ner/ace2004/data/English/'
    used_files = ['arabic_treebank', 'bnews', 'chinese_treebank', 'nwire']
    total_files = []
    for file in used_files:
        used_file_path = os.path.join(filedir, file)
        files = glob.glob(used_file_path+'/*.sgm')
        files = [x[:-4] for x in files]
        total_files.extend(files)
    np.random.shuffle(total_files)
    file_nums = len(total_files)
    out_dict = {'type': [],
                'path': []}
    train_files = total_files[:356]
    dev_files = total_files[356:397]
    test_files = total_files[397:]
    split_names = ['train', 'dev', 'test']
    split_files = [train_files, dev_files, test_files]
    for name, files in zip(split_names, split_files):
        for file in files:
            out_dict['type'].append(name)
            out_dict['path'].append(file)
    dt = pd.DataFrame(out_dict)
    dt = dt.set_index('type')
    dt.to_csv('data_list.csv')


def get_ACE2005_mrc_data(file_dir, context_save_file):
    """将conll转化为mrc格式"""
    from nltk.tokenize import word_tokenize
    import re

    def check_all_entity(d, categories):
        flag = False
        for entity in d:
            if entity in categories:
                flag = True
        return flag

    filenames = ['train.json', 'dev.json', 'test.json']

    label2query = {
        'PER': 'person',
        'ORG': 'organization',
        'GPE': 'geographical entity',
        'LOC': 'location',
        'FAC': 'facility',
        'WEA': 'weapon',
        'VEH': 'vehicle'
    }
    total_labels = set()
    categories = list(label2query.keys())
    total_D = []
    for file in filenames:
        file = os.path.join(file_dir, file)
        data = json.load(open(file))
        count = 0
        D = []
        maxlen = 0
        total_entities = 0
        ignored = 0
        for line in tqdm(data, f'process {file}'):
            count += 1

            text = word_tokenize(line['sentence'])
            d = {'text': text,
                 'labels': []}
            entities = line['golden-entity-mentions']
            for entity in entities:
                entity['text'] = word_tokenize(entity['text'])
                start, end = entity['start'], entity['end']
                total_entities += 1
                total_labels.add(entity['entity-type'].split(':')[0])
                if entity['text'] == text[start:end] and entity['entity-type'].split(':')[0] in categories:
                    if end > len(text):
                        print('end > length of text')
                        end = len(text)
                    d['labels'].append([start, end-1, entity['entity-type'].split(':')[0]])
                else:
                    ignored += 1

            D.append(d)
            maxlen = max(maxlen, len(line['sentence'].split()))
        total_D.extend(D)
        print(f'total entities {total_entities}, ignored {ignored}')

        # MRC
        mrc_out = []

        for line in D:
            for query in categories:
                labels = collections.defaultdict(list)
                if line['labels']:
                    for label in line['labels']:
                        labels[label[2]].append(label[:2])
                mrc_line = {'context': ' '.join(line['text']),
                            'start_position': [],
                            'end_position': [],
                            'entity_label': query,
                            'query': label2query[query],
                            'entity': []
                            }

                if query in labels:
                    for q in labels[query]:
                        mrc_line['start_position'].append(q[0])
                        mrc_line['end_position'].append(q[1])
                        mrc_line['entity'].append(line['text'][q[0]:q[1]+1])
                mrc_out.append(mrc_line)

        out_file = os.path.join(file_dir, 'mrc-ner.'+file.split('/')[-1].split('.')[0])
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(mrc_out, ensure_ascii=False, indent=2))

        print(count)
        print(maxlen)
    # word2vec context file
    generate_ACE2005_word2vec_context_label(total_D, context_save_file)


def generate_ACE2005_word2vec_context_label(D, context_save_file):
    """生成训练上下文标签向量的文本
    """

    output = []
    for d in tqdm(D, 'create context corpus'):
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                text = d['text'].copy()
                start = label[0]
                end = label[1]
                new_label = '[' + label[2] + ']'
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                text = ' '.join(text)
                text = re.sub(' +', ' ', text)
                output.append(original_text + ' ' + text)
        else:
            output.append(original_text)

    save_file = context_save_file
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


def get_ontonotes4_mrc_data():
    """将conll转化为mrc格式"""
    file_dir = '/home/zxa/ps/open_data/ner/ontonotes4/'
    filenames = ['train.char.bmes', 'dev.char.bmes', 'test.char.bmes']

    label2query = {
        'PER': '人名',
        'GPE': '城市 国家 省份 州 县',
        'LOC': '地点',
        'ORG': '组织',
    }
    categories = set()
    total_D = []
    for file in filenames:
        file = os.path.join(file_dir, file)
        data = open(file, encoding='utf8').read().split('\n\n')
        count = 0
        D = []
        maxlen = 0
        avgs = []
        ents = 0
        for line in tqdm(data, f'process {file}'):
            if not line:
                continue
            count += 1

            d = {'text': [],
                 'labels': []}
            sen_split = line.split('\n')
            for i, items in enumerate(sen_split):
                items = items.strip().split()
                flag = items[-1]
                assert len(items) == 2
                d['text'].append(items[0])
                if flag[0] == 'B':
                    d['labels'].append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'E':
                    d['labels'][-1][1] = i
                    ents += 1
                elif flag[0] == 'S':
                    d['labels'].append([i, i, flag[2:]])
                    categories.add(flag[2:])
                    ents += 1
            D.append(d)
            maxlen = max(maxlen, len(d['text']))
            avgs.append(len(d['text']))
        total_D.extend(D)

        # MRC
        mrc_out = []

        for line in D:
            for query in categories:
                labels = collections.defaultdict(list)
                if line['labels']:
                    for label in line['labels']:
                        labels[label[2]].append(label[:2])
                mrc_line = {'context': ' '.join(line['text']),
                            'start_position': [],
                            'end_position': [],
                            'entity_label': query,
                            'query': label2query[query],
                            'entity': []
                            }

                if query in labels:
                    for q in labels[query]:
                        mrc_line['start_position'].append(q[0])
                        mrc_line['end_position'].append(q[1])
                        mrc_line['entity'].append(line['text'][q[0]:q[1]+1])
                mrc_out.append(mrc_line)

        out_file = os.path.join(file_dir, 'mrc-ner.'+file.split('/')[-1].split('.')[0])
        with open(out_file, 'w', encoding='utf8') as f:
            f.write(json.dumps(mrc_out, ensure_ascii=False, indent=2))

        print(count)
        print(maxlen)
    # word2vec context file
    generate_ontonotes4_word2vec_context_label(total_D)


def generate_ontonotes4_word2vec_context_label(D):
    """生成训练上下文标签向量的文本
    """

    output = []
    for d in D:
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                text = list(d['text'])
                start = label[0]
                end = label[1]
                new_label = '[' + label[2] + ']'
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                text = ' '.join(text)
                text = re.sub(' +', ' ', text)
                output.append(original_text + ' ' + text)
        else:
            output.append(original_text)

    save_file = '/home/zxa/ps/open_data/ner/ontonotes4/ontonotes4_context_label_word2vec.txt'
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


def find_nested_samples_from_ace2004():
    file_dir = '/home/zxa/ps/open_data/ner/ace2004/train.json'
    out = []
    with open(file_dir, encoding='utf8') as f:
        data = json.load(f)
        for line in tqdm(data):
            sentence = line['sentence']
            entities = line['golden-entity-mentions']
            if 14 <= len(sentence.split()) <= 15:
                span = []
                nested = False
                for entity in entities:
                    span.append([entity['start'], entity['end']])
                span = sorted(span, key=lambda x: x[0])
                for i in range(len(span)):
                    for j in range(len(span)):
                        if i == j:
                            continue
                        e1 = span[i]
                        e2 = span[j]
                        if e1[0] <= e2[0] <= e2[1] <= e1[1] or e2[0] <= e1[0] <= e1[1] <= e2[1]:
                            nested = True
                        if e1[0] <= e2[0] <= e1[1] <= e2[1] or e2[0] <= e1[0] <= e2[1] <= e1[1]:
                            nested = True
                if nested:
                    out.append(sentence)
    print(out)


def generate_bid_word2vec_context_label(is_cutword=False):
    """生成训练上下文标签向量的文本
    """
    query2label = {
        "招标单位名称": "tenderee",
        "招标单位联系人": "tendereePerson",
        "招标单位联系电话": "tendereePhone",
        "招标单位地址": "tendereeLocation",
        "招标单位联系邮箱": "tendereeEmail",
        "招标代理机构单位名称": "agency",
        "招标代理机构单位联系人": "agencyPerson",
        "招标代理机构单位联系电话": "agencyPhone",
        "招标代理机构单位地址": "agencyLocation",
        "招标代理机构单位联系邮箱": "agencyEmail",
        "中标单位名称或供应商名称": "firstBiddingWinner",
        "中标单位地址或供应商地址": "firstBiddingWinnerLocation",
        "中标金额": "firstBiddingWinnerMoney",
        "项目预算": "projectBudget",
        "可能为预算的金额": "possibleProjectBudget",
        "项目编号": "projectNo",
        "项目名称": "projectName",
        "标书获取截止时间或报价截止时间或报名截止时间": "bidDocumentDeadline",
        "投标截止时间": "bidDeadline",
        "开标日期或资格预审日期或评审日期": "bidOpeningDate",
        "标的物名称": "projectObject",
        "参与竞标的公司包括中标候选人": "bidder",
        "其他联系人": "otherPerson",
        "其他联系电话": "otherPhone",
        "其他联系邮箱": "otherEmail",
        "投标人响应人资质": "bidderQualification",
        "项目负责人的资格要求": "projectLeaderQualification",
        "投标人响应人其他要求": "bidderRequirements"
    }
    categories = set()
    D = []
    file_dir = r'/home/zxa/ps/open_data/ner/bid/'
    for file in ['mrc-ner.train', 'mrc-ner.dev']:
        with open(os.path.join(file_dir, file), encoding='utf8') as f:
            data = json.load(f)
        d = {}
        for line in tqdm(data, 'load text and label'):
            if line['start_position']:
                label = line['entity_label']
                categories.add(label)
                text = line['context']
                for start, end in zip(line['start_position'], line['end_position']):
                    if text in d:
                        d[text].append([start, end, label])
                    else:
                        d[text] = [[start, end, label]]
        for text, labels in d.items():
            D.append({'text': text, 'labels': labels})

    output = []
    for d in tqdm(D, 'process'):
        original_text = ' '.join(d['text'])
        if d['labels']:
            for label in d['labels']:
                text = list(d['text'])
                start = label[0]
                end = label[1]
                if not is_cutword:
                    new_label = '[' + label[2] + ']'
                else:
                    new_label = query2label[label[2]]
                for i in range(start, end+1):
                    if i == start:
                        text[i] = new_label
                    else:
                        text[i] = ''
                if not is_cutword:
                    text = ' '.join(text)
                    text = re.sub(' +', ' ', text)
                    output.append(original_text + ' ' + text)
                else:
                    text.insert(start, ' ')
                    text.insert(end+2, ' ')
                    text = ''.join(text)
                    original_text = list(d['text'])
                    original_text.insert(start, ' ')
                    original_text.insert(end+2, ' ')
                    original_text = ''.join(original_text)
                    text = ' '.join(jieba.lcut(text))
                    text = text.replace(new_label, '['+new_label+']')
                    text = re.sub(' +', ' ', text)
                    original_text = re.sub(' +', ' ', ' '.join(jieba.lcut(original_text)))
                    output.append(original_text + ' ' + text)
        else:
            if not is_cutword:
                output.append(original_text)
            else:
                output.append(' '.join(jieba.lcut(d['text'])))

    if not is_cutword:
        save_file = '/home/zxa/ps/open_data/ner/bid/bid_context_label_word2vec.txt'
    else:
        save_file = '/home/zxa/ps/open_data/ner/bid/bid_cut_context_label_word2vec.txt'
    with open(save_file, 'w', encoding='utf8') as f:
        f.write('\n'.join(output))
    print(f'context label file saved in {save_file}')


if __name__ == '__main__':
    # get_genia_mrc()
    # generate_genia_word2vec_context_label()
    # train_word2vec('/home/zxa/ps/open_data/ner/genia/genia_context_label_word2vec.txt',
    #                '/home/zxa/ps/open_data/ner/genia/genia_vector.txt', 200)
    # get_conll_mrc_data()
    # train_word2vec('/home/zxa/ps/open_data/ner/conll2003/conll_context_label_word2vec.txt',
    #                '/home/zxa/ps/open_data/ner/conll2003/conll_vector.txt', 100)

    # generate_msra_word2vec_context_label(is_cutword=True)
    # train_word2vec('/home/zxa/ps/open_data/ner/zh_msra/msra_context_label_word2vec.txt',
    #                '/home/zxa/ps/open_data/ner/zh_msra/msra_vector.txt', 100)

    # get_onenote5_mrc_data()
    # train_word2vec('/home/zxa/ps/open_data/ner/onenotesv5/onenotev5_context_label_word2vec.txt',
    #                '/home/zxa/ps/open_data/ner/onenotesv5/onenotev5_vector.txt', 10)

    get_ontonotes4_mrc_data()
    #
    # get_ACE2005_mrc_data(file_dir='/home/zxa/ps/open_data/ner/ace2005/',
    #                      context_save_file='/home/zxa/ps/open_data/ner/ace2005/ace2005_context_label_word2vec.txt')


    # check_no_label_rate(file_dir='/home/zxa/ps/open_data/ner/onenotesv5/')

    # load onenotes data
    # test = load_dataset('conll2012_ontonotesv5', 'english_v12', split='test')
    # create_onenote_bmes(test, mode='test')
    # dev = load_dataset('conll2012_ontonotesv5', 'english_v12', split='validation')
    # create_onenote_bmes(dev, mode='dev')
    # train = load_dataset('conll2012_ontonotesv5', 'english_v12', split='train')
    # create_onenote_bmes(train, mode='train')

    # split_ACE2005_documents()
    # split_ACE2004_documents()

    # find_nested_samples_from_ace2004()

    # generate_bid_word2vec_context_label(is_cutword=True)