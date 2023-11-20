class config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/roberta-large/'
    train_data_path = "/home/zxa/ps/open_data/ner/genia/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/genia/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/genia/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/genia/genia_vector_iter200.txt'

    loss_type = 'ce'

    max_seq_len = 180
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = True
    random_label_embedding = False
    single_anwser = False
    save_model = True
    use_efficient_global_pointer = False

    valid_portion = 0.25
    batch_size = 8
    gradient_accumulation_steps = 16
    max_epoches = 5
    lr = 2e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_start_epoch = 1
    early_stop = 5
    model_save_dir = 'weights/genia/best_model.genia.trained_label.pt'


class conll_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/roberta-large/'
    train_data_path = "/home/zxa/ps/open_data/ner/conll2003/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/conll2003/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/conll2003/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/conll2003/conll_vector_iter200.txt'

    loss_type = 'ce'

    max_seq_len = 200
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = True
    random_label_embedding = False
    single_anwser = False
    save_model = False

    valid_portion = 0.25
    batch_size = 8
    gradient_accumulation_steps = 16
    max_epoches = 10
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_start_epoch = 5

    use_efficient_global_pointer = False
    model_save_dir = 'weights/conll2003/best_model.conll.trained_label.pt'


class msra_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/chinese_roberta_wwm_ext_large/'
    train_data_path = "/home/zxa/ps/open_data/ner/zh_msra/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/zh_msra/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/zh_msra/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/zh_msra/msra_cut_vector.txt'

    loss_type = 'ce'

    max_seq_len = 128
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = True
    random_label_embedding = False
    single_anwser = False
    save_model = False

    valid_portion=0.25
    batch_size = 8
    gradient_accumulation_steps = 16
    max_epoches = 10
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_epoch = 1

    use_efficient_global_pointer = False
    model_save_dir = 'weights/msra/best_model.msra.trained_label.pt'


class ontonotes4_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/chinese_roberta_wwm_ext_large/'
    train_data_path = "/home/zxa/ps/open_data/ner/ontonotes4/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/ontonotes4/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/ontonotes4/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/ontonotes4/ontonotes4_vector_iter10.txt'

    loss_type = 'ce'

    max_seq_len = 128
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = False
    random_label_embedding = False
    single_anwser = False
    save_model = False

    valid_portion=1
    batch_size = 64
    gradient_accumulation_steps = 1
    max_epoches = 10
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-3
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_start_epoch = 1

    use_efficient_global_pointer = False
    model_save_dir = 'weights/ontonotes4/best_model.ontonotes4.trained_label_cut.pt'


class onenote_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/roberta-large/'
    train_data_path = "/home/zxa/ps/open_data/ner/onenotesv5/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/onenotesv5/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/onenotesv5/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/onenotesv5/onenotev5_vector.txt'

    loss_type = 'ce'

    max_seq_len = 210
    dropout_rate = 0.2
    num_labels = 8
    continue_train = False
    add_label_embedding = True
    random_label_embedding = False

    valid_portion = 0.25
    batch_size = 5
    gradient_accumulation_steps = 8
    max_epoches = 3
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.01
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_epoch = 1

    use_efficient_global_pointer = False
    model_save_dir = 'best_model.onenote.tmp.pt'


class ace2004_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/roberta-large/'
    train_data_path = "/home/zxa/ps/open_data/ner/ace2004/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/ace2004/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/ace2004/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/ace2004/ace2004_vector_iter10.txt'

    loss_type = 'ce'

    max_seq_len = 128
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = False
    random_label_embedding = False
    single_anwser = False
    save_model = True

    valid_portion = 0.25
    batch_size = 8
    gradient_accumulation_steps = 16
    max_epoches = 10
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_start_epoch = 1

    use_efficient_global_pointer = False
    model_save_dir = 'weights/ace2004/best_model.ace2004.trained_label.pt'


class ace2005_config:

    pretrained_model_path = '/home/zxa/ps/pretrain_models/roberta-large/'
    train_data_path = "/home/zxa/ps/open_data/ner/ace2005/mrc-ner.train"
    dev_data_path = "/home/zxa/ps/open_data/ner/ace2005/mrc-ner.dev"
    test_data_path = "/home/zxa/ps/open_data/ner/ace2005/mrc-ner.test"
    label_path = './label2id.json'
    label_embedding_path = '/home/zxa/ps/open_data/ner/ace2005/ace2005_vector_iter10.txt'

    loss_type = 'ce'

    max_seq_len = 128
    dropout_rate = 0.1
    num_labels = 8
    continue_train = False
    add_label_embedding = True
    random_label_embedding = False
    single_anwser = False
    save_model = True

    valid_portion = 0.25
    batch_size = 8
    gradient_accumulation_steps = 16
    max_epoches = 10
    lr = 1e-5
    lr_end = 3e-7
    other_lr = 1e-4
    weight_decay = 0.002
    adam_epsilon = 1e-8
    warmup_proportion = 0.1

    log_every = 200
    valid_start_epoch = 1

    use_efficient_global_pointer = False
    early_stop = 5
    model_save_dir = 'weights/ace2005/best_model.ace2005.trained_label.pt'