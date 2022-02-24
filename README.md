> This is the original source cod of KESA: A Knowledge Enhanced Approach For Sentiment Analysis

#### Requirements
> install Python 3.6 

> If you are using Windows, create python virtual environment by platform, e.g., Pycharm, and click install `requirements.txt`.
> 
> If your are using Ubuntu, create python virtual environment by `python3 -m venv kesa`, and next activate the virtual environment by `source kesa/bin/activate`, and then install python packages by `pip install -r requirements`.
 

#### Dataset and Rosources
> For dataset MR, SST2, SST5 and IMDB, and sentiment lexicon SentiWordNet 3.0, your can download directly from [here](https://bhpan.buaa.edu.cn:443/link/9F1FAE416742A4E4238452DE4C173E6A).
> For checkpoints, you can download from [here](https://huggingface.co/models).


#### Preprocess
> Preprocess SentimentWordNet 3.0 to get word-level polarity by `preprocess_lexicon/gen_word_level_polarity.py`.

#### Fine-tune
> You can fine-tune KESA on SST2 dataset based on checkpoints released by SentiLARE, with label combination is CC (conditional combination) using this command: 
> `python run_sentiment_classifier.py   --do_train   --do_eval  --model_type roberta_2task  --model_name_or_path  ../models/SentiLARE_pretrain_roberta   --task_name sst-2   --data_dir  ../dataset/SST_2/  --all_data_file   ../dataset/SST_2/sst.binary.all  --lexicon_file ../dataset/lexicon/SWN.word.polarity    --num_train_epochs 3.0   --per_gpu_eval_batch_size 1000  --per_gpu_train_batch_size 32   --max_seq_length 128   --learning_rate 2e-5   --loss_type aggregation   --loss_balance_type add_vec   --c 0.01  --seed  11`

> For more commands, please refer to `commands.txt`