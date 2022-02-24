import torch, math
from torch import nn

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import CrossEntropyLoss
from modeling_roberta import RobertaModel


def _gelu_python(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


XLNetLayerNorm = nn.LayerNorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.candidate_size = 2
        self.word_polarity_num = 3
        self.polarity_embedding = nn.Embedding(self.word_polarity_num, config.hidden_size)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.fc = nn.Linear(config.hidden_size * self.candidate_size, config.num_labels * self.candidate_size)
        self.fc_2 = nn.Linear(config.hidden_size, config.num_labels * self.word_polarity_num)

        self.fc_3 = nn.Linear(2, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.fc_3.weight)
        torch.nn.init.zeros_(self.fc_3.bias)

        self.loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=-100)  # 默认 0.5857

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                masked_input_ids=None, masked_attention_mask=None, candidate_words=None,
                masked_lm_labels=None, word_polarity=None, pretrain=False):

        word_polarity_embedding = self.polarity_embedding(word_polarity)

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               candidate_words=candidate_words,
                               mask=True)
        sequence_output = outputs[0]  # batch L d
        x = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.unsqueeze(1)  # batch 1 d
        x = self.dropout(x)  # 提出sentiment word的剩余句子的表示


        candidate_word_embedding = outputs[-1]  # batch C d
        # cls_with_word = candidate_word_embedding + x  # batch C d 相当于补全句子信息
        cls_with_word = candidate_word_embedding + word_polarity_embedding + x  # batch C d 相当于补全句子信息

        # pretext task 1: 判断word是否在句子中
        swc_preds = self.fc(cls_with_word.view(cls_with_word.size(0), -1))  # batch C1*C2 类别数：句子类别*word类别
        ### 第一种：joint loss
        if self.config.loss_type == "joint":
            joint_label = labels * self.candidate_size + masked_lm_labels
            swc_loss = self.loss_fct(swc_preds.view(-1, self.candidate_size * self.num_labels), joint_label)
        elif self.config.loss_type == 'aggregation':
            ### 第二种：aggregation loss
            wordin_logits = swc_preds[::, ::self.num_labels]  # 隔行采样 # batch C1 word类别数
            swc_loss = self.loss_fct(wordin_logits.view(-1, self.candidate_size), masked_lm_labels.view(-1))


        ## pretext 2
        true_word_embed = candidate_word_embedding[range(masked_lm_labels.size(0)), masked_lm_labels]  # batch L d -> batch d
        true_word_polarity_embed = word_polarity_embedding[range(masked_lm_labels.size(0)), masked_lm_labels]  # batch L d -> batch d
        # sentence_representation = x.squeeze(1) + true_word_embed  # 强调
        sentence_representation = x.squeeze(1) + true_word_embed + true_word_polarity_embed # 强调
        csp_preds = self.fc_2(sentence_representation)  # batch, C1*C2word极性*sentence极性
        if self.config.loss_type == "joint":
            ### 第一种： joint loss
            true_word_polarity = word_polarity[range(masked_lm_labels.size(0)), masked_lm_labels]
            joint_label = true_word_polarity * self.num_labels + labels
            csp_loss = self.loss_fct(csp_preds, joint_label)
        elif self.config.loss_type == 'aggregation':
            ### 第二种：aggregation loss
            single_logits = csp_preds[::, ::self.word_polarity_num]
            csp_loss = self.loss_fct(single_logits, labels)

        # combine task 1 and task 2
        if self.config.loss_balance_type == "weight_sum":
            lexicon_loss = self.config.a * swc_loss + self.config.b * csp_loss # pretext 1 + pretext 2
        elif self.config.loss_balance_type == "add_vec":
            a = swc_preds[::, ::self.candidate_size] # batch, label_num
            b = csp_preds[::, ::self.word_polarity_num] # batch, label_num

            t = torch.cat([a.unsqueeze(1), b.unsqueeze(1)], dim=1).permute(0, 2, 1)
            t = self.fc_3(t).squeeze(-1)
            lexicon_loss = self.config.c * self.loss_fct(t, labels)

        # 有监督
        logits = self.classifier(sequence_output)  # batch C2 句子类别数
        supervised_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 监督损失

        output = (supervised_loss, lexicon_loss,) + (logits,)

        return output

