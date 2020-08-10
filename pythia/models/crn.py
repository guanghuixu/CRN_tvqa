# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
from torch import nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer

from pythia.modules.encoders import ImageEncoder
from pythia.modules.embeddings import AttentionTextEmbedding


@registry.register_model("crn")
class CRN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.PAM_config = BertConfig(**self.config.PAM)  # Progressive Attention Module
        self.MRG_config = BertConfig(**self.config.MRG)  # Multimodal Reasoning Graph
        self.mmt_config = BertConfig(**self.config.mmt)  # Multimodal Transformer (answering module)
        self._datasets = registry.get("config").datasets.split(",")

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_model()
        self._build_output()

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='../detectron/fc6/fc7_w.pkl',
            bias_file='../detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='../detectron/fc6/fc7_w.pkl',
            bias_file='../detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        self.finetune_modules.append({
            'module': self.ocr_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_model(self):
        self.PAM_1 = Q(self.PAM_config)
        self.PAM_2 = QT(self.PAM_config)
        self.PAM_3 = QTV(self.PAM_config)
        self.MRG = MRG_Graph(self.MRG_config)
        self.mmt = MMT(self.mmt_config)

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        # only keep scores in the forward pass results
        results = {"scores": fwd_results["scores"]}
        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        fwd_results['txt_inds'] = sample_list.text

        # binary mask of valid text (question words) vs padding
        fwd_results['txt_mask'] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )

        # first forward the text BERT layers
        text_bert_out = self.text_bert(
            txt_inds=fwd_results['txt_inds'],
            txt_mask=fwd_results['txt_mask']
        )
        fwd_results['txt_emb'] = self.text_bert_out_linear(text_bert_out)

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(obj_feat)
            ) + self.obj_bbox_layer_norm(
                self.linear_obj_bbox_to_mmt_in(obj_bbox)
            )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        ocr_fc6 = sample_list.image_feature_1[:, :ocr_fasttext.size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        # OCR order vectors (legacy from LoRRA model; set to all zeros)
        # TODO remove OCR order vectors; they are not needed
        ocr_order_vectors = torch.zeros_like(sample_list.order_vectors)

        ocr_feat = torch.cat(
            [ocr_fasttext, ocr_phoc, ocr_fc7, ocr_order_vectors],
            dim=-1
        )
        ocr_bbox = sample_list.ocr_bbox_coordinates
        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
                self.linear_ocr_bbox_to_mmt_in(ocr_bbox)
            )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results['ocr_mask'] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_pam_graph(self, sample_list, fwd_results):
        self.PAM_1(fwd_results)
        self.PAM_2(fwd_results)
        self.PAM_3(fwd_results)

        self.MRG(sample_list, fwd_results)

    def _forward_mmt(self, sample_list, fwd_results):

        mmt_results = self.mmt(
            txt_emb=fwd_results['txt_emb'],
            txt_mask=fwd_results['txt_mask'],
            obj_emb=fwd_results['obj_mmt_in'],
            obj_mask=fwd_results['obj_mask'],
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results):
        mmt_dec_output = fwd_results['mmt_dec_output']
        mmt_ocr_output = fwd_results['mmt_ocr_output']
        ocr_mask = fwd_results['ocr_mask']

        fixed_scores = self.classifier(mmt_dec_output)
        dynamic_ocr_scores = self.ocr_ptr_net(
            mmt_dec_output, mmt_ocr_output, ocr_mask
        )
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results['scores'] = scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):

        self._forward_pam_graph(sample_list, fwd_results)

        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output

class Q(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        encoder_inputs = txt_emb
        attention_mask = txt_mask

        txt_max_num = txt_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        # mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output)

class QT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        obj_emb = fwd_results['ocr_mmt_in']
        obj_mask = fwd_results['ocr_mask']

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask],
            dim=1
        )

        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:])

class QTV(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, fwd_results):
        txt_emb = fwd_results['txt_emb']
        txt_mask = fwd_results['txt_mask']
        obj_emb = fwd_results['obj_mmt_in']
        obj_mask = fwd_results['obj_mask']
        ocr_emb = fwd_results['ocr_mmt_in']
        ocr_mask = fwd_results['ocr_mask']

        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        fwd_results['txt_emb'] = fwd_results['txt_emb'] + torch.tanh(mmt_seq_output[:, txt_begin:txt_end])
        fwd_results['obj_mmt_in'] = fwd_results['obj_mmt_in'] + torch.tanh(mmt_seq_output[:, txt_end:ocr_begin])
        fwd_results['ocr_mmt_in'] = fwd_results['ocr_mmt_in'] + torch.tanh(mmt_seq_output[:, ocr_begin:ocr_end])

class MRG_Graph(nn.Module):  # mimic the GRU of the GGNN
    def __init__(self, config):
        super().__init__()

        self._build_common_layer(module_name='qv', hidden_size=config.hidden_size)
        self._build_common_layer(module_name='qt', hidden_size=config.hidden_size)
        self.ques_norm = nn.LayerNorm(config.hidden_size)
    
    def _build_common_layer(self, module_name, hidden_size, edge_dim=5):
        self_attn = nn.Linear(hidden_size, 1)
        transform_edge = nn.Sequential(nn.Linear(edge_dim, hidden_size // 2),
                                        nn.ELU(),
                                        nn.Linear(hidden_size // 2, hidden_size))

        embeded = nn.Linear(hidden_size, hidden_size)
        edge_attn = nn.Linear(hidden_size, 1)

        setattr(self, '{}_self_attn'.format(module_name), self_attn)
        setattr(self, '{}_transform_edge'.format(module_name), transform_edge)
        setattr(self, '{}_embeded'.format(module_name), embeded)
        setattr(self, '{}_edge_attn'.format(module_name), edge_attn)

        # build_fc_with_layernorm
        feat_layer_1 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LayerNorm(hidden_size))
        feat_layer_2 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LayerNorm(hidden_size))
        feat_layer_3 = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                    nn.LayerNorm(hidden_size))
        input_drop = nn.Dropout(0.1)
        setattr(self, '{}_feat_layer_1'.format(module_name), feat_layer_1)
        setattr(self, '{}_feat_layer_2'.format(module_name), feat_layer_2)
        setattr(self, '{}_feat_layer_3'.format(module_name), feat_layer_3)
        setattr(self, '{}_drop'.format(module_name), input_drop)

    def _calculate_self_attn(self, ques, mask, module_name):
        attn = getattr(self, module_name+'_self_attn')(ques).squeeze(-1)  # N, 20
        attn = F.softmax(attn, dim=-1)
        attn = attn * mask
        attn = attn / attn.sum(1, keepdim=True)
        question_feature = torch.bmm(attn.unsqueeze(1), ques)  # N, 1, 768
        return question_feature
    
    def _build_compute_graph(self, edge_feat, ques, input_mask, ques_mask, module_name):
        batch, num_obj, num_subobj = edge_feat.size()[:3]
        edge_feat = getattr(self, module_name+'_transform_edge')(edge_feat)  # [N, 100, 100, 768]

        # conduct edge additive attetnion
        ques_feature = self._calculate_self_attn(ques, ques_mask, module_name)  # [N, 1, 768]
        ques_feature = getattr(self, module_name+'_embeded')(ques_feature).unsqueeze(1).expand(-1, num_obj, num_subobj, -1)
        ques_guided_edge_attn = (getattr(self, module_name+'_edge_attn')(torch.tanh(ques_feature + edge_feat))).squeeze(-1)
        A_edge_attn = F.softmax(ques_guided_edge_attn, -1)  # N, 100, 100
        A_edge_attn = A_edge_attn * input_mask.unsqueeze(-1)
        A_edge_attn = A_edge_attn / (A_edge_attn.sum(dim=-1, keepdim=True) + 1e-12)
        
        updated_edge_feat = edge_feat * A_edge_attn.unsqueeze(-1)  # N, 100*100, 768
        updated_edge_feat = updated_edge_feat.sum(2)  # [N, 100, 768]

        return A_edge_attn, updated_edge_feat
    
    def forward(self, sample_list, fwd_results):
        v_feat = fwd_results['obj_mmt_in']
        t_feat = fwd_results['ocr_mmt_in']
        v2t_edge = sample_list['obj_ocr_edge_feat']
        t2v_edge = sample_list['ocr_obj_edge_feat']
        q_emb = fwd_results['txt_emb']

        v_mask = fwd_results['obj_mask']
        t_mask = fwd_results['ocr_mask']
        q_mask = fwd_results['txt_mask']

        q_emb = self.ques_norm(q_emb)

        v2t_attn, v2t_feat = self._build_compute_graph(v2t_edge, q_emb, v_mask, q_mask, module_name='qv')
        t2v_attn, t2v_feat = self._build_compute_graph(t2v_edge, q_emb, t_mask, q_mask, module_name='qt')
        v2t_mask = torch.bmm(v_mask.unsqueeze(-1), t_mask.unsqueeze(1))  # [N, 100, 50]
        t2v_mask = v2t_mask.transpose(1, 2)
        v2t_attn = v2t_attn * v2t_mask
        t2v_attn = t2v_attn * t2v_mask

        new_t_feat = torch.bmm(v2t_attn.transpose(1, 2), v_feat)  # batch, 100, 768
        new_v_feat = torch.bmm(t2v_attn.transpose(1, 2), t_feat)  # batch, 100, 768

        v_feat = self.qv_feat_layer_1(v_feat) + self.qv_feat_layer_2(new_v_feat) + self.qv_feat_layer_3(v2t_feat)
        t_feat = self.qt_feat_layer_1(t_feat) + self.qt_feat_layer_2(new_t_feat) + self.qt_feat_layer_3(t2v_feat)
        
        fwd_results['txt_emb'] = q_emb
        fwd_results['obj_mmt_in'] = self.qv_drop(v_feat)
        fwd_results['ocr_mmt_in'] = self.qt_drop(t_feat)

class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                txt_emb,
                txt_mask,
                obj_emb,
                obj_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )
        encoder_inputs = torch.cat(
            [txt_emb, obj_emb, ocr_emb, dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [txt_mask, obj_mask, ocr_mask, dec_mask],
            dim=1
        )

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
        mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        results = {
            'mmt_seq_output': mmt_seq_output,
            'mmt_txt_output': mmt_txt_output,
            'mmt_ocr_output': mmt_ocr_output,
            'mmt_dec_output': mmt_dec_output,
        }
        return results

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results

