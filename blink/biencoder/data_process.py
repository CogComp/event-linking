# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from pytorch_transformers.tokenization_bert import BertTokenizer

from blink.biencoder.zeshel_utils import world_to_id
from blink.common.params import ENT_START_TAG, ENT_END_TAG, ENT_TITLE_TAG, TYPE_TAG_MAPPING


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]
        # mention_tokens = [ent_start_token] + ['[MASK]'] + [ent_end_token]

    entity_tokens = []
    if 'entities' in sample and len(sample['entities']) > 0:
        for entity in sample['entities'][0:10]:
            tokens = tokenizer.tokenize(entity['form'])
            tokens = [TYPE_TAG_MAPPING[entity['type']][0]] + tokens + [TYPE_TAG_MAPPING[entity['type']][1]]
            # tokens = ["[SEP]"] + tokens
            entity_tokens += tokens

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens) - len(entity_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - len(entity_tokens) - left_quota - 2

    if left_quota <= 0 or right_quota <= 0:
        left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
        right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
        entity_tokens = []

    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )

    context_tokens = ["[CLS]"] + context_tokens + entity_tokens + ["[SEP]"]
    # print(context_tokens)
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    if len(input_ids) != max_seq_length:
        print(max_seq_length, len(mention_tokens))
        print(left_quota, right_quota)
        print(mention_tokens)
        print(entity_tokens)
        print(len(entity_tokens))
        print(context_left)
        print(context_right)
        print(context_tokens)
        print(len(input_ids), max_seq_length)
    # else:
    #     print(len(input_ids), max_seq_length)
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, 
    tokenizer, 
    max_seq_length, 
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
    hyperlinks=None
):

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    cand_tokens = tokenizer.tokenize(candidate_desc)
    title_tokens = []
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        title_tokens = title_tokens + [title_tag]

    link_tokens = []
    # print(hyperlinks)

    if hyperlinks is not None:
        for link in hyperlinks:
            tokens = tokenizer.tokenize(link[1])
            tokens = ["[SEP]"] + tokens
            link_tokens += tokens

    # print(link_tokens)

    cand_tokens = cand_tokens[: max_seq_length - len(title_tokens) - len(link_tokens) - 2]
    cand_tokens = [cls_token] + title_tokens + cand_tokens + link_tokens + [sep_token]

    # print(cand_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    if len(input_ids) != max_seq_length:
        print(title_tokens)
        print(len(title_tokens))
        print(link_tokens)
        print(len(link_tokens))
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    mention_key="mention",
    context_key="context",
    label_key="label",
    title_key='label_title',
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    use_world = True

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        label = sample[label_key]
        title = sample.get(title_key, None)
        if 'hyperlinks' in sample:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title, hyperlinks=sample['hyperlinks']
            )
        else:
            label_tokens = get_candidate_representation(
                label, tokenizer, max_cand_length, title
            )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
        }

        if "world" in sample:
            src = sample["world"]
            src = world_to_id[src]
            record["src"] = [src]
            use_world = True
        else:
            use_world = False

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info(
                "Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]])
            )
            # logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"), dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"), dtype=torch.long,
    )
    if use_world:
        src_vecs = torch.tensor(
            select_field(processed_samples, "src"), dtype=torch.long,
        )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"), dtype=torch.long,
    )
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
    }

    if use_world:
        data["src"] = src_vecs
        tensor_data = TensorDataset(context_vecs, cand_vecs, src_vecs, label_idx)
    else:
        tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx)
    return data, tensor_data
