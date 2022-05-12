import pdb
import os
import json
from tqdm import tqdm
import torch
import spacy
from time import time

import torch.nn.functional as F
from model.masking import mask, multi_make_masks
from evaluation.evaluate import ANETcaptions
from datasets.load_features import load_features_from_npy
from utilities.captioning_utils_fcos import HiddenPrints, get_lr
from epoch_loops.proposal_epoch_loops_fcos import one_make_masks, sig_make_masks, upsample


def calculate_metrics(
        reference_paths, submission_path, tIoUs, max_prop_per_vid, verbose=True, only_proposals=False
):
    metrics = {}
    PREDICTION_FIELDS = ['results', 'version', 'external_data']
    evaluator = ANETcaptions(
        reference_paths, submission_path, tIoUs,
        max_prop_per_vid, PREDICTION_FIELDS, verbose, only_proposals)
    evaluator.evaluate()

    for i, tiou in enumerate(tIoUs):
        metrics[tiou] = {}

        for metric in evaluator.scores:
            score = evaluator.scores[metric][i]
            metrics[tiou][metric] = score

    # Print the averages

    metrics['Average across tIoUs'] = {}
    for metric in evaluator.scores:
        score = evaluator.scores[metric]
        metrics['Average across tIoUs'][metric] = sum(score) / float(len(score))

    return metrics


def greedy_decoder(cfg, model, feature_stacks, max_len, start_idx, end_idx, pad_idx, modality):
    assert model.training is False, 'call model.eval first'

    with torch.no_grad():

        if 'audio' in modality:
            B, _Sa_, _Da_ = feature_stacks['audio'].shape
            device = feature_stacks['audio'].device
        elif modality == 'video':
            B, _Sv_, _Drgb_ = feature_stacks['rgb'].shape
            device = feature_stacks['rgb'].device
        else:
            raise Exception(f'Unknown modality: {modality}')

        # a mask containing 1s if the ending tok occured, 0s otherwise
        # we are going to stop if ending token occured in every sequence
        completeness_mask = torch.zeros(B, 1).byte().to(device)
        trg = (torch.ones(B, 1) * start_idx).long().to(device)

        batch_model_a = feature_stacks['audio']
        fea_stack_a = upsample(batch_model_a, cfg.scale_audio)

        batch_model_v = feature_stacks['rgb'] + feature_stacks['flow']
        fea_stack_v = upsample(batch_model_v, cfg.scale_video)

        fea_stack_t = feature_stacks['text']

        s_a = 1200 - fea_stack_a.shape[1]
        p1d = [0, 0, 0, s_a]
        fea_stack_a = F.pad(fea_stack_a, p1d, value=pad_idx)

        s_v = 1200 - fea_stack_v.shape[1]
        p2d = [0, 0, 0, s_v]
        fea_stack_v = F.pad(fea_stack_v, p2d, value=pad_idx)

        s_t = 1200 - fea_stack_t.shape[1]
        p3d = [0, 0, 0, s_t]
        fea_stack_t = F.pad(fea_stack_t, p3d, value=pad_idx)

        # if fea_stack_a.shape[1] == fea_stack_v.shape[1]:
        #     pass
        # elif fea_stack_a.shape[1] < fea_stack_v.shape[1]:
        #     s = fea_stack_v.shape[1] - fea_stack_a.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_a = F.pad(fea_stack_a, p1d, value=pad_idx)
        # elif fea_stack_a.shape[1] > fea_stack_v.shape[1]:
        #     s = fea_stack_a.shape[1] - fea_stack_v.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_v = F.pad(fea_stack_v, p1d, value=pad_idx)
        #
        # fea_stack_t = feature_stacks['text']
        # if fea_stack_t.shape[1] == fea_stack_v.shape[1]:
        #     pass
        # elif fea_stack_t.shape[1] < fea_stack_v.shape[1]:
        #     s = fea_stack_v.shape[1] - fea_stack_t.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_t = F.pad(fea_stack_t, p1d, value=pad_idx)
        # elif fea_stack_t.shape[1] > fea_stack_v.shape[1]:
        #     s = fea_stack_t.shape[1] - fea_stack_v.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_v = F.pad(fea_stack_v, p1d, value=pad_idx)
        #     fea_stack_a = F.pad(fea_stack_a, p1d, value=pad_idx)

        if cfg.debug: print('Input feature shape \t\t', fea_stack_a.shape, fea_stack_v.shape, fea_stack_t.shape)
        masks = multi_make_masks((fea_stack_a, fea_stack_v, fea_stack_t), trg, pad_idx)

        while (trg.size(-1) <= max_len) and (not completeness_mask.all()):
            preds = model((fea_stack_a, fea_stack_v, fea_stack_t), trg, masks)  # 下一个单词的概率分布？？
            next_word = preds[:, -1].max(dim=-1)[1].unsqueeze(1)  # 取出概率值最大的单词的序号
            trg = torch.cat([trg, next_word], dim=-1)
            completeness_mask = completeness_mask | torch.eq(next_word, end_idx).byte()

        return trg  # 片段对应的caption序列


def save_model(cfg, epoch, model, optimizer,
               val_1_metrics, val_2_metrics, trg_voc_size):
    dict_to_save = {
        'config': cfg,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_1_metrics': val_1_metrics,
        'val_2_metrics': val_2_metrics,
        'trg_voc_size': trg_voc_size,
    }

    # in case TBoard is not defined make logdir (can be deleted if Config is used)
    os.makedirs(cfg.model_checkpoint_path, exist_ok=True)

    # path_to_save = os.path.join(cfg.model_checkpoint_path, f'model_e{epoch}.pt')
    path_to_save = os.path.join(cfg.model_checkpoint_path, f'best_cap_model.pt')
    torch.save(dict_to_save, path_to_save)


def multi_training_loop_fcos(cfg, model, train_loader, criterion, optimizer, epoch, TBoard):
    model.train()
    train_total_loss = 0
    train_loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'
    # assert cfg.modality == 'audio_video_text'

    for i, batch in enumerate(tqdm(train_loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption  # word idx
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]

        batch_model_a = batch['feature_stacks']['audio']
        fea_stack_a = upsample(batch_model_a, cfg.scale_audio)

        batch_model_v = batch['feature_stacks']['rgb'] + batch['feature_stacks']['flow']
        fea_stack_v = upsample(batch_model_v, cfg.scale_video)

        fea_stack_t = batch['feature_stacks']['text']

        s_a = 1200 - fea_stack_a.shape[1]
        p1d = [0, 0, 0, s_a]
        fea_stack_a = F.pad(fea_stack_a, p1d, value=train_loader.dataset.pad_idx)

        s_v = 1200 - fea_stack_v.shape[1]
        p2d = [0, 0, 0, s_v]
        fea_stack_v = F.pad(fea_stack_v, p2d, value=train_loader.dataset.pad_idx)

        s_t = 1200 - fea_stack_t.shape[1]
        p3d = [0, 0, 0, s_t]
        fea_stack_t = F.pad(fea_stack_t, p3d, value=train_loader.dataset.pad_idx)

        fea_stack_g = batch['feature_stacks']['global']
        # if fea_stack_a.shape[1] == fea_stack_v.shape[1]:
        #     pass
        # elif fea_stack_a.shape[1] < fea_stack_v.shape[1]:
        #     s = fea_stack_v.shape[1] - fea_stack_a.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_a = F.pad(fea_stack_a, p1d, value=train_loader.dataset.pad_idx)
        # elif fea_stack_a.shape[1] > fea_stack_v.shape[1]:
        #     s = fea_stack_a.shape[1] - fea_stack_v.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_v = F.pad(fea_stack_v, p1d, value=train_loader.dataset.pad_idx)
        #
        # fea_stack_t = batch['feature_stacks']['text']
        # if fea_stack_t.shape[1] == fea_stack_v.shape[1]:
        #     pass
        # elif fea_stack_t.shape[1] < fea_stack_v.shape[1]:
        #     s = fea_stack_v.shape[1] - fea_stack_t.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_t = F.pad(fea_stack_t, p1d, value=train_loader.dataset.pad_idx)
        # elif fea_stack_t.shape[1] > fea_stack_v.shape[1]:
        #     s = fea_stack_t.shape[1] - fea_stack_v.shape[1]
        #     p1d = [0, 0, 0, s]
        #     fea_stack_v = F.pad(fea_stack_v, p1d, value=train_loader.dataset.pad_idx)
        #     fea_stack_a = F.pad(fea_stack_a, p1d, value=train_loader.dataset.pad_idx)

        if cfg.debug: print('Input feature shape \t\t', fea_stack_a.shape, fea_stack_v.shape, fea_stack_t.shape)
        masks = multi_make_masks((fea_stack_a, fea_stack_v, fea_stack_t, fea_stack_g), caption_idx,
                                 train_loader.dataset.pad_idx)
        pred = model((fea_stack_a, fea_stack_v, fea_stack_t, fea_stack_g), caption_idx, masks)
        n_tokens = (caption_idx_y != train_loader.dataset.pad_idx).sum()
        loss = criterion(pred, caption_idx_y) / n_tokens

        print('------------------------Loss Backward!---------------------')
        print('batch_loss:\n', loss)
        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

        if i % 10 == 0:
            if TBoard is not None:
                TBoard.add_scalar(f'debug/loss_batch_10_avg_{epoch}', loss.item(), i)

    train_total_loss_norm = train_total_loss / len(train_loader)

    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)


def training_loop_fcos(cfg, model, loader, criterion, optimizer, epoch, TBoard):
    model.train()
    train_total_loss = 0
    loader.dataset.update_iterator()
    progress_bar_name = f'{cfg.curr_time[2:]}: train {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        optimizer.zero_grad()
        caption_idx = batch['caption_data'].caption  # word idx
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]

        upsflag = False
        upscale = None
        batch_model = None
        if cfg.modality == 'audio':
            batch_feature_stack = None
            # 3
            upscale = cfg.scale_audio
            # (B, 800, 128)
            batch_model = batch['feature_stacks']['audio']
            upsflag = True
        elif cfg.modality == 'video':
            batch_feature_stack = None
            # 8
            upscale = cfg.scale_video
            # (B, 300, 1024)
            batch_model = batch['feature_stacks']['rgb'] + batch['feature_stacks']['flow']
            upsflag = True
        elif cfg.modality == 'text':
            # (B, 2400, 300)
            batch_feature_stack = batch['feature_stacks']['text']
        else:
            raise NotImplemented

        if upsflag:
            batch_feature_stack = upsample(batch_model, upscale)

        if cfg.debug: print(f'batch["video_ids"]:{batch["video_ids"]}')
        # print(batch_feature_stack.shape, batch_feature_stack)
        masks = sig_make_masks(batch_feature_stack, caption_idx, cfg.modality,
                               loader.dataset.pad_idx)
        pred = model(batch_feature_stack, caption_idx, masks)
        n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
        loss = criterion(pred, caption_idx_y) / n_tokens

        loss.backward()

        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()

        train_total_loss += loss.item()

    train_total_loss_norm = train_total_loss / len(loader)

    if TBoard is not None:
        TBoard.add_scalar('debug/train_loss_epoch', train_total_loss_norm, epoch)
        TBoard.add_scalar('debug/lr', get_lr(optimizer), epoch)


def multi_validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]

        batch_model_a = batch['feature_stacks']['audio']
        fea_stack_a = upsample(batch_model_a, cfg.scale_audio)

        batch_model_v = batch['feature_stacks']['rgb'] + batch['feature_stacks']['flow']
        fea_stack_v = upsample(batch_model_v, cfg.scale_video)

        if fea_stack_a.shape[1] == fea_stack_v.shape[1]:
            pass
        elif fea_stack_a.shape[1] < fea_stack_v.shape[1]:
            s = fea_stack_v.shape[1] - fea_stack_a.shape[1]
            p1d = [0, 0, 0, s]
            fea_stack_a = F.pad(fea_stack_a, p1d, value=loader.dataset.pad_idx)
        elif fea_stack_a.shape[1] > fea_stack_v.shape[1]:
            s = fea_stack_a.shape[1] - fea_stack_v.shape[1]
            p1d = [0, 0, 0, s]
            fea_stack_v = F.pad(fea_stack_v, p1d, value=loader.dataset.pad_idx)

        if cfg.modality == 'audio_video_text':
            fea_stack_t = batch['feature_stacks']['text']
        else:
            fea_stack_t = None

        if cfg.debug: print(f'batch["video_ids"]:{batch["video_ids"]}')
        masks = one_make_masks((fea_stack_a, fea_stack_v), caption_idx, cfg.modality, loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model((fea_stack_a, fea_stack_v, fea_stack_t), caption_idx, masks)
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(pred, caption_idx_y) / n_tokens
            val_total_loss += loss.item()

    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm


def validation_next_word_loop(cfg, model, loader, decoder, criterion, epoch, TBoard, exp_name):
    model.eval()
    val_total_loss = 0
    loader.dataset.update_iterator()
    phase = loader.dataset.phase
    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        caption_idx = batch['caption_data'].caption
        caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]

        upsflag = False
        upscale = None
        if cfg.modality == 'audio':
            # 3
            upscale = cfg.scale_audio
            # (B, 800, 128)
            batch_model = batch['feature_stacks']['audio']
            upsflag = True
        elif cfg.modality == 'video':
            # 8
            upscale = cfg.scale_video
            # (B, 300, 1024)
            batch_model = batch['feature_stacks']['rgb'] + batch['feature_stacks']['flow']
            upsflag = True
        elif cfg.modality == 'text':
            # (B, 2400, 300)
            batch_model = batch['feature_stacks']['text']
        else:
            raise NotImplemented

        if cfg.debug: print('upsampling before feature:\n', batch_model[0])
        batch_feature_stack = None
        if upsflag:
            batch_feature_stack = upsample(batch_model, upscale)
        if cfg.debug: print('upsampling after feature:\n', batch_feature_stack)

        if cfg.debug: print(f'batch["video_ids"]:{batch["video_ids"]}')
        masks = sig_make_masks(batch_feature_stack, caption_idx, cfg.modality,
                               loader.dataset.pad_idx)

        with torch.no_grad():
            pred = model(batch_feature_stack, caption_idx, masks)
            n_tokens = (caption_idx_y != loader.dataset.pad_idx).sum()
            loss = criterion(pred, caption_idx_y) / n_tokens
            val_total_loss += loss.item()

    val_total_loss_norm = val_total_loss / len(loader)

    return val_total_loss_norm


def validation_1by1_loop(cfg, model, loader, decoder, epoch, TBoard):
    start_timer = time()

    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True,
            'details': ''
        },
        'results': {}
    }
    model.eval()
    loader.dataset.update_iterator()

    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names

    if phase == 'val_1':
        reference_paths = [cfg.val_reference_paths[0]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'val_2':
        reference_paths = [cfg.val_reference_paths[1]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'learned_props':
        reference_paths = cfg.val_reference_paths  # here we use all of them
        tIoUs = cfg.tIoUs
        # assert len(tIoUs) == 4
    else:
        raise NotImplemented

    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} 1by1 {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        # caption_idx = batch['caption_data'].caption
        # caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        # PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        ints_stack = decoder(
            cfg, model, batch['feature_stacks'], cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality
        )
        if cfg.debug: print('inis_stack.shape:\n', ints_stack.shape)
        ints_stack = ints_stack.cpu().numpy()  # what happens here if I use only cpu?
        # transform integers into strings
        # print('loader.dataset.train_vocab:\n', len(loader.dataset.train_vocab))
        list_of_lists_with_strings = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack]
        ### FILTER PREDICTED TOKENS
        # initialize the list to fill it using indices instead of appending them
        list_of_lists_with_filtered_sentences = [None] * len(list_of_lists_with_strings)

        for b, strings in enumerate(list_of_lists_with_strings):
            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()
            # add the filtered sentense to the list
            list_of_lists_with_filtered_sentences[b] = sentence

        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sent in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                              list_of_lists_with_filtered_sentences):
            segment = {
                'sentence': sent,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)

            else:
                predictions['results'][video_id] = [segment]

    if cfg.log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        save_filename = f'captioning_results_{phase}_e{epoch}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # if this is run with another loader and pretrained model
        # it substitutes the previous prediction
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        ## RUN THE EVALUATION
        # blocks the printing
        with HiddenPrints():
            val_metrics = calculate_metrics(reference_paths, submission_path, tIoUs, cfg.max_prop_per_vid)

        if phase == 'learned_props':
            print(submission_path)

        ## WRITE TBOARD
        if (TBoard is not None) and (phase != 'learned_props'):
            # todo: add info that this metrics are calculated on val_1
            TBoard.add_scalar(f'{phase}/meteor', val_metrics['Average across tIoUs']['METEOR'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu4', val_metrics['Average across tIoUs']['Bleu_4'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/bleu3', val_metrics['Average across tIoUs']['Bleu_3'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/precision', val_metrics['Average across tIoUs']['Precision'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/recall', val_metrics['Average across tIoUs']['Recall'] * 100, epoch)
            TBoard.add_scalar(f'{phase}/duration_of_1by1', (time() - start_timer) / 60, epoch)

        return val_metrics


def validation_1by1_loop_debug(cfg, model, loader, decoder, epoch, TBoard, i):
    start_timer = time()

    # init the dict with results and other technical info
    predictions = {
        'version': 'VERSION 1.0',
        'external_data': {
            'used': True,
            'details': ''
        },
        'results': {}
    }
    model.eval()
    loader.dataset.update_iterator()

    start_idx = loader.dataset.start_idx
    end_idx = loader.dataset.end_idx
    pad_idx = loader.dataset.pad_idx
    phase = loader.dataset.phase
    # feature_names = loader.dataset.feature_names

    # if phase == 'val_1':
    #     reference_paths = [cfg.reference_paths[0]]
    #     tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    # elif phase == 'val_2':
    #     reference_paths = [cfg.reference_paths[1]]
    #     tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    # elif phase == 'learned_props':
    #     reference_paths = cfg.reference_paths  # here we use all of them
    #     tIoUs = cfg.tIoUs
    #     assert len(tIoUs) == 4
    # else:
    #     raise NotImplemented
    if phase == 'val_1':
        reference_paths = [cfg.reference_paths_subs[0]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'val_2':
        reference_paths = [cfg.reference_paths_subs[1]]
        tIoUs = [0.5]  # no need to wait: they all the same as they are predicted for gt segments
    elif phase == 'learned_props':
        reference_paths = cfg.reference_paths_subs  # here we use all of them
        tIoUs = cfg.tIoUs
        assert len(tIoUs) == 4
    else:
        raise NotImplemented

    progress_bar_name = f'{cfg.curr_time[2:]}: {phase} 1by1 {epoch} @ {cfg.device}'

    for i, batch in enumerate(tqdm(loader, desc=progress_bar_name)):
        # caption_idx = batch['caption_data'].caption
        # caption_idx, caption_idx_y = caption_idx[:, :-1], caption_idx[:, 1:]
        # PREDICT TOKENS ONE-BY-ONE AND TRANSFORM THEM INTO STRINGS TO FORM A SENTENCE
        ints_stack = decoder(
            cfg, model, batch['feature_stacks'], cfg.max_len, start_idx, end_idx, pad_idx, cfg.modality
        )
        print('inis_stack.shape:\n', ints_stack.shape)
        ints_stack = ints_stack.cpu().numpy()  # what happens here if I use only cpu?
        # transform integers into strings
        list_of_lists_with_strings = [[loader.dataset.train_vocab.itos[i] for i in ints] for ints in ints_stack]
        ### FILTER PREDICTED TOKENS
        # initialize the list to fill it using indices instead of appending them
        list_of_lists_with_filtered_sentences = [None] * len(list_of_lists_with_strings)

        for b, strings in enumerate(list_of_lists_with_strings):
            # remove starting token
            strings = strings[1:]
            # and remove everything after ending token
            # sometimes it is not in the list
            try:
                first_entry_of_eos = strings.index('</s>')
                strings = strings[:first_entry_of_eos]
            except ValueError:
                pass
            # remove the period at the eos, if it is at the end (safe)
            # if trg_strings[-1] == '.':
            #     trg_strings = trg_strings[:-1]
            # join everything together
            sentence = ' '.join(strings)
            # Capitalize the sentence
            sentence = sentence.capitalize()
            # add the filtered sentense to the list
            list_of_lists_with_filtered_sentences[b] = sentence

        ### ADDING RESULTS TO THE DICT WITH RESULTS
        for video_id, start, end, sent in zip(batch['video_ids'], batch['starts'], batch['ends'],
                                              list_of_lists_with_filtered_sentences):
            segment = {
                'sentence': sent,
                'timestamp': [start.item(), end.item()]
            }

            if predictions['results'].get(video_id):
                predictions['results'][video_id].append(segment)

            else:
                predictions['results'][video_id] = [segment]

    if cfg.log_path is None:
        return None
    else:
        ## SAVING THE RESULTS IN A JSON FILE
        save_filename = f'captioning_results_{phase}_e{epoch}.json'
        submission_path = os.path.join(cfg.log_path, save_filename)

        # in case TBoard is not defined make logdir
        os.makedirs(cfg.log_path, exist_ok=True)

        # if this is run with another loader and pretrained model
        # it substitutes the previous prediction
        if os.path.exists(submission_path):
            submission_path = submission_path.replace('.json', f'_{time()}.json')

        with open(submission_path, 'w') as outf:
            json.dump(predictions, outf)

        ## RUN THE EVALUATION
        # blocks the printing
        with HiddenPrints():
            val_metrics = calculate_metrics(reference_paths, submission_path, tIoUs, cfg.max_prop_per_vid)

        if phase == 'learned_props':
            print(submission_path)

        ## WRITE TBOARD
        if (TBoard is not None) and (phase != 'learned_props'):
            # todo: add info that this metrics are calculated on val_1
            TBoard.add_scalar(f'{phase}/meteor_{epoch}', val_metrics['Average across tIoUs']['METEOR'] * 100, i)
            TBoard.add_scalar(f'{phase}/bleu4_{epoch}', val_metrics['Average across tIoUs']['Bleu_4'] * 100, i)
            TBoard.add_scalar(f'{phase}/bleu3_{epoch}', val_metrics['Average across tIoUs']['Bleu_3'] * 100, i)
            TBoard.add_scalar(f'{phase}/precision_{epoch}', val_metrics['Average across tIoUs']['Precision'] * 100, i)
            TBoard.add_scalar(f'{phase}/recall_{epoch}', val_metrics['Average across tIoUs']['Recall'] * 100, i)
            TBoard.add_scalar(f'{phase}/duration_of_1by1_{epoch}', (time() - start_timer) / 60, i)
        model.train()

        return val_metrics
