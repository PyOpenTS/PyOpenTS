import numpy as np
import scipy.spatial.distance as spd
import torch

import libmr


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):

        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        if len(mean) == 0 or len(dist) == 0:
            continue
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    num_classes = len(categories)
    # select the right label's every channel score which are max score of scores. 
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    # the labels are argsorted highly, they will obtain huge weights.
    # alpha_weight: [3/3, 2/3, 1/3]
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(num_classes)
    omega[ranked_list] = alpha_weights
    # we will change the scores and store uncertainty scores
    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)

        scores.append(score_channel)
        scores_u.append(score_channel_u)

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    # eu_dists: euclidean distance; cos_dists: cosine distance; eucos_dists: euclidean distance and cosine distance
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        if len(mcv) == 0:
            eu_dists.append([])
            eucos_dists.append([])
            cos_dists.append([])
            continue
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_mavs_and_dists(num_classes, train_dataloader, device, model):
    dummy_scores = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for batch_idx, (batch_inputs, batch_labels) in enumerate(train_dataloader):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            # this must cause error for cifar
            logits = model(batch_inputs)
            for score, label in zip(logits, batch_labels):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == label:
                    dummy_scores[label].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    # scores: argmax logits is equal to label, the scores are the stored logits
    # [400, 1 ,15](N, 1, Channel)
    scores = []
    for score_list in dummy_scores:
        if score_list:
            scores.append(torch.cat(score_list).cpu().numpy())
        else:
            scores.append([])
    # mavs: mean scores(every score in scores has been successfully predicted in the right labels which have all label)
    # [1, 15](1, channel)
    mavs = []
    for score in scores:
        if len(score) > 0:
            mavs.append(np.mean(score, axis=0))
        else:
            mavs.append([])
    # mcv: mean channel vector
    # dist: 
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]
    return mavs, dists
