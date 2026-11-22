# Per-query scoring for LogCL. The official test pools all queries of a snapshot into one query mask and builds the inverse subgraph from the
# snapshot's subjects, which are the answers of the inverse queries. FC_SINGLE scores every unique (s, r) from that query alone: its query mask and its
# history subgraph (same rule as the official helper, on the same history). FC_SINGLE=-1 scores all queries, K>0 samples K per snapshot and direction.
# A lone triple is duplicated because the official ConvTransE skips BatchNorm when the batch has one element; the duplicate's rank is dropped.
# Ranking and filtering are the official code.
p = 'src/main.py'; s = open(p).read()
assert 'FC_SINGLE' not in s
a = "        ranks_raw.append(rank_raw)\n"
assert s.count(a) == 1
s = s.replace(a, '''        _K = int(_o.environ.get('FC_SINGLE', 0))
        if _K:
            for _snap, _rk, _tag in ((test_snap, rank_filter, 'for'), (inverse_triples, rank_filter_inv, 'inv')):
                _groups = {}
                for _i, (_a, _b, _c) in enumerate(_snap.tolist()):
                    _groups.setdefault((_a, _b), []).append(_i)
                _keys = list(_groups)
                if _K > 0 and len(_keys) > _K:
                    _keys = [_keys[_j] for _j in _np.random.choice(len(_keys), _K, replace=False)]
                for _key in _keys:
                    _idx = _groups[_key]; _tr = _snap[_idx]
                    _in = _np.concatenate([_tr, _tr]) if len(_idx) == 1 else _tr
                    _q = e2r(_in, num_rels)
                    _g = _fc_graph(subg_arr, sr_to_sro, _key, num_nodes, num_rels, use_cuda, args.gpu)
                    _t = torch.LongTensor(_in).cuda()
                    _tt, _sc = model.predict(_q, _g, time_idx, history_glist, num_rels, static_graph, _t, use_cuda)
                    _, _, _, _rs = utils.get_total_rank(_tt, _sc, all_ans_list[time_idx], eval_bz=1000, rel_predict=0)
                    for _m, _i in enumerate(_idx):
                        _FC.append((_tag, float(_rk[_i].item()), float(_rs[_m].item())))
''' + a)
a = '    mrr_raw,hit_raw = utils.stat_ranks(ranks_raw, "raw")\n'
assert s.count(a) == 1
s = s.replace(a, '''    if _FC:
        for _t in ('for', 'inv'):
            _x = [r for r in _FC if r[0] == _t]
            print('FC_SINGLE', _t, 'n', len(_x), 'official_mrr', _np.mean([1 / r[1] for r in _x]), 'single_mrr', _np.mean([1 / r[2] for r in _x]), *[f'single_h{k} {_np.mean([r[2] <= k for r in _x])}' for k in (1, 3, 10)], flush=True)
''' + a)
a = 'def test(model,'
assert s.count(a) == 1
s = s.replace(a, '''_FC = []; _FCI = {}
def _fc_graph(subg_arr, sr_to_sro, key, num_nodes, num_rels, use_cuda, gpu):
    """the query's entity plus its past answers, then all their history edges with frequencies, as in get_sample_from_history_graph3"""
    if 'a' not in _FCI:
        inv = subg_arr[:, [2, 1, 0]].copy(); inv[:, 1] = inv[:, 1] + num_rels
        tri = np.unique(np.concatenate([subg_arr, inv]), axis=0, return_counts=True)
        arr = np.concatenate([tri[0], tri[1][:, None]], 1); arr = arr[np.argsort(arr[:, 0], kind='stable')]
        _FCI['a'] = arr; _FCI['s'] = np.searchsorted(arr[:, 0], np.arange(num_nodes + 1))
    ents = {key[0]} | set(sr_to_sro.get(key, ()))
    q_tri = np.concatenate([_FCI['a'][_FCI['s'][e]:_FCI['s'][e + 1]] for e in ents])
    return build_graph(num_nodes, num_rels, q_tri, use_cuda, gpu)
''' + a)
a = '    model_state_file = \'./models/\' + model_name+ ".pt"\n'
assert s.count(a) == 1
s = s.replace(a, a + "    model_state_file = _o.environ.get('FC_MODEL', model_state_file)\n    if _o.environ.get('FC_VAL'):\n        test_list, all_ans_list_test, all_ans_list_r_test = valid_list, all_ans_list_valid, all_ans_list_r_valid; valid_list = []\n")
a = "model.load_state_dict(checkpoint['state_dict'])"
assert s.count(a) == 1
s = s.replace(a, "model.load_state_dict(checkpoint['state_dict'], strict=False)")
open(p, 'w').write(s)
print('patched')
