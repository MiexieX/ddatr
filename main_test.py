import os, json, csv
import torch
from torch import nn
import argparse
import numpy as np
from modules.metrics import compute_scores
from modules.tester import Tester
from models.blip import blip_decoder, CONDITIONS, SCORES
from dataset import create_dataset_test 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from transformers import BertTokenizer 

def _norm_labels(labels, n=14):
    """兼容 int 列表与字符串 '[0, 2, ...]' 两种形态的标注。"""
    if labels is None:
        return []
    if isinstance(labels, list):
        return [int(x) for x in labels]
    if isinstance(labels, str):
        try:
            return [int(x) for x in labels.strip('[]').split(',')]
        except Exception:
            return []
    return []

def _pos_findings(labels):
    return [CONDITIONS[i] for i, v in enumerate(labels) if v in (1, 3)]

def _events(cur, prev):
    ev = []
    for i, (c, p) in enumerate(zip(cur, prev)):
        if p in (0, 2) and c in (1, 3): ev.append(f"新发:{CONDITIONS[i]}")
        elif p in (1, 3) and c in (0, 2): ev.append(f"消退:{CONDITIONS[i]}")
    return ev

def save_reports(args, log, test_res, test_gts, test_cls_preds):
    """把逐样本生成的诊断报告 + 评估得分落盘到 results/test/<数据集>/（按 index/id 一一对应）：
    - 真实报告.csv         ：放射科医生 GT（含先验报告/标签/进展事件）
    - 模型生成报告.csv      ：模型生成（含诊断 prompt token）
    - 评估得分.csv         ：本测试的指标结果表（指标, 数值）
    - 病例对照.md          ：GT vs 模型并排对照（前 report_examples 例）
    输出目录 = --report_dir（默认 results/test/{mimic|iu|rexrank}）。
    顺序与 create_loader(shuffle=False) 的 dataloader 一致 = annotation 顺序。
    """
    with open(args.ann_path) as f:
        raw = json.load(f)
    if args.dataset_name == 'iu_xray':
        ann = raw
        split_tag = 'iu'
    else:
        ann = raw['test']
        split_tag = 'rexrank' if 'longitudinal' in os.path.basename(args.ann_path) else 'mimic'

    # study_id -> time 查表（算检查间隔）
    tmap = {}
    if args.dataset_name == 'mimic_cxr':
        for sp in ('train', 'val', 'test'):
            for a in (raw[sp] if isinstance(raw, dict) else raw):
                if isinstance(a, dict) and 'study_id' in a:
                    tmap[a['study_id']] = a.get('time')
    def prior_time(rec):
        try:
            p = rec.get('context_image') or ''
            if isinstance(p, list): p = p[0]
            return tmap.get(int(str(p).split('/s')[1].split('/')[0]))
        except Exception:
            return None

    records = []
    for i in range(len(test_res)):
        rec = ann[i] if i < len(ann) else {}
        hp = bool(rec.get('has_progress', False))
        cur = _norm_labels(rec.get('labels')) if 'labels' in rec else [] if args.dataset_name == 'iu_xray' else []
        prev = _norm_labels(rec.get('context_label')) if hp else []
        pt = prior_time(rec) if hp else None
        gap = abs(rec['time'] - pt) if (pt is not None and rec.get('time') is not None) else None
        records.append({
            'index': i,
            'id': rec.get('id'),
            'study_id': rec.get('study_id'),
            'has_progress': hp,
            'view': rec.get('ViewPosition') or rec.get('view'),
            'gap_units': gap,
            'prior_report': rec.get('context_report') if hp else None,
            'gt_report': rec.get('report'),
            'pred_report': test_res[i],
            'gt_labels': cur,
            'prior_labels': prev,
            'gt_pos_findings': _pos_findings(cur),
            'prior_pos_findings': _pos_findings(prev) if (hp and prev) else [],
            'events': _events(cur, prev) if (hp and prev) else [],
            'pred_prompt_tokens': [SCORES[c] for c in test_cls_preds[i]] if i < len(test_cls_preds) else [],
        })

    out_dir = args.report_dir or os.path.join('results', 'test', split_tag)
    os.makedirs(out_dir, exist_ok=True)

    def _compact(x):
        """数组/列表/None 转 CSV 单元格可读字符串。"""
        if x is None:
            return ''
        if isinstance(x, (list, tuple)):
            return '; '.join(str(v) for v in x)
        return str(x)

    # 1) 真实报告（放射科医生 GT）：逐样本全量 CSV
    gj = os.path.join(out_dir, '真实报告.csv')
    gt_header = ['index', 'id', 'study_id', 'has_progress', 'view', 'gap_units',
                 'prior_report', 'gt_report',
                 'gt_labels', 'prior_labels',
                 'gt_pos_findings', 'prior_pos_findings', 'events']
    with open(gj, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(gt_header)
        for i, r in enumerate(records):
            w.writerow([i, _compact(r['id']), _compact(r['study_id']), r['has_progress'],
                        _compact(r['view']), r['gap_units'],
                        _compact(r['prior_report']), _compact(r['gt_report']),
                        _compact(r['gt_labels']), _compact(r['prior_labels']),
                        _compact(r['gt_pos_findings']), _compact(r['prior_pos_findings']),
                        _compact(r['events'])])
    print(f'[save_reports] 真实报告（GT） {len(records)} 份 -> {gj}')

    # 2) 模型生成的报告：逐样本全量 CSV
    pj = os.path.join(out_dir, '模型生成报告.csv')
    pred_header = ['index', 'id', 'study_id', 'view', 'has_progress', 'gap_units',
                   'pred_report', 'pred_prompt_tokens']
    with open(pj, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(pred_header)
        for i, r in enumerate(records):
            w.writerow([i, _compact(r['id']), _compact(r['study_id']), _compact(r['view']),
                        r['has_progress'], r['gap_units'],
                        _compact(r['pred_report']), ' '.join(r['pred_prompt_tokens'])])
    print(f'[save_reports] 模型生成报告 {len(records)} 份 -> {pj}')
    print(f'[save_reports] 对照关系：两份 CSV 按 index/id 一一对应')

    # 3) 评估得分结果表：CSV（指标, 数值）
    sj = os.path.join(out_dir, '评估得分.csv')
    with open(sj, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['指标', '数值'])
        for k, v in log.items():
            w.writerow([k, round(float(v), 6) if isinstance(v, (int, float)) else v])
    print(f'[save_reports] 评估得分表 -> {sj}')

    n_show = len(records) if args.report_examples < 0 else min(args.report_examples, len(records))
    md = os.path.join(out_dir, '病例对照.md')
    with open(md, 'w') as f:
        f.write(f'# 模型生成诊断报告 vs 放射科医生报告（{split_tag}，共 {len(records)} 例，展示 {n_show} 例）\n\n')
        f.write('> 模型：DDaTR 复现 ｜ 解码：beam=' + str(args.beam_size) + ' ｜ 全文与逐样本数据见同目录 JSON\n\n')
        for k, r in enumerate(records[:n_show]):
            f.write(f'\n---\n\n## 病例 {k+1} ｜ {str(r["id"])[:24]} ｜ 视角 {r["view"]} ｜ 有先验: {r["has_progress"]}')
            if r['gap_units'] is not None:
                f.write(f' ｜ 间隔(原始单位): {r["gap_units"]}')
            f.write(f'\n\n**先验发现**: {", ".join(r["prior_pos_findings"]) or "（无）"}')
            if r['events']:
                f.write(f' ｜ **GT 进展事件**: {", ".join(r["events"])}')
            f.write(f'\n\n**GT 阳性发现**: {", ".join(r["gt_pos_findings"]) or "（无）"}')
            f.write(f'\n\n**模型诊断 prompt**: {" ".join(r["pred_prompt_tokens"])}')
            if r['prior_report']:
                f.write(f'\n\n**先验报告**:\n\n> {r["prior_report"][:args.report_note_chars]}')
            f.write(f'\n\n**GT 报告（放射科医生）**:\n\n> {str(r["gt_report"])[:args.report_note_chars]}')
            f.write(f'\n\n**模型报告（生成）**:\n\n> {str(r["pred_report"])[:args.report_note_chars]}\n')
    print(f'[save_reports] 病例对照 -> {md}') 

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--image_size', type=int, default=224, help='input image size')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings 
    parser.add_argument('--load_pretrained', type=str, default=None, help='pretrained path if any')

    # Sample related
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--gen_max_len', type=int, default=150, help='the maximum token length for text generation.')
    parser.add_argument('--gen_min_len', type=int, default=100, help='the minimum token length for text generation.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--monitor_metric', type=str, default='ce_f1', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--init_lr', type=float, default=5e-5, help='.')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='.')
    parser.add_argument('--warmup_lr', type=float, default=5e-7, help='.')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='the weight decay.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed     training')
    parser.add_argument('--device', default='cuda')

    # cls head
    parser.add_argument('--cls_weight', type=float, default=4, help='Loss weight of classification branch.')

    # report saving
    parser.add_argument('--save_reports', action='store_true', help='save per-sample generated diagnostic reports (真实报告/生成报告/评估得分)')
    parser.add_argument('--report_dir', type=str, default=None, help='reports output dir (default: results/test/{mimic|iu|rexrank})')
    parser.add_argument('--report_examples', type=int, default=24, help='number of cases in the markdown review file (-1 = all)')
    parser.add_argument('--report_note_chars', type=int, default=1200, help='truncate report text in markdown')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    utils.init_distributed_mode(args) # from blip
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank() # from blip
    torch.manual_seed(args.seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # create tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_tokens(['[BLA]', '[POS]', '[NEG]', '[UNC]'])

    #### Dataset #### 
    print("Creating dataset...")
    test_dataset = create_dataset_test('generation_%s'%args.dataset_name, tokenizer, args)
    print('number of testing samples: %d'%len(test_dataset))
    
    samplers = [None]

    test_dataloader = create_loader([test_dataset], samplers, batch_size=[args.batch_size], num_workers=[4], is_trains=[False], collate_fns=[None])[0] 

    # build model architecture
    labels_temp = ['[BLA]'] * 14 # for calculate length only
    prompt_temp = ' '.join(labels_temp)+' '
    model = blip_decoder(args, tokenizer, image_size=args.image_size, prompt=prompt_temp)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        model.load_state_dict(state_dict)
        print("load checkpoint from {}".format(args.load_pretrained))

    # get function handles of loss and metrics
    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)   
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    

    # build trainer and start to train
    tester = Tester(model, criterion_cls, metrics, args, device, test_dataloader)

    log, test_res, test_gts, test_cls_preds = tester.test_blip()

    for key, value in log.items():
        print('\t{:15s}: {}'.format(str(key), value))

    if args.save_reports:
        save_reports(args, log, test_res, test_gts, test_cls_preds)
    
if __name__ == '__main__':
    main()
