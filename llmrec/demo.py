"""CPU-only, synthetic demonstration: python -m llmrec.demo."""
import json
from pathlib import Path
from .data import clean_caption, supervision_mode, select_trajectories, build_dpo_pairs
from .objectives import build_sequence, causal_cross_entropy, dpo_loss, reinforce_loss
from .evaluation import evaluate

def main():
    data = json.loads(Path('examples/synthetic.json').read_text(encoding='utf-8'))
    target = data['catalog'][0]
    masked = build_sequence([0, 1], [2, 3], [4, 5], mask_target=True)
    result = {
        'status': 'synthetic_demo_only',
        'cleaned_caption': clean_caption('限时抢购 便携露营灯 充电款 包邮到家'),
        'modes': {d: supervision_mode(d) for d in ['prod', 'video', 'ad', 'living']},
        'selected': select_trajectories(data['trajectories'], target),
        'original_sft': build_sequence([0, 1], [2, 3], [4, 5]),
        'masked_rft': masked,
        'masked_ce': causal_cross_entropy([[0.0]*8 for _ in masked['input_ids']], masked['labels']),
        'dpo_pairs': build_dpo_pairs(target, data['catalog']),
        'dpo_loss': dpo_loss(-1, -2, -1.5, -1.5),
        'reinforce_loss': reinforce_loss([-0.7, -1.2], [1, 0], baseline=0.5),
        'evaluation': evaluate(data['evaluation'], data['catalog']),
    }
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/demo.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()
