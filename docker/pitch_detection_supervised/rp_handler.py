import json
import subprocess
from typing import Any, Dict

import runpod

from pitch_detection_supervised.train import single_run

from run_utils import build_config, instantiate_model, prepare_loaders


def handler(event: Dict[str, Any]) -> str:
    print('Worker Start')
    payload = event['input']

    if 'sweep_id' in payload:
        sweep_id = payload['sweep_id']
        print(f'Runs with sweep id: {sweep_id}')
        args = ['wandb', 'agent', sweep_id]
        if 'count' in payload:
            args.extend(['--count', str(payload['count'])])
        result = subprocess.run(args, check=False)
        return f'exit_code {result.returncode}'

    config = build_config(payload)
    print(f'Single run with configuration: {config}')
    model = instantiate_model(payload.get('model'), config)
    train_loader, val_loader = prepare_loaders(payload, config)
    project = payload.get('project', 'pitch-detection-supervised')

    results = single_run(config=config, model=model, train_loader=train_loader, val_loader=val_loader, project=project)
    return json.dumps(results)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
