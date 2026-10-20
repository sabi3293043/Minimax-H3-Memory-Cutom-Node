"""Optional CPU SAM 2 worker. Own process, own threads, temporary JPEGs cleaned."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile


def run(root):
    import site
    site.addsitedir(str(Path(root) / 'packages'))
    sys.path.insert(0, str(Path(root) / 'packages'))
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    import numpy as np
    from sam2.build_sam import build_sam2_video_predictor
    with redirect_stdout(sys.stderr):
        predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
            str(Path(root)/'sam2.1_hiera_tiny.pt'), device='cpu', apply_postprocessing=False)
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not 1 <= len(request['frames']) <= 16 or not 1 <= len(request['regions']) <= 4:
                raise ValueError('Tracker accepts 1..16 frames and 1..4 regions')
            records = []
            with tempfile.TemporaryDirectory(prefix='minimax-sam2-', dir=request['temp_root']) as directory, torch.inference_mode(), redirect_stdout(sys.stderr):
                for index, encoded in enumerate(request['frames']):
                    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
                    image.thumbnail((384, 384))
                    image.save(Path(directory)/f'{index:05d}.jpg')
                state = predictor.init_state(directory, offload_video_to_cpu=True, offload_state_to_cpu=True)
                height, width = state['video_height'], state['video_width']
                for object_id, region in enumerate(request['regions'], 1):
                    source = int(region['source'])
                    box = np.array(region['box'], dtype=np.float32) * np.array([width,height,width,height])
                    predictor.add_new_points_or_box(state, frame_idx=source, obj_id=object_id, box=box)
                initial_areas = {}
                for frame, objects, logits in predictor.propagate_in_video(state):
                    for row, object_id in enumerate(objects):
                        region = request['regions'][int(object_id)-1]
                        mask = logits[row, 0] > 0
                        area = int(mask.sum())
                        if frame == int(region['source']):
                            initial_areas[object_id] = max(1, area)
                        ratio = area / initial_areas.get(object_id, max(1, area))
                        visible = area >= 16 and ratio >= .1 and frame >= int(region['source'])
                        if visible:
                            ys, xs = mask.nonzero(as_tuple=True)
                            box = [float(xs.min())/width, float(ys.min())/height,
                                   float(xs.max()+1)/width, float(ys.max()+1)/height]
                            confidence = float(logits[row,0][mask].sigmoid().mean())
                        else:
                            box, confidence = region['box'], 0.
                        grid = torch.nn.functional.interpolate(mask[None,None].float(),
                            (32,32),mode='nearest')[0,0].byte().tolist()
                        records.append(dict(frame=request['ids'][frame], region=region['name'],
                            visible=visible, box=box, confidence=confidence, mask=grid))
                predictor.reset_state(state)
                del state
            print(json.dumps({'records':records}), flush=True)
        except Exception as exc:
            print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
