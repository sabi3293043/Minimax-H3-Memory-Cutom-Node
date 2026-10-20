"""Optional CPU SAM 2 worker. Own process, own threads, temporary JPEGs cleaned."""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import time


def run(root, cpu_threads=0):
    import site
    site.addsitedir(str(Path(root) / 'packages'))
    sys.path.insert(0, str(Path(root) / 'packages'))
    import torch
    from .cpu_threads import resolve_cpu_threads
    torch.set_num_threads(resolve_cpu_threads(cpu_threads))
    from PIL import Image
    import numpy as np
    from sam2.build_sam import build_sam2_video_predictor
    with redirect_stdout(sys.stderr):
        predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_t.yaml',
            str(Path(root)/'sam2.1_hiera_tiny.pt'), device='cpu', apply_postprocessing=False)
    print(json.dumps({'ready': True,'cpu_threads':torch.get_num_threads()}), flush=True)
    for line in sys.stdin:
        try:
            started = time.perf_counter()
            detection_seconds = 0.
            propagation_seconds = 0.
            request = json.loads(line)
            if not 1 <= len(request['frames']) <= 16 or not 0 <= len(request['regions']) <= 4:
                raise ValueError('Tracker accepts 1..16 frames and 0..4 optional regions')
            records = []
            regions = list(request['regions'])
            with tempfile.TemporaryDirectory(prefix='minimax-sam2-', dir=request['temp_root']) as directory, torch.inference_mode(), redirect_stdout(sys.stderr):
                for index, encoded in enumerate(request['frames']):
                    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert('RGB')
                    image.thumbnail((384, 384))
                    image.save(Path(directory)/f'{index:05d}.jpg')
                if not regions:
                    detection_started = time.perf_counter()
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                    from .tracking_selection import automatic_regions
                    source = int(request['source'])
                    first = np.array(Image.open(Path(directory)/f'{source:05d}.jpg').convert('RGB'))
                    generator = SAM2AutomaticMaskGenerator(predictor,points_per_side=8,
                        points_per_batch=16,pred_iou_thresh=.8,stability_score_thresh=.9,
                        crop_n_layers=0,min_mask_region_area=0)
                    regions = automatic_regions(generator.generate(first),first.shape[1],first.shape[0])
                    for region in regions:
                        region['source'] = source
                    generator.predictor.reset_predictor()
                    del generator,first
                    detection_seconds = time.perf_counter()-detection_started
                if regions:
                    propagation_started = time.perf_counter()
                    state = predictor.init_state(directory, offload_video_to_cpu=True, offload_state_to_cpu=True)
                    height, width = state['video_height'], state['video_width']
                    for object_id, region in enumerate(regions, 1):
                        source = int(region['source'])
                        box = np.array(region['box'], dtype=np.float32) * np.array([width,height,width,height])
                        if 'segmentation' in region:
                            predictor.add_new_mask(state,frame_idx=source,obj_id=object_id,mask=region['segmentation'])
                        else:
                            predictor.add_new_points_or_box(state, frame_idx=source, obj_id=object_id, box=box)
                    initial_areas = {}
                    for frame, objects, logits in predictor.propagate_in_video(state):
                        for row, object_id in enumerate(objects):
                            region = regions[int(object_id)-1]
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
                    propagation_seconds = time.perf_counter()-propagation_started
            summary = [dict(name=r['name'],box=r['box']) for r in regions]
            reason = (f'{len(regions)} tracked regions supplement whole-frame memory' if regions else
                      'No confident objects detected; whole-frame memory remains active')
            metrics = dict(cpu_threads=torch.get_num_threads(), detection_seconds=detection_seconds,
                propagation_seconds=propagation_seconds,worker_seconds=time.perf_counter()-started)
            print(json.dumps({'records':records,'regions':summary,'reason':reason,'metrics':metrics}), flush=True)
        except Exception as exc:
            print(json.dumps({'error':f'{type(exc).__name__}: {exc}'}), flush=True)


if __name__ == '__main__':
    run(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 0)
