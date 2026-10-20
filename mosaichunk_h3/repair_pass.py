"""Experimental post-generation appearance repair using native H3 latent masks.

Candidate selection and automatic budgets are heuristics, not semantic error
detection. A candidate is kept only when its bounded RGB reference metric
improves without losing structural agreement. No VAE/model is downloaded.
"""
from dataclasses import dataclass, replace
import copy
import logging
import math
import time
from contextlib import nullcontext

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecondPassSettings:
    enabled: bool = False
    automatic: bool = False
    steps: int = 6
    max_steps: int = 12
    strength: float = .25
    threshold: float = .10
    max_area: float = .20

    def validate(self):
        for name, value in (('second_pass_steps',self.steps),('second_pass_max_steps',self.max_steps)):
            if isinstance(value,bool) or not isinstance(value,int) or not 2 <= value <= 32:
                raise ValueError(f'{name} must be an integer from 2 to 32')
        for name,value,low,high in (('strength',self.strength,.05,.60),
                                    ('threshold',self.threshold,.02,.30),('max_area',self.max_area,.01,.50)):
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'second_pass_{name} must be finite and within {low}..{high}')
        return self

    def budget(self, *, area, severity):
        if area <= 0:
            return 0
        if not self.automatic:
            return self.steps
        # Coverage and measured drift have a meaning; attention-call counts do
        # not indicate the number of independent mistakes or steps to fix them.
        estimate = math.ceil(3 + 4*math.sqrt(min(1.,area)) + 4*min(1.,max(0.,severity)))
        return min(self.max_steps,max(2,estimate))


def _rgb_features(frame, box):
    import torch
    import torch.nn.functional as F
    from .region_memory import crop
    value = crop(frame,box).detach().cpu().float()
    if frame.dtype == torch.uint8:
        value = value/255.
    color = F.adaptive_avg_pool2d(value[None],(6,6))[0]
    gray = F.adaptive_avg_pool2d(value.mean(0)[None,None],(16,16))[0,0]
    edges = torch.cat(((gray[1:]-gray[:-1]).flatten(),(gray[:,1:]-gray[:,:-1]).flatten()))
    norm = float(edges.norm())
    return color, edges/max(norm,1e-8), norm


def _structure(reference, target):
    if reference[2] < .02 and target[2] < .02:
        return 1.
    if min(reference[2],target[2]) < .02:
        return 0.
    return float((reference[1]*target[1]).sum().clamp(-1,1))


@dataclass
class RepairPlan:
    mask: object
    plans: tuple
    references: tuple
    area: float
    severity: float
    reason: str

    def assess(self, frames):
        if not self.references:
            return 0., 1.
        errors, structures = [], []
        for target, box, reference, cells in self.references:
            current = _rgb_features(frames[target],box)
            errors.append(float((reference[0]-current[0]).abs().mean(0)[cells].mean()))
            structures.append(_structure(reference,current))
        return sum(errors)/len(errors), min(structures)


def build_repair_plan(frames, latent_shape, settings, *, sources=(0,), targets=(), regions=(), tracked=None,
                      protected=()):
    """Find bounded color/texture drift in structurally aligned current frames.

    Stored RGB is NOT a trained identity signal. Scene/camera changes are
    deliberately rejected instead of being labelled incorrect appearances.
    """
    import torch
    import torch.nn.functional as F
    from .native_guidance import decoded_to_latent
    from .region_memory import RegionPlan, move_region, grid_ids
    settings.validate()
    _,channels,latent_t,height,width = latent_shape
    mask = torch.zeros((1,1,latent_t,height,width))
    sources = tuple(sorted(set(sources)))
    if not sources or any(not 0 <= i < len(frames) for i in sources):
        raise ValueError('Repair sources must refer to completed observation frames')
    groups = tuple(decoded_to_latent(i,len(frames),latent_t) for i in range(len(frames)))
    if any(not 0 <= i < len(frames) for i in protected):
        raise ValueError('A protected repair frame is outside the completed observation')
    anchors = {groups[i] for i in (*sources,*protected)}
    target_ids = tuple(targets) or tuple(range(len(frames)))
    boxes = tuple(regions) or (None,)
    candidates = []
    for source in sources:
        for region in boxes:
            source_box = region.box if region is not None else (0,0,1,1)
            anchor = None
            if tracked is not None:
                anchor = tracked.get((source,region.name)) if region is not None else None
                initial = tracked.get((min(sources),region.name)) if region is not None else None
                if not anchor or not initial or not anchor.get('trusted',False) or not initial.get('trusted',False):
                    continue
                source_box = move_region(source_box,initial['box'],anchor['box'])
                if source_box is None:
                    continue
            reference = _rgb_features(frames[source],source_box)
            # A flat unlabelled scene has no meaningful object correspondence.
            if region is None and float(reference[0].std()) < .03:
                continue
            for target in target_ids:
                if not 0 <= target < len(frames) or target <= source or groups[target] in anchors:
                    continue
                box = source_box
                confidence = 1.
                target_mask = source_mask = None
                if tracked is not None:
                    record = tracked.get((target,region.name))
                    if not record or not record.get('trusted',False):
                        continue
                    box = move_region(source_box,anchor['box'],record['box'])
                    if box is None:
                        continue
                    source_mask,target_mask = anchor['mask'],record['mask']
                    confidence = min(anchor['confidence'],record['confidence'])
                current = _rgb_features(frames[target],box)
                structure = _structure(reference,current)
                if structure < .92:
                    continue
                drift = (reference[0]-current[0]).abs().mean(0)
                cells = drift >= settings.threshold
                if not bool(cells.any()):
                    continue
                # A whole-view color/exposure shift is not a localized defect.
                if region is None and float(cells.float().mean()) > .35:
                    continue
                local = torch.zeros((height,width))
                l,t,r,b = box
                x0,y0 = int(l*width),int(t*height)
                x1,y1 = max(x0+1,math.ceil(r*width)),max(y0+1,math.ceil(b*height))
                local[y0:y1,x0:x1] = F.interpolate(cells.float()[None,None],size=(y1-y0,x1-x0),mode='nearest')[0,0]
                local = F.max_pool2d(local[None,None],2,2)[0,0].repeat_interleave(2,0).repeat_interleave(2,1)
                if target_mask is not None:
                    valid = grid_ids(box,height//2,width//2,mask=target_mask)
                    allowed = torch.zeros((height//2)*(width//2));allowed[valid] = 1
                    local *= allowed.reshape(height//2,width//2).repeat_interleave(2,0).repeat_interleave(2,1)
                if not bool(local.any()):
                    continue
                severity = float(drift[cells].mean())
                candidates.append((severity,structure*confidence,source,target,source_box,box,local,
                                   reference,cells,source_mask,region.name if region is not None else 'whole view'))
    candidates.sort(key=lambda item:(-item[0]*item[1],item[3],item[2]))
    plans, references, seen = [], [], set()
    for severity,confidence,source,target,source_box,box,local,reference,cells,source_mask,name in candidates:
        group = groups[target]
        key = (group,name)
        if key in seen:
            continue
        proposed = mask.clone()
        proposed[0,0,group] = torch.maximum(proposed[0,0,group],local)
        if float(proposed.mean()) > settings.max_area:
            return RepairPlan(torch.zeros_like(mask),(),(),0.,0.,
                'combined repair coverage exceeds second_pass_max_area; whole proposal withheld')
        mask = proposed
        seen.add(key)
        plans.append(RegionPlan(groups[source],group,f'repair candidate: {name}',source_box,box,confidence,
            source,target,source_mask,local,regional=True))
        references.append((target,box,reference,cells))
        if len(plans) > 128:
            return RepairPlan(torch.zeros_like(mask),(),(),0.,0.,
                'repair exceeds 128 region/frame candidates; whole proposal withheld')
    area = float(mask.mean())
    severity = sum(float((r[2][0]-_rgb_features(frames[r[0]],r[1])[0]).abs().mean(0)[r[3]].mean())
                   for r in references)/max(1,len(references))
    return RepairPlan(mask,tuple(plans),tuple(references),area,severity,
        'bounded aligned-appearance candidates' if plans else 'no confident localized repair candidates')


def repair_sigmas(sigmas, steps, strength):
    """Resample the user's schedule over its low-noise tail, ending at zero."""
    import torch
    values = sigmas.detach().float().cpu()
    if (values.ndim != 1 or len(values) < 2 or not torch.isfinite(values).all()
            or float(values[-1]) != 0. or bool((values[1:] > values[:-1]).any())
            or float(values[0]) <= 0.):
        raise ValueError('Second pass requires a finite descending sigma schedule ending at zero')
    cap = min(float(values[0])*strength,float(values[0])*.60)
    tail = values[(values < cap) & (values > 0)]
    knots = torch.cat((torch.tensor([cap]),tail,torch.zeros(1)))
    positions = torch.linspace(0,len(knots)-1,steps+1)
    left = positions.floor().long();right = positions.ceil().long()
    result = knots[left]+(knots[right]-knots[left])*(positions-left)
    result[-1] = 0.
    return result


def execute_second_pass(observer, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                        callback, disable_pbar, seed, *, latent_shapes, shared):
    import torch
    from .native_visual_observer import _load_tae_decoder, _unpack_video_x0, preserve_observation_rng, KEY
    from .native_guidance import GuidanceSession, frame_numbers, decoded_to_latent, install_guidance
    from .region_memory import parse_regions
    from .native_status import notify_native
    from .memory_controls import compatibility
    from comfy.nested_tensor import NestedTensor
    from comfy.patcher_extension import WrappersMP
    from contextlib import ExitStack
    import comfy.utils

    settings = observer.second_pass
    guider = getattr(executor,'class_obj',None)
    if guider is None or not hasattr(guider,'original_conds'):
        raise ValueError('Second pass requires a native H3 guider and sampler')
    compatibility(guider,guided=True,accelerated=False,denoise_mask=denoise_mask)
    if observer.guidance is None or not observer.guidance.enabled:
        raise ValueError('Second pass requires retrieved_frames or manual_frames guidance with nonzero strength')
    if not latent_shapes or len(latent_shapes) != 2 or latent_shapes[0][1] != 24:
        raise ValueError('Second pass requires native H3 video/audio latents')
    if len(sigmas) < 2 or float(sigmas[-1]) != 0.:
        raise ValueError('Finish the first pass at sigma zero before using second-pass repair')
    repair_sigmas(sigmas,settings.steps,settings.strength)
    root = guider.model_patcher.get_model_object('diffusion_model')
    if not getattr(root,'blocks',None) or any(i>=len(root.blocks) for i in frame_numbers(observer.guidance.layers)):
        raise ValueError('Second pass requires valid guidance layers on native H3')
    GuidanceSession(observer.guidance,latent_shapes[0]).close()
    if not callable(getattr(guider.model_patcher.model,'_denoise_mask_conds',None)):
        raise ValueError('Second pass requires native H3 video/audio denoise-mask support in ComfyUI')
    baseline_observer = copy.copy(observer)
    baseline_observer.guidance = None
    baseline_observer.defer_observation = True
    baseline_observer.memory_options = replace(observer.memory_options,export_report=False)
    started = time.perf_counter()
    baseline = baseline_observer._call(executor,noise,latent_image,sampler,sigmas,denoise_mask,
        callback,disable_pbar,seed,latent_shapes=latent_shapes,shared=shared)
    metrics = dict(mode='auto estimate' if settings.automatic else 'manual',accepted=False,steps=0,
                   first_pass_seconds=time.perf_counter()-started)
    from .memory_inspector import MemoryInspector
    inspector = MemoryInspector(enabled=observer.memory_options.inspect)
    inspector_entry = None
    from dataclasses import asdict
    configuration = dict(second_pass=asdict(settings),check_skipping=asdict(observer.check_skipping),
        memory=asdict(observer.memory_options),guidance=asdict(observer.guidance),
        acceleration=asdict(observer.acceleration) if observer.acceleration is not None else None,
        observer=dict(tiny_vae_name=observer.tiny_vae_name,max_frames=observer.max_frames,
            max_memory_frames=observer.max_memory_frames,max_observation_megapixels=observer.max_observation_megapixels),
        sampling=dict(latent_shapes=latent_shapes,sigmas=sigmas.detach().cpu().tolist(),
            sampler=getattr(getattr(sampler,'sampler_function',None),'__name__',type(sampler).__name__)))
    callback_failed = False
    def forward_callback(*args):
        nonlocal callback_failed
        if callback is not None:
            try:
                callback(*args)
            except BaseException:
                callback_failed = True
                raise
    def finalize(final, total):
        # The outer sampler's x0 output must agree even after rejection/failure.
        total = max(1,total)
        if shared is not None and observer.node_id is not None and observer.memory_options.preview=='auto':
            shared.suppress_standard_preview()
            with shared.completed_step():
                forward_callback(total-1,guider.model_patcher.model.process_latent_in(final),final,total)
        else:
            forward_callback(total-1,guider.model_patcher.model.process_latent_in(final),final,total)
        return final
    def status(message, *, finished=False, **extra):
        notify_native(observer.node_id,message,log=True,state='complete' if finished else 'second pass',
            second_pass={**metrics,**extra},shared_decode=shared.metrics() if shared is not None else None,
            guidance_details=metrics.get('guidance_details'),
            inspector=inspector_entry,run_id=inspector.run_id,settings=configuration,seed=seed)
        if finished and observer.memory_options.export_report:
            try:
                from pathlib import Path
                import folder_paths
                from .memory_inspector import write_report
                path = write_report(Path(folder_paths.get_output_directory())/'minimax_memory_reports',
                    settings=configuration,
                    seed=seed,inspector=inspector,error=extra.get('error'),
                    metrics=dict(second_pass={**metrics,**extra},total_seconds=time.perf_counter()-started))
                notify_native(observer.node_id,f'Second-pass report saved: {path}',report_path=path)
            except Exception:
                LOG.warning('Unable to save optional second-pass report',exc_info=True)
    from .native_preview import RepairComparison
    options = observer.memory_options
    comparison = RepairComparison(observer.node_id,enabled=options.preview!='off' and options.compare_repair,
        frames=options.preview_frames,max_resolution=options.preview_resolution,fps=options.preview_fps)
    capture = observer.comparison_capture
    def finish_preview(after=None,*,outcome,caption,result):
        if capture is not None and capture.enabled:
            try:
                capture.finish(after,outcome=outcome,result=result,latent_shapes=latent_shapes,caption=caption)
            except Exception as exc:
                capture.reset(f'Comparison output capture failed: {type(exc).__name__}: {str(exc)[:180]}')
                LOG.warning('Comparison output unavailable; sampled result preserved',exc_info=True)
        if comparison.enabled:
            comparison.finish(after,outcome=outcome,caption=caption)
            return
        # Comparison is opt-in. Keep the existing single retained-result view
        # without encoding/holding a separate first-pass preview when it is off.
        final_frames = after if outcome=='accepted' else frames
        if options.preview=='off' or observer.node_id is None or final_frames is None:
            return
        try:
            from .native_preview import send,select_indices,MemoryPreview
            indices = select_indices(len(final_frames),options.preview_frames)
            selected = torch.stack([final_frames[i] for i in indices])
            send(observer.node_id,mode='auto',frames=MemoryPreview.encode(selected,options.preview_resolution),
                caption=caption,indices=indices,step=metrics['steps'],total=metrics['steps'],
                fps=options.preview_fps,width=final_frames.shape[-1],height=final_frames.shape[-2])
        except Exception:
            LOG.warning('Final repair preview unavailable; sampled result preserved',exc_info=True)
    frames = None
    try:
        status('Second pass: inspecting completed video; candidates are heuristic')
        with preserve_observation_rng(), shared.completed_step() if shared is not None else nullcontext():
            decoder = _load_tae_decoder(observer.tiny_vae_name)
            video = _unpack_video_x0(baseline,latent_shapes)
            if video is None:
                raise ValueError('Cannot unpack completed H3 video latents')
            decoded_count = (sum(1 if i%5==0 else 4 for i in range(video.shape[2]))
                             if getattr(decoder,'decodes_prefix',False) else video.shape[2])
            pixels = decoded_count*video.shape[3]*video.shape[4]*getattr(decoder,'upscale_ratio',16)**2
            if pixels > observer.max_observation_megapixels*1e6:
                raise ValueError('Second-pass inspection exceeds max_observation_megapixels')
            frames = decoder.decode_video(video[:1],frame_indices=None).cpu()
        if capture is not None and capture.enabled:
            try:
                capture.start(frames)
            except Exception as exc:
                capture.reset(f'Comparison first-pass capture failed: {type(exc).__name__}: {str(exc)[:180]}')
                LOG.warning('Comparison output unavailable; sampling preserved',exc_info=True)
        comparison.start(frames)
        protected = frame_numbers(observer.memory_options.protected_frames)
        final_count = sum(1 if i%5==0 else 4 for i in range(latent_shapes[0][2]))
        primary_source = observer.guidance.source_frame if observer.guidance.mode=='manual_frames' else 0
        sources = tuple(dict.fromkeys((primary_source,*protected)))[:8]
        def mapped(ids):
            return tuple(sorted(set(decoded_to_latent(i,final_count,latent_shapes[0][2])
                if len(frames)==latent_shapes[0][2] else i for i in ids)))
        sources = mapped(sources)
        protected = set(mapped(protected))
        # First/last and explicit native keyframe guides must not become repair targets.
        for conds in guider.original_conds.values():
            for cond in conds:
                for guide in cond.get('minimax_keyframes',()):
                    start = guide.get('resolved_frame_index')
                    latent = guide.get('latent')
                    if start is None or latent is None:
                        continue
                    length = sum(1 if i%5==0 else 4 for i in range(latent.shape[2]))
                    protected.update(mapped(range(start,min(final_count,start+length))))
        targets = mapped(frame_numbers(observer.guidance.target_frames)) if observer.guidance.mode=='manual_frames' else ()
        regions = parse_regions(observer.memory_options.regions)
        tracked = None
        if observer.memory_options.occlusion:
            from .optional_occlusion import ensure_installed
            from .occlusion_tracking import OcclusionTracker
            with OcclusionTracker(ensure_installed(),cpu_threads=observer.memory_options.cpu_threads) as tracker:
                tracked = tracker.track(frames,regions,sources,selection='uniform',frame_budget=16,step=-1)
                regions = tracker.regions
                metrics['tracking'] = tracker.last_metrics
            if not regions:
                finish_preview(outcome='skipped',result=baseline,caption='Original retained · no confident tracked objects · silent Tiny VAE preview')
                status('Second pass skipped: no confident tracked objects; original video retained',finished=True)
                return baseline
        plan = build_repair_plan(frames,latent_shapes[0],settings,sources=sources,targets=targets,
                                 regions=regions,tracked=tracked,protected=tuple(protected))
        if observer.memory_options.inspect:
            try:
                starts=[];cursor=0
                for i in range(latent_shapes[0][2]):
                    starts.append(cursor);cursor += 1 if i%5==0 else 4
                groups=tuple(decoded_to_latent(i,len(frames),latent_shapes[0][2]) for i in range(len(frames)))
                inspector_entry = inspector.observation(frames,plan.plans,step=len(sigmas)-1,
                    video_frames=tuple(starts[g] if len(frames)==len(starts) else i for i,g in enumerate(groups)),
                    frame_groups=groups,events=[dict(region='second pass',
                        reason='Post-generation repair proposals; not confirmed semantic errors')])
            except Exception:
                LOG.debug('Repair inspector unavailable; sampling preserved',exc_info=True)
        before, structure_before = plan.assess(frames)
        steps = settings.budget(area=plan.area,severity=plan.severity)
        metrics.update(steps=steps,area=plan.area,severity=plan.severity,score_before=before,
                       candidates=len(plan.plans),reason=getattr(plan,'reason','bounded repair proposal'))
        if not steps:
            finish_preview(outcome='skipped',result=baseline,caption='Original retained · no confident repair candidates · silent Tiny VAE preview')
            status(f'Second pass skipped: {metrics["reason"]}; original video retained',finished=True)
            return baseline
        status(f'Second pass: {steps} masked repair steps ({metrics["mode"]}); model still processes full context')
        patcher = guider.model_patcher.clone()
        patcher.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE,KEY)
        # Clone the MODEL's clean options, not the already-prepared parent
        # guider: native preparation would otherwise merge its wrappers twice.
        wrappers = patcher.model_options.get('transformer_options',{}).get('wrappers',{})
        for group in wrappers.values():
            group.pop(KEY,None)
        repair_guider = copy.copy(guider)
        repair_guider.model_patcher = patcher
        repair_guider.model_options = patcher.model_options
        repair_guider.original_conds = {k:[dict(c) for c in v] for k,v in guider.original_conds.items()}
        clean = comfy.utils.unpack_latents(baseline,latent_shapes)
        mask_video = plan.mask.expand_as(clean[0]).contiguous()
        masks = NestedTensor((mask_video,torch.zeros_like(clean[1],device='cpu')))
        generator = torch.Generator(device='cpu').manual_seed((int(seed or 0)+131071)%(2**63-1))
        repair_noise = NestedTensor(torch.randn(s.shape,generator=generator,dtype=torch.float32) for s in clean)
        session = GuidanceSession(replace(observer.guidance,end_progress=1.),latent_shapes[0])
        session.set_region_plans(plan.plans)
        def repair_callback(step,x0,x,total):
            session.progress = (step+1)/max(1,total)
            if callback is not None:
                if getattr(x0,'is_nested',False):
                    x0 = comfy.utils.pack_latents(x0.unbind())[0]
                    x = comfy.utils.pack_latents(x.unbind())[0]
                forward_callback(step,x0,x,total)
        with preserve_observation_rng(), ExitStack() as stack:
            stack.enter_context(install_guidance(repair_guider,session))
            if shared is not None:
                from .shared_tiny_vae import install_shared_callback
                stack.enter_context(install_shared_callback(repair_guider,shared))
            repaired = repair_guider.sample(repair_noise,NestedTensor(clean),sampler,
                repair_sigmas(sigmas,steps,settings.strength),denoise_mask=masks,
                callback=repair_callback,disable_pbar=disable_pbar,seed=seed)
            # Capture diagnostics before install_guidance clears its temporary
            # plans/heatmap. Counts describe the attempted pass, even if rejected.
            if observer.memory_options.inspect:
                metrics['guidance_details'] = session.diagnostics()
        metrics.update(attention_corrections=session.corrections,corrected_rows=session.corrected_rows)
        candidate_streams = repaired.unbind()
        # Preserve every unselected latent and the complete soundtrack exactly.
        candidate_video = torch.where(mask_video.to(candidate_streams[0].device)>0,
                                      candidate_streams[0],clean[0].to(candidate_streams[0].device))
        candidate = comfy.utils.pack_latents((candidate_video,clean[1].to(candidate_video.device)))[0]
        if not torch.isfinite(candidate).all():
            raise ValueError('Nonfinite repair result')
        with preserve_observation_rng(), shared.completed_step() if shared is not None else nullcontext():
            after_frames = decoder.decode_video(candidate_video[:1],frame_indices=None).cpu()
        after, structure_after = plan.assess(after_frames)
        accepted = after < before-max(.002,before*.01) and structure_after >= structure_before-.02
        metrics.update(accepted=accepted,score_after=after,structure_before=structure_before,
                       structure_after=structure_after)
        final = candidate if accepted else baseline
        # SamplerCustomAdvanced also exposes denoised_output: keep its x0 callback
        # consistent with the actually accepted/restored output, including audio scale.
        finalize(final,steps)
        finish_preview(after_frames,result=final,outcome='accepted' if accepted else 'rejected',caption=
            ('Repair accepted · saving after' if accepted else 'Repair rejected · saving before')+' · silent Tiny VAE preview')
        status('Second pass accepted: reference metric improved; visual quality needs inspection' if accepted
               else 'Second pass rejected: reference metric did not improve safely; original video retained',finished=True)
        return final
    except Exception as exc:
        from comfy.model_management import InterruptProcessingException
        if isinstance(exc,InterruptProcessingException) or callback_failed:
            raise
        metrics['accepted'] = False
        finish_preview(outcome='failed',result=baseline,caption='Repair failed · original retained · silent Tiny VAE preview')
        status(f'Second pass failed: {type(exc).__name__}: {str(exc)[:140]}; original video retained',
               finished=True,error=str(exc)[:300])
        return finalize(baseline,metrics['steps'] or len(sigmas)-1)
