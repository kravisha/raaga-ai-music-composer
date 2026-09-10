"""Explicitly authorized, bounded real-SPB one-utterance VITS fine-tune.

Engineering/memorization pilot, not a held-out quality evaluation. Originals
are immutable; every output lives in a newly created run directory.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
FREEZE_TEXT_DURATION = '--freeze-text-duration' in sys.argv
OUT = ROOT/('pilot-fixed-text-duration-02' if FREEZE_TEXT_DURATION else 'pilot-one-utterance-01')
STEPS, TIMEOUT, SEED = 30, 600, 20260910
os.environ.update(HF_HOME=str(ROOT/'cache/huggingface'), TRANSFORMERS_CACHE=str(ROOT/'cache/huggingface/hub'),
                  HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
                  MPLCONFIGDIR=str(ROOT/'cache/matplotlib'), WANDB_MODE='disabled', WANDB_DISABLED='true')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

if '--worker' not in sys.argv:
    OUT.mkdir(exist_ok=False)
    (OUT/'pilot-script.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'mms_portable_alignment.py').write_bytes(Path(__file__).with_name('mms_portable_alignment.py').read_bytes())
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(), 'timeout_seconds':TIMEOUT,
             'max_optimizer_steps':STEPS, 'scope':'Real reviewed SPB one-utterance engineering pilot',
             'authorization':'Krish explicitly requested a working direct TTS pilot and continued refinement. The first pilot is preserved; variant02 isolates text/duration drift after his pronunciation-regression feedback.',
             'freeze_text_encoder_and_duration_predictor':FREEZE_TEXT_DURATION}
    with (OUT/'worker.log').open('x',encoding='utf-8') as log:
        command=[sys.executable,'-B',__file__,'--worker']+(['--freeze-text-duration'] if FREEZE_TEXT_DURATION else [])
        p=subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['pid']=p.pid
        (OUT/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(receipt),flush=True)
        try: receipt['exit_code']=p.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait(); receipt.update(exit_code=p.returncode, timeout=True)
    receipt['finished_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'launch.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt),flush=True)
    sys.exit(0 if receipt['exit_code']==0 else 1)

status={'stage':'initializing', 'optimizer_steps_completed':0, 'training_seconds':0,
        'evaluation_kind':'exposed_training_phrase_and_unseen_text_probe; no held-out reference',
        'limits':['One18.686-second approved utterance; memorization/overfitting likely.',
                  'A saved checkpoint and finite loss do not establish SPB resemblance, pronunciation or singing.',
                  'No formal train/eval export performed; explicit user-authorized one-utterance pilot.'],
        'metrics':[], 'audio':[]}
def save(**changes):
    status.update(changes)
    (OUT/'result.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(changes,ensure_ascii=True),flush=True)

try:
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(SEED); np.random.seed(SEED)
    pairfile=ROOT/'data/paired-speech-01/pairing.json'
    pair=json.loads(pairfile.read_text(encoding='utf-8'))
    assert pair['accepted_for_training'] is True and pair['split']=='train'
    assert pair['target']=='spb_speech'
    assert all(pair[k]=='reviewed' for k in ['text_review','speaker_review','quality_review'])
    for pathkey,hashkey in [('path','sha256'),('source_path','source_sha256'),('text_file','text_sha256'),
                          ('parent_clip_path','parent_clip_sha256'),
                          ('approved_full_verbatim_file','approved_full_verbatim_sha256')]:
        assert sha(pair[pathkey])==pair[hashkey], f'{pathkey} hash changed'
    (OUT/'training-pair.json').write_bytes(pairfile.read_bytes())
    source_hashes={pair[k]:pair[h] for k,h in [('path','sha256'),('source_path','source_sha256'),
                      ('parent_clip_path','parent_clip_sha256'),('approved_full_verbatim_file','approved_full_verbatim_sha256')]}
    audio, rate=sf.read(pair['path'],dtype='float32')
    assert rate==16000 and audio.ndim==1 and np.isfinite(audio).all()
    assert 1 <= len(audio)/rate <= 20 and np.max(np.abs(audio))<1
    text=pair['text']
    assert text==Path(pair['text_file']).read_text(encoding='utf-8').strip()
    sys.path.insert(0,str(ROOT/'training-source'))
    from utils import VitsModelForPreTraining, VitsFeatureExtractor, slice_segments
    from transformers import VitsTokenizer
    from mms_portable_alignment import maximum_path
    facade=types.ModuleType('monotonic_align'); facade.maximum_path=maximum_path
    sys.modules['monotonic_align']=facade
    from run_vits_finetuning import discriminator_loss, feature_loss, generator_loss, kl_loss
    folder=ROOT/'assets/mms-tts-tam-train-481128b97cd2'
    weight=folder/'model.safetensors'
    assert sha(weight)=='bf08d6c5444f01c7c764fddf2e8322efc1c9028fef51eab214a9b3d03ba431dc'
    source_hashes[str(weight)]=sha(weight)
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(.85,0)
    device='cuda'
    tokenizer=VitsTokenizer.from_pretrained(folder,local_files_only=True)
    fx=VitsFeatureExtractor.from_pretrained(folder,local_files_only=True)
    tokens=tokenizer(text,return_tensors='pt').to(device)
    assert 0 < tokens['input_ids'].shape[-1] < 768
    assert int(tokens['input_ids'].min())>=0 and int(tokens['input_ids'].max())<58
    model=VitsModelForPreTraining.from_pretrained(folder,local_files_only=True).to(device)
    generator_initial={name:sha_tensor for name,value in model.state_dict().items()
                       if not name.startswith('discriminator.')
                       for sha_tensor in [hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()]}
    frozen_names=[name for name in generator_initial if name.startswith(('text_encoder.','duration_predictor.'))]
    if FREEZE_TEXT_DURATION:
        for module in [model.text_encoder,model.duration_predictor]:
            for parameter in module.parameters(): parameter.requires_grad_(False)
    model.decoder.apply_weight_norm()
    for flow in model.flow.flows:
        torch.nn.utils.weight_norm(flow.conv_pre); torch.nn.utils.weight_norm(flow.conv_post)
    discriminator=model.discriminator; del model.discriminator
    for disc in discriminator.discriminators: disc.apply_weight_norm()
    probes=[('training-phrase',text),('new-sentence','இன்று ஒரு புதிய பாடலைப் பாடுகிறேன்.')]
    (OUT/'probe-texts.json').write_text(json.dumps(dict(probes),ensure_ascii=False,indent=2),encoding='utf-8')
    save(stage='reviewed_data_loaded', real_spb_audio_used=True, training_seconds=len(audio)/rate,
         training_token_count=int(tokens['input_ids'].shape[-1]), precision='float32', batch_size=1,
         seed=SEED, learning_rate=2e-5, gpu=torch.cuda.get_device_name(0), maximum_steps=STEPS,
         frozen_text_duration=FREEZE_TEXT_DURATION,
         trainable_generator_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

    def render(label):
        model.eval()
        for i,(name,phrase) in enumerate(probes):
            torch.manual_seed(SEED+i)
            ti=tokenizer(phrase,return_tensors='pt').to(device)
            def guard(module,args,out):
                assert torch.isfinite(out).all(), 'nonfinite predicted log duration'
                assert float(out.max())<12, 'explosive predicted duration'
                frames=torch.ceil(torch.exp(out)).sum()
                assert float(frames)*fx.hop_length/rate < 45, 'predicted utterance exceeds45-second limit'
            handle=model.duration_predictor.register_forward_hook(guard)
            try:
                with torch.no_grad(): y=model(**ti,return_dict=True).waveform.detach().float().cpu().numpy().reshape(-1)
            finally: handle.remove()
            assert y.size and np.isfinite(y).all() and len(y)/rate<45 and np.max(np.abs(y))>1e-6
            path=OUT/f'{label}-{name}-raw.wav'
            sf.write(path,y,rate,subtype='FLOAT')
            status['audio'].append(dict(label=label,probe=name,path=str(path),sha256=sha(path),
                       duration_seconds=len(y)/rate,peak=float(np.max(np.abs(y))),rms=float(np.sqrt(np.mean(y*y)))))
            save(stage=f'{label}_{name}_rendered')

    render('baseline')
    model.train(); discriminator.train()
    torch.manual_seed(SEED)
    wave=torch.from_numpy(audio).to(device).unsqueeze(0)
    with torch.no_grad(): labels,mel=fx._torch_extract_fbank_features(wave)
    assert labels.shape[-1]>=tokens['input_ids'].shape[-1], 'not enough acoustic frames for monotonic alignment'
    mask=torch.ones(1,labels.shape[-1],device=device)
    gen_opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5,betas=(.8,.99),weight_decay=.01)
    disc_opt=torch.optim.AdamW(discriminator.parameters(),lr=2e-5,betas=(.8,.99),weight_decay=.01)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started=time.perf_counter()
    for step in range(1,STEPS+1):
        assert time.perf_counter()-started < TIMEOUT-90, 'training time budget exhausted'
        model.zero_grad(set_to_none=True); discriminator.zero_grad(set_to_none=True)
        output=model(**tokens,labels=labels,labels_attention_mask=mask,
                     monotonic_alignment_function=maximum_path,return_dict=True)
        mel_target=slice_segments(mel,output.ids_slice,model.segment_size)
        mel_gen=fx._torch_extract_fbank_features(output.waveform.squeeze(1))[1]
        target=slice_segments(wave.unsqueeze(1),output.ids_slice*fx.hop_length,model.config.segment_size)
        real,_=discriminator(target); fake,_=discriminator(output.waveform.detach())
        ld=discriminator_loss(real,fake)[0]*3
        assert torch.isfinite(ld), 'nonfinite discriminator loss'
        ld.backward()
        dn=torch.nn.utils.clip_grad_norm_(discriminator.parameters(),1.,error_if_nonfinite=True)
        disc_opt.step(); disc_opt.zero_grad(set_to_none=True)
        _,fmreal=discriminator(target); fake,fmfake=discriminator(output.waveform)
        duration=output.log_duration.sum()
        lm=torch.nn.functional.l1_loss(mel_target,mel_gen)
        lk=kl_loss(output.prior_latents,output.posterior_log_variances,output.prior_means,
                   output.prior_log_variances,output.labels_padding_mask)
        lg=duration+35*lm+1.5*lk+feature_loss(fmreal,fmfake)+generator_loss(fake)[0]
        assert torch.isfinite(lg), 'nonfinite generator loss'
        lg.backward()
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gen_opt.step(); gen_opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        status['metrics'].append(dict(step=step,generator=float(lg.detach()),discriminator=float(ld.detach()),
                                     mel=float(lm.detach()),duration=float(duration.detach()),
                                     generator_grad_norm=float(gn),discriminator_grad_norm=float(dn)))
        save(stage='training',optimizer_steps_completed=step,training_wall_seconds=time.perf_counter()-started,
             peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
        del output,mel_gen,mel_target,target,real,fake,fmreal,fmfake,lg,ld,lm,lk,duration
    del gen_opt,disc_opt,labels,mel,wave
    model.zero_grad(set_to_none=True); discriminator.zero_grad(set_to_none=True)
    model.discriminator=discriminator
    for disc in model.discriminator.discriminators: disc.remove_weight_norm()
    model.decoder.remove_weight_norm()
    for flow in model.flow.flows:
        torch.nn.utils.remove_weight_norm(flow.conv_pre); torch.nn.utils.remove_weight_norm(flow.conv_post)
    changed=sum(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()!=generator_initial[name]
                for name,value in model.state_dict().items() if not name.startswith('discriminator.'))
    assert changed>0, 'generator weights did not change'
    if FREEZE_TEXT_DURATION:
        actual=model.state_dict()
        assert all(hashlib.sha256(actual[name].detach().cpu().contiguous().numpy().tobytes()).hexdigest()==generator_initial[name]
                   for name in frozen_names), 'A frozen text/duration tensor changed'
        save(frozen_tensor_count=len(frozen_names),frozen_tensors_byte_identical=True)
    checkpoint=OUT/'checkpoint'
    model.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint); fx.save_pretrained(checkpoint)
    assert all(sha(p)==h for p,h in source_hashes.items()), 'original source changed'
    save(stage='checkpoint_saved',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint/'model.safetensors'),
         changed_generator_tensors=changed,original_files_preserved=True)
    del model.discriminator
    render('adapted')
    if FREEZE_TEXT_DURATION:
        for name,_ in probes:
            baseline=sf.info(OUT/f'baseline-{name}-raw.wav')
            adapted=sf.info(OUT/f'adapted-{name}-raw.wav')
            assert baseline.frames==adapted.frames, 'Same-seed duration changed despite frozen text/duration weights'
        save(same_seed_probe_durations_preserved=True)
    # Common RMS target and peak safeguard for each baseline/adapted pair.
    for name,_ in probes:
        paths=[OUT/f'{label}-{name}-raw.wav' for label in ['baseline','adapted']]
        waves=[sf.read(p,dtype='float32')[0] for p in paths]
        levels=[float(np.sqrt(np.mean(y.astype('float64')**2))) for y in waves]
        target_rms=min(.08,*(.95*levels[i]/float(np.max(np.abs(y))) for i,y in enumerate(waves)))
        for label,y,rms in zip(['baseline','adapted'],waves,levels):
            gain=target_rms/rms; z=(y*gain).astype('float32')
            p=OUT/f'{label}-{name}-listen.wav'; sf.write(p,z,rate,subtype='FLOAT')
            status['audio'].append(dict(label=label,probe=name,path=str(p),sha256=sha(p),
                                       duration_seconds=len(z)/rate,gain_only=True,gain=gain,
                                       peak=float(np.max(np.abs(z))),target_rms=target_rms))
    save(stage='completed',finished_utc=datetime.now(timezone.utc).isoformat(),
         listening_acceptance='pending Krish; no quality improvement claimed')
except Exception as ex:
    import traceback
    save(stage='failed',error_type=type(ex).__name__,error=str(ex))
    traceback.print_exc()
    sys.exit(1)
