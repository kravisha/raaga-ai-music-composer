"""Bounded synthetic VITS GAN step. No human data or learned checkpoint saved."""
import os,sys,json,time,subprocess,hashlib,types
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/Krish/ClaudeStuff/Raaga-SPB-Direct-TTS-20260910')
LONG='--long' in sys.argv
WEIGHTNORM='--weightnorm' in sys.argv
OUT=ROOT/('runtime-probe-20s-weightnorm-01' if WEIGHTNORM else ('runtime-probe-20s-01' if LONG else 'runtime-probe-01'))
for k,v in {'HF_HOME':str(ROOT/'cache/huggingface'),'TRANSFORMERS_CACHE':str(ROOT/'cache/huggingface/hub'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','MPLCONFIGDIR':str(ROOT/'cache/matplotlib'),'WANDB_MODE':'disabled','WANDB_DISABLED':'true'}.items():os.environ[k]=v
if '--worker' not in sys.argv:
    OUT.mkdir(exist_ok=False)
    (OUT/'probe-script.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'mms_portable_alignment.py').write_bytes(Path(__file__).with_name('mms_portable_alignment.py').read_bytes())
    rec={'started_utc':datetime.now(timezone.utc).isoformat(),'timeout_seconds':300,'human_audio_used':False,'learned_checkpoint_saved':False}
    with (OUT/'worker.log').open('w',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,'-B',__file__,'--worker']+(['--long'] if LONG else [])+(['--weightnorm'] if WEIGHTNORM else []),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        rec['pid']=p.pid;(OUT/'launch.json').write_text(json.dumps(rec,indent=2))
        print(json.dumps(rec),flush=True)
        try:rec['exit_code']=p.wait(timeout=300)
        except subprocess.TimeoutExpired:p.kill();p.wait();rec.update({'exit_code':p.returncode,'timeout':True})
    rec['finished_utc']=datetime.now(timezone.utc).isoformat();(OUT/'launch.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec),flush=True)
    sys.exit(0 if rec['exit_code']==0 else 1)

status={'stage':'imports','human_audio_used':False,'learned_checkpoint_saved':False,'optimizer_steps_completed':0}
def save(**kw):
    status.update(kw);(OUT/'result.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(kw),flush=True)
try:
    import torch,numpy as np
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(20260910);np.random.seed(20260910)
    sys.path.insert(0,str(ROOT/'training-source'))
    from utils import VitsModelForPreTraining,VitsFeatureExtractor,slice_segments
    from utils.modeling_vits_training import monotonic_align_max_path
    from transformers import VitsTokenizer
    from mms_portable_alignment import maximum_path
    facade=types.ModuleType('monotonic_align');facade.maximum_path=maximum_path;sys.modules['monotonic_align']=facade
    from run_vits_finetuning import discriminator_loss,feature_loss,generator_loss,kl_loss
    checked=0
    for frames in [8,17,64]:
        for tokens in [1,3,7]:
            for valid_frames in [frames,frames-1]:
                if valid_frames<tokens:continue
                scores=torch.randn(2,frames,tokens);mask=torch.zeros_like(scores);mask[:,:valid_frames,:]=1
                expected=monotonic_align_max_path(scores.clone(),mask.clone())
                actual=maximum_path(scores.clone(),mask.clone())
                assert torch.equal(expected,actual)
                checked+=1
    save(stage='alignment_verified',alignment_cases=checked,alignment_backend='explicit Numba adapter; source Cython build failed missing MSVC; upstream Python recurrence exact-match fixtures')
    device='cuda';assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.85,0)
    folder=ROOT/'assets/mms-tts-tam-train-481128b97cd2'
    weight=folder/'model.safetensors';before=hashlib.sha256(weight.read_bytes()).hexdigest()
    model=VitsModelForPreTraining.from_pretrained(folder,local_files_only=True).to(device).train()
    if WEIGHTNORM:
        model.decoder.apply_weight_norm()
        for flow in model.flow.flows:
            torch.nn.utils.weight_norm(flow.conv_pre)
            torch.nn.utils.weight_norm(flow.conv_post)
    discriminator=model.discriminator;del model.discriminator
    for disc in discriminator.discriminators:disc.apply_weight_norm()
    tokenizer=VitsTokenizer.from_pretrained(folder,local_files_only=True)
    fx=VitsFeatureExtractor.from_pretrained(folder,local_files_only=True)
    seconds=20 if LONG else 1
    tokens=tokenizer('அத '*30 if LONG else 'அத',return_tensors='pt').to(device)
    wave=(.05*torch.sin(2*torch.pi*220*torch.arange(16000*seconds,device=device)/16000)).unsqueeze(0)
    labels,mel=fx._torch_extract_fbank_features(wave)
    mask=torch.ones(1,labels.shape[-1],device=device)
    gen_opt=torch.optim.AdamW(model.parameters(),lr=2e-5,betas=(.8,.99))
    disc_opt=torch.optim.AdamW(discriminator.parameters(),lr=2e-5,betas=(.8,.99))
    save(stage='model_loaded',gpu=torch.cuda.get_device_name(0),input_seconds=seconds,text_token_count=int(tokens['input_ids'].shape[-1]),batch_size=1,precision='float32',generator_weightnorm_matches_upstream_recipe=WEIGHTNORM,generator_parameters=sum(p.numel() for p in model.parameters()),discriminator_parameters=sum(p.numel() for p in discriminator.parameters()))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    output=model(**tokens,labels=labels,labels_attention_mask=mask,monotonic_alignment_function=maximum_path,return_dict=True)
    mel_target=slice_segments(mel,output.ids_slice,model.segment_size)
    mel_gen=fx._torch_extract_fbank_features(output.waveform.squeeze(1))[1]
    target=slice_segments(wave.unsqueeze(1),output.ids_slice*fx.hop_length,model.config.segment_size)
    real,_=discriminator(target);fake,_=discriminator(output.waveform.detach())
    ld=discriminator_loss(real,fake)[0]*3
    assert torch.isfinite(ld);ld.backward();torch.nn.utils.clip_grad_norm_(discriminator.parameters(),1.);disc_opt.step();disc_opt.zero_grad(set_to_none=True)
    _,fmreal=discriminator(target);fake,fmfake=discriminator(output.waveform)
    duration=output.log_duration.sum();lm=torch.nn.functional.l1_loss(mel_target,mel_gen)
    kl=kl_loss(output.prior_latents,output.posterior_log_variances,output.prior_means,output.prior_log_variances,output.labels_padding_mask)
    lg=duration+35*lm+1.5*kl+feature_loss(fmreal,fmfake)+generator_loss(fake)[0]
    assert torch.isfinite(lg);lg.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);gen_opt.step();gen_opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    save(stage='synthetic_step_passed',optimizer_steps_completed=1,step_seconds=time.perf_counter()-started,loss_generator=float(lg.detach()),loss_discriminator=float(ld.detach()),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),generator_optimizer_state_entries=len(gen_opt.state),discriminator_optimizer_state_entries=len(disc_opt.state),checkpoint_unchanged=hashlib.sha256(weight.read_bytes()).hexdigest()==before,limits=[f'One{seconds}-second synthetic signal only; not speech training, not quality evidence','Does not establish real-speech or repeated-step memory/stability','Training source unchanged; explicit portable alignment module injected for this diagnostic'])
except Exception as ex:
    import traceback
    save(stage='failed',error_type=type(ex).__name__,error=str(ex));traceback.print_exc();sys.exit(1)
