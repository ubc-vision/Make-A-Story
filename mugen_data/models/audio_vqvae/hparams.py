# ------------------------------------------------------------------------------
# Copyright (c) Meta Platforms, Inc. All Right reserved.
# ------------------------------------------------------------------------------

HPARAMS_REGISTRY = {}
DEFAULTS = {}
AUDIO_SAMPLE_RATE = 22050
AUDIO_SAMPLE_LENGTH = AUDIO_SAMPLE_RATE * 3.2 # each video is 3.2 seconds

class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

def setup_hparams(hparam_set_names, kwargs):
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")
    hparam_sets = [HPARAMS_REGISTRY[x.strip()] for x in hparam_set_names if x] + [kwargs]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H

HPARAMS_REGISTRY["vqvae_coinrun_128x_full_mix"] = Hyperparams(
    sr = AUDIO_SAMPLE_RATE,
    sample_length = AUDIO_SAMPLE_LENGTH,
    levels = 1,
    downs_t = (7,),
    strides_t = (2,),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    loss_fn = 'l2',
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
    restore_vqvae = 'checkpoints/generation/audio_vqvae/jukebox_vqvae_8x_downsample_22khz_small_mix_audio_128x/checkpoint_step_860001.pth.tar',
)

HPARAMS_REGISTRY["vqvae_coinrun_256x_full_mix"] = Hyperparams(
    sr = AUDIO_SAMPLE_RATE,
    sample_length = AUDIO_SAMPLE_LENGTH,
    levels = 1,
    downs_t = (8,),
    strides_t = (2,),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    loss_fn = 'l2',
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
    restore_vqvae = 'checkpoints/generation/audio_vqvae/jukebox_vqvae_8x_downsample_22khz_small_mix_audio_256x/checkpoint_step_640001.pth.tar',
)

HPARAMS_REGISTRY["vqvae_coinrun_512x_full_mix"] = Hyperparams(
    sr = AUDIO_SAMPLE_RATE,
    sample_length = AUDIO_SAMPLE_LENGTH,
    levels = 1,
    downs_t = (9,),
    strides_t = (2,),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    loss_fn = 'l2',
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
    restore_vqvae = 'checkpoints/generation/audio_vqvae/jukebox_vqvae_512x_downsample_22khz_mix_audio/checkpoint_step_1280001.pth.tar',
)

HPARAMS_REGISTRY["vqvae_coinrun_1024x_full_mix"] = Hyperparams(
    sr = AUDIO_SAMPLE_RATE,
    sample_length = AUDIO_SAMPLE_LENGTH,
    levels = 1,
    downs_t = (10,),
    strides_t = (2,),
    emb_width = 64,
    l_bins = 1024,
    l_mu = 0.99,
    commit = 0.02,
    spectral = 0.0,
    multispectral = 1.0,
    loss_fn = 'l2',
    width = 32,
    depth = 4,
    m_conv = 1.0,
    dilation_growth_rate = 3,
    restore_vqvae = 'checkpoints/generation/audio_vqvae/jukebox_vqvae_1024x_downsample_22khz_mix_audio/checkpoint_step_1120001.pth.tar',
)

all_fp16 = Hyperparams(
    fp16=True,
    fp16_params=True,
    fp16_opt=True,
    fp16_scale_window=250,
)
HPARAMS_REGISTRY["all_fp16"] = all_fp16

cpu_ema = Hyperparams(
    ema=True,
    cpu_ema=True,
    cpu_ema_freq=100,
    ema_fused=False,
)
HPARAMS_REGISTRY["cpu_ema"] = cpu_ema


DEFAULTS["rcall"] = Hyperparams(
    rcall_command="<unknown_rcall_command>",
    git_commit="<unknown_git_commit>",
)

DEFAULTS["script"] = Hyperparams(
    name='',
    debug_mem=False,
    debug_eval_files=False,
    debug_speed=False,
    debug_iters=100,
    debug_batch=False,
    debug_grad_accum=False,
    debug_inputs=False,
    local_path='',
    local_logdir='logs',
    max_len=24,
    max_log=32,
    save=True,
    save_iters=20000,
    seed=0,
    prior=False,
    log_steps=100,
    func='',
)

DEFAULTS["data"] = Hyperparams(
    audio_files_dir='',
    finetune='',
    english_only=False,
    bs=1,
    bs_sample=1,
    nworkers=1,
    aug_shift=False,
    aug_blend=False,
    video_cond=False,
    video_gen=False,
    train_test_split=0.9,
    train_shrink_factor=1.0,
    test_shrink_factor=1.0,
    p_unk=0.1,
    min_duration=None,
    max_duration=None,
    n_tokens=0,
    n_vocab=0,
    use_tokens=False,
    curr_epoch=-1,
    video_enc=None,
)

DEFAULTS["vqvae"] = Hyperparams(
    restore_vqvae='',
    levels=2,
    downs_t=(1,1),
    strides_t=(2,2),
    hvqvae_multipliers=None,
    revival_threshold=1.0,
    emb_width=64,
    l_bins=512,
    l_mu=0.99,
    commit=1.0,
    spectral=0.0,
    multispectral=1.0,
    loss_fn='l2',
    linf_k=2048,
    lmix_l1=0.0,
    lmix_l2=0.0,
    lmix_linf=0.0,
    use_bottleneck=True,
)

DEFAULTS["vqvae_conv_block"] = Hyperparams(
    depth=3,
    width=128,
    m_conv=1.0,
    dilation_growth_rate=1,
    dilation_cycle=None,
    vqvae_reverse_decoder_dilation=True,
)

DEFAULTS["prior"] = Hyperparams(
    restore_prior='',
    restore_prior_ddp=False,
    max_bow_genre_size=None,
    y_bins=0,
    level=0,
    cond_levels=None,
    t_bins=64,
    y_cond_as_bias=False,
    copy_input=False,
    merged_decoder=False,
    single_enc_dec=False,
    alignment_layer=None,
    alignment_head=None,
)

DEFAULTS["prior_attn_block"] = Hyperparams(
    n_ctx=1024,
    prior_depth=3,
    prior_width=128,
    heads=1,
    attn_order=0,
    blocks=None,
    spread=None,
    attn_dropout=0.0,
    resid_dropout=0.0,
    emb_dropout=0.0,
    zero_out=False,
    res_scale=False,
    pos_init=False,
    init_scale=1.0,
    m_attn=0.25,
    m_mlp=1.0,
    c_res=0,
    c_attn=0,
    c_mlp=0,
)

DEFAULTS["cond_conv_block"] = Hyperparams(
    cond_depth=3,
    cond_width=128,
    cond_m_conv=1.0,
    cond_zero_out=False,
    cond_res_scale=False,
    cond_dilation_growth_rate=1,
    cond_dilation_cycle=None,
    cond_c_res=0,
)

DEFAULTS["sample"] = Hyperparams(
    primed_chunk_size=None,
    selected_artists='',
    temp_top=1.0,
    temp_rest=0.99,
    sample_length_in_seconds=24,
    total_sample_length_in_seconds=240,
)

DEFAULTS["prime"] = Hyperparams(
    #encoder_kv_width=128,
    prime_loss_fraction=0.1,
    restore_decoder='',
)
DEFAULTS["prime_attn_block"] = Hyperparams(
    prime_depth=3,
    prime_width=128,
    prime_heads=1,
    prime_attn_order=0,
    prime_blocks=None,
    prime_spread=None,
    prime_attn_dropout=0.0,
    prime_resid_dropout=0.0,
    prime_emb_dropout=0.0,
    prime_zero_out=False,
    prime_res_scale=False,
    prime_pos_init=False,
    prime_init_scale=1.0,
    prime_m_attn=0.25,
    prime_m_mlp=1.0,
    prime_c_res=0,
    prime_c_attn=0,
    prime_c_mlp=0,
    prime_rel_attn=False,
    prime_posemb_timescale=10000,
)

DEFAULTS["opt"] = Hyperparams(
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
)

DEFAULTS["fp16"] = Hyperparams(
    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,
)

DEFAULTS["train_test_eval"] = Hyperparams(
    labels=False,
    labels_v3=False,
    dump=False,
    ema=True,
    ema_fused=True,
    cpu_ema=False,
    cpu_ema_freq=100,
    reset_best_loss=False,
    reset_step=False,
    reset_opt=False,
    reset_shd=False,
    train=False,
    test=False,
    sample=False,
    sampler='ancestral',
    codes_logdir='',
    date=None,
    labeller='top_genres',
    label_line=0,
    iters_before_update=1,
    grad_accum_iters=0,
    mu=None,
    piped=False,
    pipe_depth=8,
    break_train=1e10,
    break_test=1e10,
    exit_train=1e10,
)

DEFAULTS["audio"] = Hyperparams(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=44100,
    channels=2,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_length=0,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,
    use_nonrelative_specloss=True,
    multispec_loss_n_fft=(2048,1024,512),
    multispec_loss_hop_length=(240,120,50),
    multispec_loss_window_size=(1200,600,240),
)

DEFAULTS["distributed"] = Hyperparams(
    bucket=128
)
