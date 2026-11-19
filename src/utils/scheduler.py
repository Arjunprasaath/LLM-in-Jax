import optax

def warmup_stable_decay_scheduler(
    init_value,
    peak_value,
    warmup_steps,
    stable_steps,
    decay_steps,
    end_value,
    decay_type,
):
    # Warmup phase
    warmup_sched = optax.linear_schedule(
        init_value = init_value,
        end_value = peak_value,
        transition_steps = warmup_steps,
    )
    
    # Stable phase
    stable_sched = optax.constant_schedule(peak_value)

    # Decay phase
    if decay_type == "linear":
        decay_sched = optax.linear_schedule(
            init_value = peak_value,
            end_value = end_value,
            transition_steps = decay_steps,
        )
    elif decay_type == "cosine":
        decay_sched = optax.cosine_decay_schedule(
            init_value = peak_value,
            decay_steps = decay_steps,
            alpha = end_value / peak_value,
        )
    else:
        raise ValueError(f"Unsupported decay_type: {decay_type}")
        
    schedules = [warmup_sched, stable_sched, decay_sched]
    boundries = [warmup_steps, warmup_steps + stable_steps]
    return optax.join_schedules(schedules, boundries)
