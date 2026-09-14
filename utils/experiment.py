def numeric_tag(value):
    """Return a filesystem-safe, stable label for a numeric value."""

    return format(float(value), "g").replace("-", "m").replace(".", "p")


def experiment_setting(args, seed):
    """Build the checkpoint/result identity for one forecasting run."""

    setting = '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.cycle,
        seed,
    )
    if args.model == 'GTRNTE':
        setting += '_nte_k{}_a{}_g{}'.format(
            numeric_tag(args.nte_cutoff_ratio),
            numeric_tag(args.nte_alpha),
            numeric_tag(args.nte_gamma_max),
        )
    return setting
