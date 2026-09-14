from layers.NTE import NTE
from models.GTR import Model as GTRModel


class Model(GTRModel):
    """GTR with a parameter-free NTE wrapper around the complete backbone."""

    def __init__(self, configs):
        super().__init__(configs)
        self.nte = NTE(
            pred_len=configs.pred_len,
            cutoff_ratio=getattr(configs, "nte_cutoff_ratio", 0.1),
            alpha=getattr(configs, "nte_alpha", 1.0),
            gamma_max=getattr(configs, "nte_gamma_max", 20.0),
            guard_sigma=getattr(configs, "nte_guard_sigma", 3.0),
        )

    def forward(self, x, cycle_index):
        residual = self.nte(x, mode="norm")
        residual_forecast = super().forward(residual, cycle_index)
        return self.nte(residual_forecast, mode="denorm")
