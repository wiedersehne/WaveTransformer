import torch

def stack_batches(dataloader):
    # Stack all test data for test hook
    CNA, covariates, labels, survival_time, survival_status = [], [], [], [], []
    for batch in iter(dataloader()):
        CNA.append(batch["CNA"])
        covariates.append(batch["covariates"])
        survival_time.append(batch["survival_time"])
        survival_status.append(batch["survival_status"])
        labels.append(batch["label"])
    return {"CNA": torch.concat(CNA, 0),
            "covariates": torch.concat(covariates, 0),
            "label": torch.concat(labels, 0),
            "survival_time": torch.concat(survival_time, 0),
            "survival_status": torch.concat(survival_status, 0)
            }