# from models.export_model import ExportModel
# count_parameters(ExportModel(**train.model, device=train.config.training.device))
# exit(0)


def count_parameters(model):
    table = PrettyTable(["Module", "Parameters (M)"])
    summary = defaultdict(float)
    total_params = 0

    for name, parameter in model.named_parameters():
        module = ".".join(name.split(".")[:2])
        summary[module] += parameter.numel() / 1_000_000
        total_params += parameter.numel() / 1_000_000

    for module, params in summary.items():
        table.add_row([module, f"{params:.3}M"])

    print(table)
    print(f"Total Trainable Params: {total_params:,.2f}M")
    return total_params
