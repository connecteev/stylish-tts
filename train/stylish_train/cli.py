import click
import importlib.resources
import config


def get_configs(config_path, model_path):
    if len(model_path) == 0:
        f = importlib.resources.open_text(config, "model.yml")
    else:
        f = open(model_path, "r")


@click.group()
def cli():
    pass


@cli.group("dataprep", short_help="Utilities for preparing a dataset for training")
def dataprep():
    pass


@dataprep.command(
    "train-align",
    short_help="Train an alignment model to use for pre-caching alignments.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    help="Path to config file (use config/config.yml as a template)",
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--out", type=str, help="Output directory for logs, checkpoints, and models"
)
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
@click.argument(
    "out-file",
    required=True,
    type=str,
)
def train_align(*args, **kwargs):
    """Train alignment model

    <out-file> is the filename where the resulting alignment model will be saved.
    """
    print(args, kwargs)


@dataprep.command(
    short_help="Use a pretrained alignment model to create a cache of alignments for training."
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    help="Path to config file (use config/config.yml as a template)",
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--model",
    type=str,
    help="Pretrained alignment model created previously with the train-align command.",
)
@click.argument(
    "out-file",
    type=str,
)
def align(*args, **kwargs):
    """Align dataset

    Use an alignment model to precache the alignments for your dataset. The alignments are saved to the alignment_path from the config file.
    """
    print(args, kwargs)


@dataprep.command(short_help="Create a cache of pitches to use for training.")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    help="Path to config file (use config/config.yml as a template)",
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.argument(
    "out-file",
    type=str,
)
def pitch(*args, **kwargs):
    """Calculate pitch for a dataset

    Calculates the fundamental frequencies for every segment in your dataset. The pitches are saved to the pitch_path from the config file.
    """
    print(args, kwargs)


@cli.command(short_help="Train a model using the specified configuration.")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    help="Path to config file (use config/config.yml as a template)",
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--out", type=str, help="Output directory for logs, checkpoints, and models"
)
@click.option(
    "--stage",
    default="acoustic",
    type=str,
    help="Training stage should be one of 'acoustic', 'textual', 'style', 'duration'.",
)
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
def train(**kwargs):
    """Train a model

    Train a Stylish TTS model. You must have already precached alignment and pitch information for the dataset. Stage should be 'acoustic' to begin with unless you are loading a checkpoint.
    """
    print(kwargs)


@cli.command(short_help="Convert a model to ONNX for use in inference.")
@click.option(
    "--checkpoint", type=str, help="Path to a model checkpoint to load for conversion"
)
@click.argument(
    "out-file",
    required=True,
    type=str,
)
def convert(*args, **kwargs):
    """Convert a model to ONNX

    The converted model will be saved in <out-file>.
    """
    print(args, kwargs)


if __name__ == "__main__":
    cli()
