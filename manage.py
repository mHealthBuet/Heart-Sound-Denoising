import click

from predict import CleanHeartSounds


class Context:
    log_group_name = ""
    log_stream_name = ""


@click.group()
def cli():
    pass


@cli.command()
def debug():
    chs = CleanHeartSounds()

    clean = chs.clean_heart_sounds(
        audiofiles_dir="Data/predict",
        model_dir="Models/LU-Net.h5",
        model_input_shape=800,
    )
    print(clean)

    chs.compare_clean("Data/predict/PHS.wav")
    chs.compare_clean("Data/predict/ICBHI.wav")


if __name__ == "__main__":
    cli()
