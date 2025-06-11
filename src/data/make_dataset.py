# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from src.features.build_features import build_all_features

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Procesa datos crudos desde input_filepath,
    genera nuevas features y guarda el resultado en output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'📥 Cargando datos desde: {input_filepath}')

    # Leer datos
    df = pd.read_csv(input_filepath)
    logger.info(f'✅ Datos cargados con shape: {df.shape}')

    # Aplicar ingeniería de variables
    df = build_all_features(df)
    logger.info(f'🧠 Features generadas: {df.columns.tolist()}')

    # Guardar datos transformados
    output_path = Path(output_filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f'💾 Datos guardados en: {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())

    main()