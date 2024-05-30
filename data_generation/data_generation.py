import argparse
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata


def generate_data(input_path: str, output_path: str, rows_number: int) -> None:
    data = pd.read_csv(input_path)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data)

    new_data = synthesizer.sample(num_rows=rows_number)

    new_data.to_csv(output_path, index=False)

def parse_arguments()-> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='Path to input csv file.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                    help='Output path for generated data.')
    parser.add_argument('-r', '--rows_number', type=int, required=True,
                        help='Number of rows to generate')
    return parser.parse_args()

def main(args: argparse.Namespace)-> None:
    generate_data(args.input_path, args.output_path, args.rows_number)


if __name__ == '__main__':
    main(parse_arguments())