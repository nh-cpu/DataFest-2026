"""Alternative command-line entry point for the same notebook workflow."""
import argparse
from warranty_model import prepare_data, make_splits, fit_references, train_model, export_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--output-dir', default='outputs')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    data, audit = prepare_data(args.data_dir)
    splits = make_splits(data)
    print(f'Prepared {len(data)} units', flush=True)
    references, rows = fit_references(data, splits, args.jobs)
    print('Baselines complete; tuning 150 forests', flush=True)
    search, model = train_model(data, splits, args.jobs)
    metrics = export_results(data, splits, audit, references, rows, search, model, args.output_dir)
    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
