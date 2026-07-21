"""Run the causal inference and uplift modeling pipeline."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import CausalPipeline


def main():
    parser = argparse.ArgumentParser(description="Causal Inference & Uplift Modeling Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    parser.add_argument("--data", type=str, default=None, help="Path to data file")
    parser.add_argument("--sample", action="store_true", help="Use sample data")
    args = parser.parse_args()

    if args.data is None and not args.sample:
        print("No data provided. Using sample data.")
        args.sample = True

    pipeline = CausalPipeline(config_path=args.config)
    results = pipeline.run(data_path=args.data, use_sample_data=args.sample)

    print("\nKey results:")
    print(f"  ATE: {results['dowhy']['ate']:.6f}")
    print(f"  Best model: {results['uplift'].iloc[0]['model']}")
    print(f"  Qini: {results['uplift'].iloc[0]['mean_qini']:.4f}")


if __name__ == "__main__":
    main()
