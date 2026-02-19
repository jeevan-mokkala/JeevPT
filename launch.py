import argparse
import sagemaker
from sagemaker.pytorch import PyTorch


def main():
    parser = argparse.ArgumentParser(description="Launch CLTSM training on SageMaker")
    parser.add_argument("--role", type=str,
                        default="arn:aws:iam::309986686416:role/ec2-ml-s3-access",
                        help="SageMaker IAM role ARN")
    parser.add_argument("--data", type=str, default="s3://jeevanchy-data-demo/data/",
                        help="S3 URI for training data")
    parser.add_argument("--instance-type", type=str, default="ml.g4dn.xlarge")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    sess = sagemaker.Session()

    estimator = PyTorch(
        entry_point="train.py",
        source_dir=".",
        role=args.role,
        sagemaker_session=sess,
        instance_type=args.instance_type,
        instance_count=1,
        framework_version="2.1",
        py_version="py310",
        hyperparameters={
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "seq-len": 512,
            "d-model": 256,
            "n-heads": 8,
            "n-layers": 6,
            "d-ff": 1024,
            "dropout": 0.1,
        },
    )

    estimator.fit({"training": args.data})

    print(f"\nTraining complete!")
    print(f"Model artifacts: {estimator.model_data}")


if __name__ == "__main__":
    main()
