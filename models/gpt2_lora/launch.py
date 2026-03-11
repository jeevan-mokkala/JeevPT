import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = "arn:aws:iam::309986686416:role/ec2-ml-s3-access"

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    dependencies=["../../data.py"],
    role=role,
    base_job_name="gpt2-lora-r4",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "rank": 4,
        "batch_size": 8,
        "epochs": 10,
        "lr": 3e-4,
    },
)

estimator.fit({"training": "s3://jeevanchy-data-demo/data/"})
