import sagemaker
from sagemaker.pytorch import PyTorch

session = sagemaker.Session()
role = sagemaker.get_execution_role()

estimator = PyTorch(
    entry_point="train.py",
    source_dir=".",
    dependencies=["../../data.py"],
    role=role,
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "d_model": 768,
        "num_blocks": 12,
        "num_heads": 12,
        "seq_len": 256,
        "batch_size": 8,
        "epochs": 3,
        "lr": 1e-5,
    },
)

estimator.fit({"training": "s3://jeevanchy-data-demo/data/"})
