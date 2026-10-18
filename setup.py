from setuptools import setup, find_packages

setup(
    name="lightning-whisper-mlx",
    version="0.0.10",
    packages=find_packages(),
    package_data={
        "lightning_whisper_mlx": ["assets/*"],
        "benchmarks": ["whisper_cpp_worker.cpp"],
    },
    install_requires=[
        "huggingface_hub",
        "mlx>=0.32.2",
        "numba",
        "numpy",
        "more-itertools",
        "tiktoken>=0.3.3,<1",
        "scipy",
    ],
    extras_require={"dev": ["pytest", "black", "flake8"]},
    entry_points={
        "console_scripts": [
            "lightning-whisper-mlx=lightning_whisper_mlx.cli:main",
        ],
    },
)
