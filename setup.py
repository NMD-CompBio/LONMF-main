import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LONMF",# Replace with your own username
    version="0.0.1",
    author="Mengdi Nan",
    author_email="2139698547@qq.com",
    description="A non-negative matrix factorization model based on graph Laplacian and optimal transmission for Paired single-cell multi-omics data integration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NMD-CompBio/LONMF",
    packages=setuptools.find_packages(),
    py_modules=['LONMF.model', 'LONMF.utils'],  # 添加多个模块
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

