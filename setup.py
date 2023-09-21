from setuptools import setup, find_packages

setup(
    name='out_and_in',
    version='0.8',
    packages=find_packages(),
    install_requires=[
        'pandas==2.0.3',
        'scikit-learn==1.3.0',
        'tqdm==4.65.0'
    ],
    author='chenfeng Meng',
    author_email='chenfengmeng0@gmail.com',
    description='a backward feature elimination followed with forwared features selection strategy. ',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/chenfengMeng2021/out_and_in',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
