from setuptools import setup, find_packages

setup(
    name='opents',
    version='0.1.3.1',
    packages=find_packages(
        exclude=[
            'demo',
            'doc',
            'dist'
            ]
    ),
    license='MIT',
    author='hushuguo',
    author_email='husg8217@mails.jlu.edu.cn',
    install_requires=[
        'torch',
        'numpy',
        'tqdm'
    ],
    description='OpenTS is a friendly Python Library for time series analysis',
    long_description=open('README.md','r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/PyOpenTS/PyOpenTS'
)
