import setuptools

setuptools.setup(name='quoteextract',
version='1.0',
description='Speaker pipeline on top of quote extraction',
url='#',
author='max',
install_requires=[
    'bson>=0.5.10',
    'spacy>=2.1.3', 
    'nltk>=3.4.5',
    'neuralcoref>=4.0',
    'en-core-web-lg @ https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.3.0/en_core_web_lg-3.3.0-py3-none-any.whl'
    ],
author_email='',
packages=setuptools.find_packages(),
include_package_data=True,
zip_safe=False)