from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('README.md') as f:
    readme = f.read()

setup(name='NumPhi',
      version='0.0.0',
      # packages=['numphi'],
      packages=find_packages(), # use it to find packages in subdirectories of numphi/numphi
      description='The NumPy of philosophical computations.',
      url='https://github.com/bugo99iot/NumPhi',
      author='Ugo Bee',
      author_email='ugo.bertello@gmail.com',
      license='CREATIVE COMMONS',
      install_requires=required,
      long_description=readme,
      include_package_data=True,
      zip_safe=False
      )
