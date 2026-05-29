from setuptools import find_packages, setup


setup(
    name='lets_be_rational',
    version='1.1.2',
    packages=find_packages(),
    url='http://jaeckel.org',
    license='MIT',
    maintainer='vollib',
    maintainer_email='vollib@gammoncap.com',
    description='Pure python implementation of Peter Jaeckel\'s LetsBeRational.',
    install_requires = [
        'cody-special>=1.0.0,<2.0.0',
        'piecewise-rational>=1.0.0,<2.0.0',
        'numpy>=1.20'
    ]
)
