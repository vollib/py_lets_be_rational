from distutils.core import setup


setup(
    name='py_lets_be_rational',
    version='1.0.0',
    packages=['py_lets_be_rational'],
    url='http://jaeckel.org',
    license='MIT',
    maintainer='vollib',
    maintainer_email='vollib@gammoncap.com',
    description='Pure python implementation of Peter Jaeckel\'s LetsBeRational.',
    install_requires = [
        'numpy==1.12.1',
        'numba==0.31.0'
    ]
)
