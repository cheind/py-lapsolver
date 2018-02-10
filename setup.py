import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        super(CMakeExtension, self).__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

class PyTest(test):
    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['lapsolver/tests'])
        sys.exit(errno)


version = open('lapsolver/__init__.py').readlines()[-1].split()[-1].strip('\'')
if os.getenv('LAPSOLVER_RELEASE_TYPE', 'stable') == 'dev':
    version = version + '.dev' + os.getenv('LAPSOLVER_DEV_NUM', '0') # this will allow pre-releases pip install --pre lapsolver

setup(
    name='lapsolver',
    version=version,
    author='Christoph Heindl',
    url='https://github.com/cheind/py-lapsolver',
    description='Fast linear assignment problem solvers',
    license='MIT',
    long_description='',
    packages=['lapsolver', 'lapsolver.tests'],
    include_package_data=True,
    ext_modules=[CMakeExtension('lapsolverc')],
    cmdclass=dict(build_ext=CMakeBuild, test=PyTest),
    zip_safe=False,
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    keywords='hungarian munkres kuhn linear-sum-assignment bipartite-graph lap'
)