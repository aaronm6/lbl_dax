from setuptools import setup, Extension, find_packages
import os, re

def get_version():
    version_file = os.path.join(
        os.path.dirname(__file__),
        'ldax_processing',
        '_version.py')
    with open(version_file, 'r') as f:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]",
            f.read(),
            re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

module = Extension(
    "ldax_processing.c_ldax_proc", 
    sources=["src/c_ldax_proc.c"], 
    include_dirs=["src"],
    extra_compile_args=["-Wall"])

setup(
    name="ldax_processing",
    version=get_version(),
    description="Data-processing library for ldax",
    author="Aaron Manalaysay",
    license="GPL-2.1",
    ext_modules=[module],
    packages=find_packages(where='.',include=["ldax_processing"]),
    install_requires=['numpy<2','varray>=1.1.5'],
)
#    build_requires=['numpy>=2']
#    packages=["ldax_processing"]
#)
#    py_modules=["ldax_processing"],
#,"ldax_processing/c_ldax_proc"]
#)

