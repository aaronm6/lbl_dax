from setuptools import setup, Extension, find_packages

module = Extension(
    "ldax_processing.c_ldax_proc", 
    sources=["src/c_ldax_proc.c"], 
    include_dirs=["src"],
    extra_compile_args=["-Wall"])

#pkgdir = '~/pylab/lbl_dax/data_proc/ldax_processing/ldax_processing/'
#print(find_packages(include=["ldax_processing"]))
setup(
    name="ldax_processing",
    version="0.0.1",
    description="Data-processing library for ldax",
    author="Aaron Manalaysay",
    license="GPL-2.1",
    ext_modules=[module],
    packages=find_packages(where='.',include=["ldax_processing"]),
    install_requires=['numpy<2'],
)
#    build_requires=['numpy>=2']
#    packages=["ldax_processing"]
#)
#    py_modules=["ldax_processing"],
#,"ldax_processing/c_ldax_proc"]
#)

