#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

static PyObject *meth_avebox(PyObject *self, PyObject *args, PyObject *kwargs) {
	/* Usage: s_filt = avebox(s_raw, n, axis=1)
		 s_raw: raw signal.  Numpy array either 1d, or 2d.  If 2d, 
		        the signals will be filtered along axis.
		     n: The number of samples in the box. n must be an ODD number
		  axis: If s_raw is 2d, the filtering will occur along this axis.
		        Default is axis=1, which means along the ROWS.
		s_filt: The filtered signal.  Will have the same size as s_raw.*/
	static char *keywords[] = {"s_raw", "n", "axis", NULL};
	
	PyArrayObject *nd_s;
	long n;
	long axis=1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&l|l", keywords,
		PyArray_Converter, &nd_s,
		&n,
		&axis)) {
		return NULL;
	}
	if ((n%2)==0) {
		PyErr_SetString(PyExc_ValueError, "Input 'n' must be an ODD number");
	}
	if (axis < -1) {
		PyErr_SetString(PyExc_ValueError, "Optional input 'axis' must be an integer greater than -1");
	}
	//if (axis != 1) {
	//	PyErr_SetString(PyExc_ValueError, "Sorry, 'axis=1' is currently the only allowed value.");
	//}
	int ndim = PyArray_NDIM(nd_s);
	if (ndim > 2) {
		PyErr_SetString(PyExc_TypeError, "Input raw signal 's_raw' must be 1d or 2d");
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must have a dtype of numpy.float64");
	}
	// nd_f will be the filtered signal array
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_s);
	npy_intp *dims = PyArray_DIMS(nd_s);
	
	npy_float64 *s_el, *f_el; // pointers to individual array elements
	npy_float64 n_dbl = (npy_float64)n;
	npy_intp n_half_floor = (npy_intp)(n/2);
	npy_intp n_half_ceil = n_half_floor + 1;
	npy_float64 f_sum;
	if (ndim==1) {
		npy_float64 *s_el, *f_el; // pointers to individual array elements
		f_sum = 0.;
		for (npy_intp i=0; i<n_half_ceil; i++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
			f_sum += *s_el;
		}
		//First for loop covers elements whose indices are less than half the width of the box
		for (npy_intp i=0; i<n_half_floor; i++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
			f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
			*f_el = f_sum / n_dbl;
			f_sum += s_el[n_half_ceil];
		}
		
		//Second for loop covers the main array
		for (npy_intp i=n_half_floor; i<(numEl-n_half_ceil); i++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
			f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
			*f_el = f_sum / n_dbl;
			f_sum += s_el[n_half_ceil];
			f_sum -= s_el[-n_half_floor];
		}
		
		//Third for loop covers the elements whose indices are closer to the end of the
		//array than half the width of the box
		for (npy_intp i=(numEl-n_half_ceil); i<numEl; i++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
			f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
			*f_el = f_sum / n_dbl;
			f_sum -= s_el[-n_half_floor];
		}
	} else {
		npy_float64 *s = (npy_float64 *)PyArray_DATA(nd_s);
		npy_float64 *f = (npy_float64 *)PyArray_DATA(nd_f);
		npy_intp *s_str_bytes = PyArray_STRIDES(nd_s);
		npy_intp *f_str_bytes = PyArray_STRIDES(nd_f);
		npy_intp s_str[ndim], f_str[ndim];
		npy_intp s_itemsize = PyArray_ITEMSIZE(nd_s);
		for (int k=0; k<ndim; k++) {
			s_str[k] = s_str_bytes[k] / s_itemsize;
			f_str[k] = f_str_bytes[k] / s_itemsize;
		}
		npy_intp i_maj = 0;
		npy_intp i_min = 0;
		for (npy_intp i=0; i<dims[1-axis]; i++) {
			f_sum = 0.;
			for (npy_intp j=0; j<n_half_ceil; j++) {
				f_sum += s[j*s_str[axis]+i*s_str[1-axis]];
			}
			// First loop covers elements whose indices are less than half the width of the box
			for (npy_intp j=0; j<n_half_floor; j++) {
				f[j*f_str[axis]+i*f_str[1-axis]] = f_sum / n_dbl;
				f_sum += s[(j+n_half_ceil)*s_str[axis]+i*s_str[1-axis]];
			}
			// Second loop covers the main signal
			for (npy_intp j=n_half_floor; j<(dims[axis]-n_half_ceil); j++) {
				f[j*f_str[axis]+i*f_str[1-axis]] = f_sum / n_dbl;
				f_sum += s[(j+n_half_ceil)*s_str[axis]+i*s_str[1-axis]];
				f_sum -= s[(j-n_half_floor)*s_str[axis]+i*s_str[1-axis]];
			}
			// Third loop covers elements close to the end of the array
			for (npy_intp j=(dims[axis]-n_half_ceil); j<dims[axis]; j++) {
				f[j*f_str[axis]+i*f_str[1-axis]] = f_sum / n_dbl;
				f_sum -= s[(j-n_half_floor)*s_str[axis]+i*s_str[1-axis]];
			}
		}
	}
	Py_DECREF(nd_s);
	return nd_f;
}


PyDoc_STRVAR(
	avebox__doc__,
	"avebox(s_raw, n, axis=1)\n--\n\n"
	"Apply a box average filter to a signal.\n"
	" s_raw: raw signal.  Numpy array either 1d, or 2d.  If 2d,\n"
	"        the signals will be filtered along axis.\n"
	"     n: The number of samples in the box. n must be an ODD number\n"
	"  axis: If s_raw is 2d, the filtering will occur along this axis.\n"
	"        Default is axis=1, which means along the ROWS.\n"
	"s_filt: The filtered signal.  Will have the same size as s_raw.");

static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef ldax_module = {
	PyModuleDef_HEAD_INIT,
	"ldax_methods",
	"Data processing methods for ldax data",
	-1,
	ldax_methods
};

PyMODINIT_FUNC PyInit_c_ldax_proc(void) {
	import_array();
	return PyModule_Create(&ldax_module);
}
