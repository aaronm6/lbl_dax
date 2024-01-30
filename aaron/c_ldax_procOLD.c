#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

/* ----------------- <AUX FUNCTIONS> ----------------- */
npy_intp intp_max(npy_intp a, npy_intp b) {
	if (a > b) {
		return a;
	}
	return b;
}
npy_intp intp_min(npy_intp a, npy_intp b) {
	if (a < b) {
		return a;
	}
	return b;
}

/* ----------------- <MODULE FUNCTIONS> ----------------- */
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

static PyObject *meth_arbfilt(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *nd_d, *nd_f;
	long axis = 1;
	static char *input_names[] = {"s_raw", "filter", "axis", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|l", input_names	,
		PyArray_Converter, &nd_d,
		PyArray_Converter, &nd_f,
		&axis)) {
		return NULL;
	}
	
	if ((axis < 0) || (axis > 1)) {
		PyErr_SetString(PyExc_ValueError, "Input 'axis' must be an int, either 0 or 1");
	}
	
	int d_ndim = PyArray_NDIM(nd_d); // dimensionality of the raw signal array
	if (d_ndim > 2) {
		PyErr_SetString(PyExc_TypeError, "Input signal must be either 1d or 2d");
	}
	
	int f_ndim = PyArray_NDIM(nd_f); // dimensionality of the filter function
	if (f_ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Filter function must be 1d");
	}
	
	if (PyArray_TYPE(nd_d) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input signal array must be of dtype numpy.float64");
	}
	if (PyArray_TYPE(nd_f) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Filter function array must be of dtype numpy.float64");
	}
	
	// Create the output signal
	PyObject *nd_o = PyArray_NewLikeArray(nd_d, NPY_ANYORDER, NULL, 1);
	
	npy_intp *dims = PyArray_DIMS(nd_d);
	npy_intp nFilt = PyArray_SIZE(nd_f); // lenght of filter array
	npy_float64 *d = (npy_float64 *)PyArray_DATA(nd_d); // get pointer to first element in raw signal array
	npy_float64 *f = (npy_float64 *)PyArray_DATA(nd_f); // get pointer to first element in filter array
	npy_float64 *o = (npy_float64 *)PyArray_DATA(nd_o);
	npy_intp *d_str_bytes = PyArray_STRIDES(nd_d); // strides in bytes of signal array
	npy_intp *f_str_bytes = PyArray_STRIDES(nd_f); // strides in bytes of filter array
	npy_intp *o_str_bytes = PyArray_STRIDES(nd_o);
	npy_intp d_str[d_ndim], f_str[f_ndim], o_str[d_ndim]; // strides of the two arrays in elements (not bytes)
	npy_intp d_itemsize = PyArray_ITEMSIZE(nd_d); // size of npy_float64, which should be 8
	npy_intp f_itemsize = PyArray_ITEMSIZE(nd_f); // size of npy_float64, which should be 8
	for (int k=0; k<d_ndim; k++) {
		d_str[k] = d_str_bytes[k] / d_itemsize;
		o_str[k] = o_str_bytes[k] / d_itemsize;
	}
	for (int k=0; k<f_ndim; k++) {
		f_str[k] = f_str_bytes[k] / f_itemsize;
	}
	
	npy_float64 f_sum = 0.;
	npy_intp j_mid = nFilt / 2;
	
	// loop over i (not axis), and then j (axis). if axis=1, then i is the row index, j is the column index
	npy_intp i_max, j0_max, a_str, b_str, af_str, bf_str;
	// a_str is the stride (in elements) along the signal axis of the raw array, b_str is along the event axis
	// af_str is the stride (in elements) along the signal axis of the filtered (output) array.
	if (d_ndim == 1) {
		i_max = 1;
		j0_max = dims[0];
		a_str = d_str[0];
		b_str = 0;
		af_str = o_str[0];
		bf_str = 0;
	} else {
		if (axis == 1) {
			i_max = dims[0];
			j0_max = dims[1];
			a_str = d_str[1];
			af_str = o_str[1];
			b_str = d_str[0];
			bf_str = o_str[0];
		} else {
			i_max = dims[1];
			j0_max = dims[0];
			a_str = d_str[0];
			af_str = o_str[0];
			b_str = d_str[1];
			bf_str = o_str[1];
		}
	}
	npy_intp j_max, j_min;
	for (npy_intp i=0; i<i_max; i++) {  // loop over rows (if axis=1, since we're filtering along rows)
		for (npy_intp j0=0; j0<j0_max; j0++) {
			f_sum = 0.;
			j_min = intp_max(j0-j_mid, 0) - j0 + j_mid;
			j_max = intp_min(j0+nFilt, j0_max) - j0;
			for (npy_intp j=j_min; j<j_max; j++) {
				f_sum += f[(nFilt-1-j)*f_str[0]] * d[(j+j0-j_mid)*a_str+i*b_str];
			}
			o[j0*af_str+i*bf_str] = f_sum;
		}
	}
	
	Py_DECREF(nd_d);
	Py_DECREF(nd_f);
	return nd_o;
}
static PyObject *meth_findpulses(PyObject *self, PyObject *args, PyObject *kwargs) {
	// find peaks twice.  I.e. will find the max, step left and right, record.  find the next max, repeat.
	PyArrayObject *nd_d;
	long axis = 1;
	long num_pulses = 2;
	static char *input_names[] = {"f_signal", "axis", "num_pulses", NULL};
	
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|ll", input_names,
		PyArray_Converter, &nd_d,
		&axis,
		&num_pulses)) {
		return NULL;
	}
	if ((axis < 0) || (axis > 1)) {
		PyErr_SetString(PyExc_ValueError, "Input 'axis' must be an int, either 0 or 1");
	}
	if ((num_pulses < 1) || (num_pulses > 32)) {
		PyErr_SetString(PyExc_ValueError, "num_pulses must be between 1 and 32");
	}
	
	int d_ndim = PyArray_NDIM(nd_d); // dimensionality of the raw signal array
	if (d_ndim > 2) {
		PyErr_SetString(PyExc_TypeError, "Input signal must be either 1d or 2d");
	}
	if (PyArray_TYPE(nd_d) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input signal array must be of dtype numpy.float64");
	}
	
	npy_intp *dims = PyArray_DIMS(nd_d);
	npy_intp ndim_o;
	npy_intp dims_o[2] = {0,0};
	// Create the output signal
	if (d_ndim == 1) {
		ndim_o = 1;
		dims_o[0] = 4;
		dims_o[1] = 0;
	} else {
		ndim_o = 2;
		dims_o[0] = 4;
		dims_o[1] = dims[1-axis];
	}
	PyObject *nd_o = PyArray_EMPTY(ndim_o, dims_o, NPY_FLOAT64, NPY_CORDER);
	
	// Create aux array to mask each event when searching twice for pulses.  Will initialize with
	// ones. Conveniently, data are contiguous in memory, so no need for strides
	npy_int8 d_mask[dims[axis]];
	
	npy_float64 *d = (npy_float64 *)PyArray_DATA(nd_d); // get pointer to first element in raw signal array
	npy_float64 *o = (npy_float64 *)PyArray_DATA(nd_o);
	npy_intp *d_str_bytes = PyArray_STRIDES(nd_d); // strides in bytes of signal array
	npy_intp *o_str_bytes = PyArray_STRIDES(nd_o);
	npy_intp d_str[d_ndim], o_str[d_ndim]; // strides of the two arrays in elements (not bytes)
	npy_intp d_itemsize = PyArray_ITEMSIZE(nd_d); // size of npy_float64, which should be 8
	for (int k=0; k<d_ndim; k++) {
		d_str[k] = d_str_bytes[k] / d_itemsize;
		o_str[k] = o_str_bytes[k] / d_itemsize;
	}
	
	// loop over i (not axis), and then j (axis). if axis=1, then i is the row index, j is the column index
	npy_intp i_max, j_max, a_str, b_str, op_str, oe_str;
	// a_str is the stride (in elements) along the signal axis of the raw array, b_str is along the event axis
	// op_str is the output array along the pulse axis, oe_str is the output array along the event axis;
	if (d_ndim == 1) {
		i_max = 1;
		j_max = dims[0];
		a_str = d_str[0];
		b_str = 0;
		op_str = o_str[0];
		oe_str = 0;
	} else {
		op_str = o_str[0];
		oe_str = o_str[1];
		if (axis == 1) {
			i_max = dims[0];
			j_max = dims[1];
			a_str = d_str[1];
			b_str = d_str[0];
			//const npy_intp o_dims[] = {2,dims[0]};
			//nd_o = PyArray_EMPTY(2, o_dims, NPY_FLOAT64, NPY_CORDER);
		} else {
			i_max = dims[1];
			j_max = dims[0];
			a_str = d_str[0];
			b_str = d_str[1];
			//const npy_intp o_dims[] = {2,dims[1]};
			//nd_o = PyArray_EMPTY(2, o_dims, NPY_FLOAT64, NPY_CORDER);
		}
	}
	npy_float64 max_val;
	npy_intp max_pos, p_begin, p_end, k_w;
	printf("i_max = %li\n",i_max);
	for (npy_intp i=0; i<(i_max*b_str); i+=b_str) {  // loop over rows (if axis=1, since we're filtering along rows)
		printf("i = %li\n",i);
		// Ones out the mask array
		for (npy_intp km=0; km<dims[axis]; km++) {
			d_mask[km] = 1;
		}
		for (npy_intp k=0; k<2; k++) {
			printf("    k = %li\n", k);
			// Find max of event
			max_val = -1000.;
			for (npy_intp j=0; j<(j_max*a_str); j+=a_str) {
				if ((d[j+i] > max_val)&&(d_mask[j]!=0)) {
					max_val = d[j+i];
					max_pos = j;
				}
			}
			printf("maxval = %f, maxpos = %li: [", max_val, max_pos);
			// Go to max position and step left until 5% of max val is reached
			k_w = max_pos;
			while ((d[k_w*a_str+i*b_str] > (0.05*max_val))&&(k_w>=0)) {
				k_w--;
			}
			p_begin = k_w;
			printf("%li, ", k_w);
			o[2*k*oe_str+i*op_str] = p_begin;
			
			// Go to max position and step right until 5% of max val is reached
			k_w = max_pos;
			while ((d[k_w*a_str+i*b_str] > (0.05*max_val))&&(k_w<dims[axis])) {
				k_w++;
			}
			p_end = k_w;
			printf("%li]\n", k_w);
			o[(2*k+1)*oe_str+i*op_str] = p_end;
		}
	}
	
	Py_DECREF(nd_d);
	return nd_o;
}

/* ----------------- </MODULE FUNCTIONS> ----------------- */


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
PyDoc_STRVAR(
	arbfilt__doc__,
	"arbfilt(s_raw, filt_array, axis=1)\n--\n\n"
	"Apply an arbitrary filter shape to the signal.  Same usage as avebox.");

static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{"arbfilt",(PyCFunction)meth_arbfilt,METH_VARARGS|METH_KEYWORDS, arbfilt__doc__},
	{"findpeaks",(PyCFunction)meth_findpeaks,METH_VARARGS|METH_KEYWORDS,"help documentation..."},
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
