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

PyArrayObject *slice_1d(PyObject *array, long idx, long axis) {
	// axis=0 means: slice off a column
	// axis=1 means: slice off a row
	PyArrayObject *n_array;
	int ndim_i = PyArray_NDIM(array);
	npy_intp *dims_i = PyArray_DIMS(array);
	npy_intp *strides_i = PyArray_STRIDES(array);
	
	npy_intp ndim_s = 1;
	npy_intp dims_s[] = {dims_i[axis]};
	npy_intp strides_s[] = {strides_i[axis]};
	
	Py_INCREF(PyArray_DESCR(array));
	
	n_array = (PyArrayObject *)PyArray_NewFromDescr(
		&PyArray_Type,
		PyArray_DESCR(array),
		ndim_s,
		dims_s,
		strides_s,
		PyArray_DATA(array) + idx * strides_i[1-axis],
		PyArray_FLAGS(array),
		(PyObject *)array);
	
	//n_array->base = ((PyArrayObject *)array)->base ? ((PyArrayObject *)array)->base : array;
	/* 
	 Technically, the base of an array needs to be the array that owns the data, which might
	 not be 'array' (if 'array' was built from another object).  So for example:
	 >>> a0 = np.array([0,1,2,3,4,5]) # i.e. 1d
	 >>> a = a.reshape((2,3)) # i.e. 2d, 2 rows, 3 columns: [[0,1,2],[3,4,5]]
	 >>> b = slice1d(a, 1, axis=1) # b = [3,4,5] 1d
	 In the above, a's base is a0, and also b's base SHOULD also be a0, since a0 owns the data.
	 But in these functions, we need to be able to access the dimensions of the most-recent 
	 parent from which b was sliced, so we set 'a' as the base of b (even though 'a' 
	 doesn't own its data).
	*/
	n_array->base = array;
	Py_INCREF(n_array->base);
	
	return n_array;
}
void next_idx(PyArrayObject *sl_r, long axis) {
	// sl_r should have been produced as a 1d slice, e.g. from slice_1d above
	// axis=0 means: sl_r is a row of the 2d base array. next_idx will point to the next row.
	// axis=1 means: sl_r is a column of the 2d base array. next_idx will point to the next column.
	// The incrementing wraps around so one should never be pointing to memory outside the base
	// array's data.
	if (!(sl_r->base)) {
		PyErr_SetString(PyExc_ValueError, "Only a 1d view of a 2d array can be incremented");
	}
	int ndim_base = PyArray_NDIM(sl_r->base);
	npy_intp *dims_base = PyArray_DIMS(sl_r->base);
	npy_intp *strides_base = PyArray_STRIDES(sl_r->base);
	void *base_data = PyArray_DATA(sl_r->base);
	
	npy_intp current_slice = (npy_intp)(PyArray_DATA(sl_r) - PyArray_DATA(sl_r->base)) / 
		strides_base[1-axis];
	npy_intp next_slice = (current_slice + 1) % dims_base[1-axis]; // wrap around if at the end.
	sl_r->data = base_data + next_slice * strides_base[1-axis];
}

void avebox_row(PyObject *args) {
	PyObject *nd_s, *nd_f;
	//printf("\t\t\taveboxrow ---start--- Py_REFCNT(args[0]) = %li\n", Py_REFCNT(PyTuple_GetItem(args, 0)));
	long n;
	if (!PyArg_ParseTuple(args, "O&O&l",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_f,
		&n)) {
		PyErr_SetString(PyExc_ValueError,"Something wrong with inputs unpacking");
	}
	//printf("\t\t\taveboxrow ---postparse--- Py_REFCNT(args[0]) = %li\n", Py_REFCNT(PyTuple_GetItem(args, 0)));
	npy_intp numel = PyArray_SIZE(nd_s);
	npy_intp numelf = PyArray_SIZE(nd_f);
	if (numel != numelf) {
		PyErr_SetString(PyExc_IndexError, "Input and output rows must have the same length");
	}
	npy_float64 *s_el, *f_el; // s_el is the pointer to an element in nd_i, f_el in nd_o
	npy_float64 f_sum = 0.;
	npy_float64 n_dbl = (npy_float64)n;
	npy_intp n_half_floor = (npy_intp)(n/2);
	npy_intp n_half_ceil = n_half_floor + 1;
	
	// Get the sum of the first half-box elements
	for (npy_intp i=0; i<n_half_ceil; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_sum += *s_el;
	}
	
	// Now do the actual filtering
	// First loop covers elements whose indices are less than half the width of the box
	for (npy_intp i=0; i<n_half_floor; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		//*f_el = f_sum / n_dbl;
		*f_el = f_sum / ((npy_float64)(n_half_ceil + i));
		f_sum += *((npy_float64 *)PyArray_GETPTR1(nd_s, i+n_half_ceil));
	}
	
	// Second for loop covers the main array (apart from the end bit)
	for (npy_intp i=n_half_floor; i<(numel-n_half_ceil); i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		*f_el = f_sum / n_dbl;
		f_sum += *((npy_float64 *)PyArray_GETPTR1(nd_s, i+n_half_ceil));
		f_sum -= *((npy_float64 *)PyArray_GETPTR1(nd_s, i-n_half_floor));
	}
	
	// Third for loop covers elements whose indices are closer to the end of the
	// array than half the width of the box
	for (npy_intp i=(numel-n_half_ceil); i<numel; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		//*f_el = f_sum / n_dbl;
		*f_el = f_sum / ((npy_float64)(n_half_ceil + (numel-i-1)));
		f_sum -= *((npy_float64 *)PyArray_GETPTR1(nd_s,i-n_half_floor));
	}
	Py_DECREF(nd_s);
	Py_DECREF(nd_f);
	//printf("\t\t\taveboxrow ---end--- Py_REFCNT(args[0]) = %li\n", Py_REFCNT(PyTuple_GetItem(args, 0)));
	//fflush(stdout);
}

void rowbyrow(void (*f)(PyObject *args), PyObject *nd_i, PyObject *nd_o, long axis, PyObject *optargs) {
	// This function needs to take in the input array (1 or 2d) AND the output array.
	// Also it needs the axis, to know which axis to break off 1d slices (row, by
	// default, i.e. axis=1).  For avebox, n (box size) would go in optargs (a tuple)
	if (axis < -1) {
		PyErr_SetString(PyExc_ValueError, "Optional input 'axis' must be an integer greater than -1");
	}
	int ndim = PyArray_NDIM(nd_i);
	npy_intp *dims = PyArray_DIMS(nd_i);
	Py_ssize_t optarg_length = PyTuple_Size(optargs);
	PyObject *passargs = PyTuple_New(2 + optarg_length);
	for (int i=0; i<optarg_length; i++) {
		PyTuple_SetItem(passargs, i+2, PyTuple_GetItem(optargs,i));
	}
	Py_INCREF(PyTuple_GetItem(optargs, 0)); // needed because the above tuple packing doesn't incref n.
	if (ndim == 1) {
		PyTuple_SetItem(passargs, 0, nd_i);
		PyTuple_SetItem(passargs, 1, nd_o);
		Py_INCREF(nd_i);
		Py_INCREF(nd_o);
		f(passargs);
		Py_DECREF(passargs);
		Py_DECREF(nd_i);
		Py_DECREF(nd_o);
	} else {
		PyArrayObject *sl_i = slice_1d(nd_i, 0L, axis);
		PyArrayObject *sl_o = slice_1d(nd_o, 0L, axis);
		PyTuple_SetItem(passargs, 0, (PyObject *)sl_i);
		PyTuple_SetItem(passargs, 1, (PyObject *)sl_o);
		for (npy_intp i=0; i<dims[1-axis]; i++) {
			f(passargs);
			next_idx(sl_i, axis);
			next_idx(sl_o, axis);
		}
		//Py_DECREF(sl_i);
		//Py_DECREF(sl_o);
		Py_DECREF(passargs);
		Py_DECREF(nd_i);
		Py_DECREF(nd_o);
	}
}

/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_avebox(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "n", "axis", NULL};
	PyArrayObject *nd_s;
	PyObject *n;
	long axis=1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O|l", keywords,
		PyArray_Converter, &nd_s,
		&n,
		&axis)) {
		return NULL;
	}
	if ((PyLong_AsLong(n)%2)==0) {
		PyErr_SetString(PyExc_ValueError, "Input 'n' must be an ODD number");
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	PyObject *optargs = PyTuple_Pack(1, n);
	Py_INCREF(nd_s);
	Py_INCREF(nd_f);
	rowbyrow(avebox_row, (PyObject *)nd_s, nd_f, axis, optargs);
	Py_DECREF(nd_s);
	Py_DECREF(optargs);
	return nd_f;
}

static PyObject *meth_slice1d(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *nd_i;
	long idx;
	long axis = 1L;
	static char *keywords[] = {"", "", "axis", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&l|l", keywords,
		PyArray_Converter, &nd_i,
		&idx,
		&axis)) {
		return NULL;
	}
	PyArrayObject *nd_o = slice_1d((PyObject *)nd_i, idx, axis);
	Py_DECREF(nd_i);
	return (PyObject *)nd_o;
}

static PyObject *meth_next_idx(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *nd_i;
	long axis = 1L;
	static char *keywords[] = {"", "axis", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|l", keywords,
		PyArray_Converter, &nd_i, &axis)) {
		return NULL;
	}
	next_idx(nd_i, axis);
	Py_DECREF(nd_i);
	Py_RETURN_NONE;
}

/*
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
*/
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
	slice1d__doc__,
	"slice1d(arr, idx, axis=1)\n--\n\n"
	"Slice a 1-dimensional array from a 2d array.");
PyDoc_STRVAR(
	next_idx__doc__,
	"next_idx(arr, axis=1)\n--\n\n"
	"Take a 1d slice from a 2d array, and increment which row or column it came from.");

static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{"slice1d", (PyCFunction)meth_slice1d, METH_VARARGS|METH_KEYWORDS, slice1d__doc__},
	{"next_idx", (PyCFunction)meth_next_idx, METH_VARARGS|METH_KEYWORDS, next_idx__doc__},
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
