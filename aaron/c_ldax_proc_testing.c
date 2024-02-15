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

void testcopy(PyObject *args) {
	printf("...............JUST COPYING....\n");
	PyObject *nd_s, *nd_f;
	long n;
	if (!PyArg_ParseTuple(args, "O&O&l",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_f,
		&n)) {
		PyErr_SetString(PyExc_ValueError,"Something wrong with inputs unpacking");
	}
	npy_intp numel = PyArray_SIZE(nd_s);
	npy_intp numelf = PyArray_SIZE(nd_f);
	printf("numel = %li\n", numel);
	if (numel != numelf) {
		PyErr_SetString(PyExc_IndexError, "Input and output rows must have the same length");
	}
	npy_float64 *s_el, *f_el; // s_el is the pointer to an element in nd_i, f_el in nd_o
	for (npy_intp i=0; i<numel; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		*f_el = *s_el;
	}
}

void avebox_row(PyObject *args) {
	PyObject *nd_s, *nd_f;
	long n;
	if (!PyArg_ParseTuple(args, "O&O&l",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_f,
		&n)) {
		PyErr_SetString(PyExc_ValueError,"Something wrong with inputs unpacking");
	}
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
		*f_el = f_sum / n_dbl;
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
		*f_el = f_sum / n_dbl;
		f_sum -= *((npy_float64 *)PyArray_GETPTR1(nd_s,i-n_half_floor));
	}
	
	//Py_DECREF(nd_s);
	//Py_DECREF(nd_f);
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
	// Fill in the optional arguments
	for (int i=0; i<optarg_length; i++) {
		PyTuple_SetItem(passargs, i+2, PyTuple_GetItem(optargs,i));
	}
	Py_INCREF(PyTuple_GetItem(optargs, 0)); // needed because the above tuple packing doesn't incref n.
	if (ndim == 1) {
		PyTuple_SetItem(passargs, 0, nd_i);
		PyTuple_SetItem(passargs, 1, nd_o);
		f(passargs);
		Py_DECREF(passargs);
		Py_DECREF(nd_i);
		Py_DECREF(nd_o);
	} else {
		//avebox_row((PyObject *)nd_s, (PyObject *)nd_f, n);
		PyObject *slice_fix = PySlice_New(NULL, NULL, NULL);
		//PyObject *slice_fix = PySlice_New(Py_None, Py_None, Py_None);
		//PyObject **p_slice_fix = &slice_fix;
		PyObject *row_num[1];
		PyObject *slices;
		PyObject *sl_i, *sl_o;
		//slices = PyTuple_Pack(2, PyLong_FromLong(1L), slice_fix);
		//sl_i = PyObject_GetItem(nd_i, slices);
		//npy_intp slSize = PyArray_SIZE(sl_i);
		//printf("slSize = %li\n", slSize);
		PyObject **pp0, **pp1;
		if (axis==1) {
			pp0 = row_num;
			pp1 = &slice_fix;
			//pp1 = p_slice_fix;
		} else if (axis==0) {
			pp0 = &slice_fix;
			//pp0 = p_slice_fix;
			pp1 = row_num;
		} else {
			PyErr_SetString(PyExc_ValueError, "Input 'axis' must be 0 or 1");
		}
		PyObject *p_i = (PyObject *)nd_i;
		PyObject *p_o = (PyObject *)nd_o;
		//PyObject *passargs = 
		for (npy_intp i=0; i<dims[1-axis]; i++) {
			*row_num = PyLong_FromLong(i);
			printf("\t\t*row_num = %p\n", *row_num);
			printf("\t\tPy_REFCNT(*row_num) = %li\n", Py_REFCNT(*row_num));
			printf("\tGetting row: %li\n", PyLong_AsLong(*row_num));
			slices = PyTuple_Pack(2, *pp0, *pp1);
			sl_i = PyObject_GetItem(p_i, slices);
			printf("\tlen(sl_i) = %li\n", PyArray_SIZE(sl_i));
			sl_o = PyObject_GetItem(p_o, slices);
			PyTuple_SetItem(passargs, 0, sl_i);
			PyTuple_SetItem(passargs, 1, sl_o);
			fflush(stdout);
			f(passargs);
			Py_DECREF(*row_num);
			Py_DECREF(slices);
			Py_DECREF(sl_i);
			Py_DECREF(sl_o);
		}
		printf("\t\tRBR ---post f loop--- Py_REFCNT(slice_fix) = %li\n", Py_REFCNT(slice_fix));
		Py_DECREF(passargs);
		printf("\t\tRBR ---post pass dec--- Py_REFCNT(passargs) = %li\n", Py_REFCNT(passargs));
		Py_DECREF(slice_fix);
		printf("\t\tRBR ---post sfix dec--- Py_REFCNT(slice_fix) = %li\n", Py_REFCNT(slice_fix));
		Py_DECREF(nd_i);
		Py_DECREF(nd_o);
		printf("\t\tRBR ---end--- Py_REFCNT(*row_num) = %li\n", Py_REFCNT(*row_num));
		fflush(stdout);
	}
}

/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_avebox_rbr(PyObject *self, PyObject *args, PyObject *kwargs) {
	printf("\tC*** abRBR ---start--- Py_REFCNT(s) = %li\n", Py_REFCNT(PyTuple_GetItem(args,0)));
	fflush(stdout);
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
	printf("\tC*** abRBR ---post parse--- Py_REFCNT(s) = %li\n", Py_REFCNT(nd_s));
	fflush(stdout);
	if ((PyLong_AsLong(n)%2)==0) {
		PyErr_SetString(PyExc_ValueError, "Input 'n' must be an ODD number");
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	printf("\tC*** abRBR ---post sf creation--- Py_REFCNT(sf) = %li\n", Py_REFCNT(nd_f));
	PyObject *optargs = PyTuple_Pack(1, n);
	Py_INCREF(nd_s);
	Py_INCREF(nd_f);
	printf("\tC*** abRBR ---pre RBR pass--- Py_REFCNT(s) = %li\n", Py_REFCNT(nd_s));
	printf("\tC*** abRBR ---pre RBR pass--- Py_REFCNT(sf) = %li\n", Py_REFCNT(nd_f));
	fflush(stdout);
	rowbyrow(avebox_row, (PyObject *)nd_s, nd_f, axis, optargs);
	printf("\tC*** abRBR ---post RBR pass--- Py_REFCNT(s) = %li\n", Py_REFCNT(nd_s));
	printf("\tC*** abRBR ---post RBR pass--- Py_REFCNT(sf) = %li\n", Py_REFCNT(nd_f));
	Py_DECREF(nd_s);
	Py_DECREF(optargs);
	printf("\tC*** abRBR ---end--- Py_REFCNT(s) = %li\n", Py_REFCNT(nd_s));
	printf("\tC*** abRBR ---end--- Py_REFCNT(sf) = %li\n", Py_REFCNT(nd_f));
	fflush(stdout);
	return nd_f;
}
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
	int ndim = PyArray_NDIM(nd_s);
	if (ndim > 2) {
		PyErr_SetString(PyExc_TypeError, "Input raw signal 's_raw' must be 1d or 2d");
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must have a dtype of numpy.float64");
	}
	// nd_f will be the filtered signal array
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	
	PyObject *py_n = PyLong_FromLong(n);
	
	npy_intp *dims = PyArray_DIMS(nd_s);
	PyObject *args_tuple;
	if (ndim==1) {
		Py_INCREF(nd_s);
		Py_INCREF(nd_f);
		//avebox_row((PyObject *)nd_s, (PyObject *)nd_f, n);
		args_tuple = PyTuple_Pack(3,(PyObject *)nd_s, (PyObject *)nd_f, py_n);
		avebox_row(args_tuple);
		Py_DECREF(args_tuple);
	} else {
		PyObject *slice_row = PySlice_New(NULL, NULL, NULL);
		PyObject *row_num, *slices;
		PyObject *sl_s, *sl_f;
		PyObject *p_s = (PyObject *)nd_s;
		PyObject *p_f = (PyObject *)nd_f;
		//loop over rows:
		for (npy_intp i=0; i<dims[1-axis]; i++) {
			row_num = PyLong_FromLong(i);
			if (axis==1) {
				slices = PyTuple_Pack(2, row_num, slice_row);
			} else {
				slices = PyTuple_Pack(2, slice_row, row_num);
			}
			sl_s = PyObject_GetItem(p_s, slices);
			sl_f = PyObject_GetItem(p_f, slices);
			//sl_f = PyObject_GetItem(nd_f, slices);
			Py_INCREF(sl_s);
			Py_INCREF(sl_f);
			//avebox_row(sl_s, sl_f, n);
			args_tuple = PyTuple_Pack(3,sl_s, sl_f, py_n);
			avebox_row(args_tuple);
			Py_DECREF(args_tuple);
			Py_DECREF(row_num);
			Py_DECREF(slices);
			Py_DECREF(sl_s);
			Py_DECREF(sl_f);
		}
		Py_DECREF(slice_row);
	}
	Py_DECREF(nd_s);
	Py_DECREF(py_n);
	return nd_f;
}
/*
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
/*
PyDoc_STRVAR(
	arbfilt__doc__,
	"arbfilt(s_raw, filt_array, axis=1)\n--\n\n"
	"Apply an arbitrary filter shape to the signal.  Same usage as avebox.");
*/
static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{"avebox_rbr",(PyCFunction)meth_avebox_rbr,METH_VARARGS|METH_KEYWORDS, "temp"},
	{NULL, NULL, 0, NULL}
};
/*
	{"arbfilt",(PyCFunction)meth_arbfilt,METH_VARARGS|METH_KEYWORDS, arbfilt__doc__},
	{"findpeaks",(PyCFunction)meth_findpeaks,METH_VARARGS|METH_KEYWORDS,"help documentation..."},
*/
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
