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

Py_ssize_t py_ssize_t_max(Py_ssize_t a, Py_ssize_t b) {
	if (a > b) {
		return a;
	}
	return b;
}

long get_axis(long axis_in, npy_intp ndim) {
	// C's mod arithmatic works differently than python's.
	// python: -1 % 3 = 2
	//      C: -1 % 3 = -1
	// if axis is -1 and ndim is 3, then axis should be 2.  Easy
	// in python, but requires more steps in C.
	return (long)((ndim + ((npy_intp)axis_in % ndim)) % ndim);
}

void exp_filt_row(PyObject *args) {
	PyObject *nd_s, *nd_f;
	double t0; // the decay constant of the exp filter, in units of samples
	if (!PyArg_ParseTuple(args, "O&O&d",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_f,
		&t0)) {
		PyErr_SetString(PyExc_ValueError, "Something wrong with inputs unpacking");
	}
	npy_intp numel = PyArray_SIZE(nd_s);
	npy_intp numelf = PyArray_SIZE(nd_f);
	if (numel != numelf) {
		PyErr_SetString(PyExc_IndexError, "Input and output rows must have the same length");
	}
	npy_float64 *s_el, *f_el, *f_el_last;
	npy_float64 pre_sum = 0.;
	s_el = (npy_float64 *)PyArray_GETPTR1(nd_s,0);
	f_el = (npy_float64 *)PyArray_GETPTR1(nd_f,0);
	*f_el = *s_el;
	npy_float64 exp_neg1 = exp(-1./t0);
	
	// If the filter starts directly at the beginning of the signal, then it requires
	// a settling-in period; the filter is a weighted sum of all past samples, so if
	// there are not so many past samples, the early values of the filtered signal will
	// be biased by the first few samples of the unfiltered signal.  To mitigate this
	// issue, I basically take the first 2*t0 samples, reflect them about zero, thus
	// padding the signal array for the filter.  The first for loop below does this;
	// I start at the (2*t0)'th sample and work the filter backwards to zero, then begin 
	// forwards again.  'pre-sum' is the filtered value of this padding, which is not
	// a pointer because it doesn't need to be saved anywhere.
	//pre_sum = *((npy_float64 *)PyArray_GETPTR1(nd_s, npy_intp(t0)));
	pre_sum = *((npy_float64 *)PyArray_GETPTR1(nd_s, (npy_intp)(2.*t0)));
	
	for (npy_intp k=(npy_intp)(2*t0)-1; k>=0; k--) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k);
		pre_sum = (*s_el)/t0 + exp_neg1*pre_sum;
	}
	f_el_last = (npy_float64 *)PyArray_GETPTR1(nd_f, 0);
	*f_el_last = pre_sum;
	
	for (npy_intp k=1; k<numel; k++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s,k);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f,k);
		f_el_last = (npy_float64 *)PyArray_GETPTR1(nd_f, k-1);
		*f_el = (*s_el)/t0 + exp_neg1*(*f_el_last);
	}
	Py_DECREF(nd_s);
	Py_DECREF(nd_f);
}

void avebox_row(PyObject *args) {
	// This is a filter function that works on a 1d array.
	PyObject *nd_s, *nd_f;
	//printf("\t\t\taveboxrow ---start--- Py_REFCNT(args[0]) = %li\n", Py_REFCNT(PyTuple_GetItem(args, 0)));
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
}
void find_peaks_row(PyObject *args) {
	PyObject *nd_i, *nd_o;
	double p_thresh;
	long n;
	if (!PyArg_ParseTuple(args, "O&O&ld",
		PyArray_Converter, &nd_i,
		PyArray_Converter, &nd_o,
		&n, &p_thresh)) {
		PyErr_SetString(PyExc_ValueError, "Something went wrong with input unpacking in `find_peaks_row`.");
	}
	// p_thresh is the threshold above which to consider peaks to be found
	// n is the number of peaks to look for
	//
	// This function should look over a trace for peaks and return, for each of n peaks:
	//		0, 50%, 100%, 50%, 0% height fraction times
	//			[0 defined as first sample <10% of peak height, minus 4 samples]
	//		25%, 50%, 75% area fraction times (0, 100% might be omitted because in heights)
	//		pulse area, pulse height.
	//		These are only filled if there was a pulse found above p_thresh. Can
	//		include a boolean (I guess in double format) to indicate a pulse was found or not
	//		Easiest to order the pulses by decreasing height (rather than time-ordered)
	// 
	
	// Here's an idea for handling excluded areas: create an array the same length
	// as the pulse.  It is initialized with zeros, but any index that is excluded
	// replaces the zero with a -10000.  Then any iteration actually looks at sig[i] + excl[i]
	// where 'sig[0]' is the pulse trace and excl[i] is this extra array.
	npy_float64 *el_i, *el_o;
	npy_float64 max_val, min_val, pulse_area, prt_sum; // max_val is pulse height
	npy_intp idx_0hl, idx_50hl, max_el, idx_50hr, idx_0hr; // height fraction times
	npy_intp idx_25a, idx_50a, idx_75a; // area fraction times
	npy_intp ii;
	npy_intp n_el_i = PyArray_SIZE(nd_i);
	npy_intp n_el_o = PyArray_SIZE(nd_o);
	npy_intp num_rqs = n_el_o / n;
	
	// create and initialize exclusion array
	npy_float64 excl_arr[n_el_i];
	for (npy_intp i=0; i<n_el_i; i++) {
		excl_arr[i] = 0.;
	}
	// Need to initialize the other RQs and move to the top, and clean up top-code initializations
	for (npy_intp pk=0; pk<n; pk++) {
		// max_val will hold the maximum value in the event (excluding already-found pulses)
		max_val = -100000.;
		// min_val will hold the minimum value IN THE PULSE (not in the whole window)
		min_val = 100000000.;
		max_el = 0;
		pulse_area = 0.;
		prt_sum = 0.;
		for (npy_intp i=0; i<n_el_i; i++) {
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, i);
			if ((*el_i+excl_arr[i]) > max_val) {
				max_val = *el_i;
				max_el = i;
			}
		}
		// go to pulse max, step to left to 50%
		ii = max_el;
		el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		while (((*el_i)>(0.5*max_val))&&(ii>0)) {
			ii--;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		}
		ii++; // ii is now the 50% idx.
		idx_50hl = ii;
		// continue until 5%
		while (((*el_i)>(0.05*max_val))&&(ii>0)) {
			ii--;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		}
		ii++; // ii is now the 5% idx
		idx_0hl = intp_max(0, ii-5); // set pulse start to be 5 samples before 5%
		
		// go back to pulse max, step to right to 50%
		ii = max_el;
		el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		while (((*el_i)>(0.5*max_val))&&(ii<n_el_i)) {
			ii++;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		}
		ii--; // ii is now the 50% idx.
		idx_50hr = ii;
		// continue until 5%
		while (((*el_i)>(0.05*max_val))&&(ii<n_el_i)) {
			ii++;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i,ii);
		}
		ii--; // ii is now the 5% idx
		idx_0hr = intp_min(n_el_i-1, ii+5); // set pulse start to be 5 samples after 5%
		
		// go to start of pulse and integrate to get total area
		for (ii=idx_0hl; ii<=idx_0hr; ii++) {
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, ii);
			pulse_area += *el_i;
			if (*el_i < min_val) {
				min_val = *el_i;
			}
		}
		// go back to start of pulse and integrate again to get area fraction times
		ii = idx_0hl;
		el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, ii);
		prt_sum = *el_i;
		// integrate to 25% of pulse area
		while ((prt_sum<0.25*pulse_area)&&(ii<=idx_0hr)) {
			ii++;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, ii);
			prt_sum += *el_i;
		}
		idx_25a = ii - 1;
		// continue on to 50% area
		while ((prt_sum<0.5*pulse_area)&&(ii<=idx_0hr)) {
			ii++;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, ii);
			prt_sum += *el_i;
		}
		idx_50a = ii - 1;
		// continue on to 75%
		while ((prt_sum<0.75*pulse_area)&&(ii<=idx_0hr)) {
			ii++;
			el_i = (npy_float64 *)PyArray_GETPTR1(nd_i, ii);
			prt_sum += *el_i;
		}
		idx_75a = ii - 1;
		
		// finally, need to punch out the exclusion array (but only if necessary)
		if ((pk+1)<n) {
			for (ii=idx_0hl; ii<=idx_0hr; ii++) {
				excl_arr[ii] = -100000.;
			}
		}
		// now we will fill the output array... realizing that we'll have to cast all
		// the ints to floats
		// the order will be:
		//      0          1        2        3        4        5         6        7        8        9       10
		//	pulse_area, max_val, min_val, idx_0hl, idx_50hl, max_el, idx_50hr, idx_0hr, idx_25a, idx_50a, idx_75a
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 0+pk*num_rqs);
		*el_o = pulse_area;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 1+pk*num_rqs);
		*el_o = max_val;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 2+pk*num_rqs);
		*el_o = min_val;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 3+pk*num_rqs);
		*el_o = (npy_float64)idx_0hl;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 4+pk*num_rqs);
		*el_o = (npy_float64)idx_50hl;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 5+pk*num_rqs);
		*el_o = (npy_float64)max_el;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 6+pk*num_rqs);
		*el_o = (npy_float64)idx_50hr;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 7+pk*num_rqs);
		*el_o = (npy_float64)idx_0hr;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 8+pk*num_rqs);
		*el_o = (npy_float64)idx_25a;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 9+pk*num_rqs);
		*el_o = (npy_float64)idx_50a;
		el_o = (npy_float64 *)PyArray_GETPTR1(nd_o, 10+pk*num_rqs);
		*el_o = (npy_float64)idx_75a;
	}
	Py_DECREF(nd_i);
	Py_DECREF(nd_o);
}

void rowbyrow_optargs(void (*f)(PyObject *args), PyObject *nd_i, PyObject *nd_o, long axis, PyObject *optargs) {
	int ndim = PyArray_NDIM(nd_i);
	npy_intp *dims = PyArray_DIMS(nd_i);
	Py_ssize_t optarg_length = PyTuple_Size(optargs);
	PyObject *passargs = PyTuple_New(2 + py_ssize_t_max(optarg_length, 0));
	for (int i=0; i<py_ssize_t_max(optarg_length, 0); i++) {
		PyTuple_SetItem(passargs, i+2, PyTuple_GetItem(optargs, i));
	}
	
	long raxis = get_axis(axis, ndim);
	
	if (optarg_length > 0) {
		Py_INCREF(PyTuple_GetItem(optargs, 0)); //needed because the above tuple packing doesn't incref n
	}
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
		PyObject *slice_full = PySlice_New(NULL, NULL, NULL);
		PyObject *slices_1d = PyTuple_New(ndim);
		PyObject *slices_nd = PyTuple_New(ndim);
		for (int k=0; k<ndim; k++) {
			if (k == raxis) {
				PyTuple_SetItem(slices_1d, k, slice_full);
				PyTuple_SetItem(slices_nd, k, PyLong_FromLong(0));
			} else {
				PyTuple_SetItem(slices_1d, k, PyLong_FromLong(0));
				PyTuple_SetItem(slices_nd, k, slice_full);
			}
		}
		
		PyArrayObject *aslice_row_i = (PyArrayObject *)PyObject_GetItem(nd_i, slices_1d);
		PyArrayObject *aslice_mat_i = (PyArrayObject *)PyObject_GetItem(nd_i, slices_nd);
		PyArrayObject *aslice_row_o = (PyArrayObject *)PyObject_GetItem(nd_o, slices_1d);
		PyArrayObject *aslice_mat_o = (PyArrayObject *)PyObject_GetItem(nd_o, slices_nd);
		PyTuple_SetItem(passargs, 0, (PyObject *)aslice_row_i);
		PyTuple_SetItem(passargs, 1, (PyObject *)aslice_row_o);
		
		NpyIter *iter_i, *iter_o;
		NpyIter_IterNextFunc *iternext_i, *iternext_o;
		char **dataptr_i, **dataptr_o;
		
		iter_i = NpyIter_New(aslice_mat_i, NPY_ITER_READONLY,
			NPY_CORDER, NPY_NO_CASTING, NULL);
		iter_o = NpyIter_New(aslice_mat_o, NPY_ITER_READONLY,
			NPY_CORDER, NPY_NO_CASTING, NULL);
		
		iternext_i = NpyIter_GetIterNext(iter_i, NULL);
		iternext_o = NpyIter_GetIterNext(iter_o, NULL);
		if (iternext_i == NULL) {
			NpyIter_Deallocate(iter_i);
		}
		if (iternext_o == NULL) {
			NpyIter_Deallocate(iter_o);
		}
		
		dataptr_i = NpyIter_GetDataPtrArray(iter_i);
		dataptr_o = NpyIter_GetDataPtrArray(iter_o);
		char *data_i, *data_o;
		do {
			data_i = *dataptr_i;
			data_o = *dataptr_o;
			aslice_row_i->data = (void *)data_i;
			aslice_row_o->data = (void *)data_o;
			f(passargs);
			iternext_o(iter_o);
		} while(iternext_i(iter_i));
		
		NpyIter_Deallocate(iter_i);
		NpyIter_Deallocate(iter_o);
		Py_DECREF(passargs);
		Py_DECREF(aslice_mat_i);
		Py_DECREF(aslice_mat_o);
		Py_DECREF(nd_i);
		Py_DECREF(nd_o);
	}
}

void rowbyrow(void (*f)(PyObject *args), PyObject *nd_i, PyObject *nd_o, long axis) {
	// this is an overloaded wrapper for the main function, which requires `optargs`
	PyObject *optargs = PyTuple_New(0);
	rowbyrow_optargs(f, nd_i, nd_o, axis, optargs);
	Py_DECREF(optargs);
}

/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_exp_filt(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "t0", "axis", NULL};
	PyArrayObject *nd_s;
	npy_float64 t0 = 100.; // exp decay constant in units of SAMPLES
	long axis=1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|dl", keywords,
		PyArray_Converter, &nd_s,
		&t0,
		&axis)) {
		return NULL;
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	PyObject *optargs = PyTuple_Pack(1, PyFloat_FromDouble(t0));
	Py_INCREF(nd_s);
	Py_INCREF(nd_f);
	rowbyrow_optargs(exp_filt_row, (PyObject *)nd_s, nd_f, axis, optargs);
	Py_DECREF(nd_s);
	Py_DECREF(optargs);
	return nd_f;
}

void forwardconv_row(PyObject *args) {
	// nd_s is the initial signal
	// nd_f is the filtered signal
	// nd_kern is the kernal that will be used to filter
	PyObject *nd_s, *nd_f, *nd_kern;
	if (!PyArg_ParseTuple(args, "O&O&O&",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_f,
		PyArray_Converter, &nd_kern)) {
		PyErr_SetString(PyExc_ValueError, "Something went wrong with inputs unpacking in forwardfilt_row");
	}
	npy_intp numel = PyArray_SIZE(nd_s);
	npy_intp numelf = PyArray_SIZE(nd_f);
	if (numel != numelf) {
		PyErr_SetString(PyExc_IndexError, "Input and output rows must have the same length");
	}
	npy_intp numel_kern = PyArray_SIZE(nd_kern);
	if (numel_kern >= numel) {
		PyErr_SetString(PyExc_IndexError, "Kernal cannot have more elements than the signal");
	}
	
	npy_float64 *s_el, *f_el, *f_kern; // pointers to elements in the arrays
	npy_float64 f_sum = 0.;
	npy_intp n_left; // this will tell us how many elements there are left in the signal array
	
	for (npy_intp i_s=0; i_s<numel; i_s++) {
		n_left = numel - i_s;
		f_sum = 0.;
		for (npy_intp i_k=0; i_k<intp_min(numel_kern,n_left); i_k++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i_s+i_k);
			f_kern = (npy_float64 *)PyArray_GETPTR1(nd_kern, i_k);
			f_sum += *s_el * *f_kern;
		}
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i_s);
		*f_el = f_sum;
	}
	
	Py_DECREF(nd_s);
	Py_DECREF(nd_f);
	Py_DECREF(nd_kern);
}

static PyObject *meth_avebox(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "n", "axis", NULL};
	PyArrayObject *nd_s;
	//PyObject *n;
	long n=1;
	long axis=1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|ll", keywords,
		PyArray_Converter, &nd_s,
		&n,
		&axis)) {
		return NULL;
	}
	if ((n%2)==0) {
		PyErr_SetString(PyExc_ValueError, "Input 'n' must be an ODD number");
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	PyObject *optargs = PyTuple_Pack(1, PyLong_FromLong(n));
	Py_INCREF(nd_s);
	Py_INCREF(nd_f);
	rowbyrow_optargs(avebox_row, (PyObject *)nd_s, nd_f, axis, optargs);
	Py_DECREF(nd_s);
	Py_DECREF(optargs);
	return nd_f;
}

static PyObject *meth_forwardconv(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "kernel", "axis", NULL};
	PyArrayObject *nd_s, *nd_kern;
	long axis=-1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|l", keywords,
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_kern,
		&axis)) {
			return NULL;
		}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	if (PyArray_TYPE(nd_kern) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_kern' must be of dtype numpy.float64");
	}
	int ndim = PyArray_NDIM(nd_s);
	axis = (axis + ndim) % ndim;
	PyObject *nd_f = PyArray_NewLikeArray(nd_s, NPY_ANYORDER, NULL, 1);
	PyObject *optargs = PyTuple_Pack(1, (PyObject *)nd_kern);
	Py_INCREF(nd_s);
	Py_INCREF(nd_f);
	//Py_INCREF(nd_kern);
	rowbyrow_optargs(forwardconv_row, (PyObject *)nd_s, nd_f, axis, optargs);
	Py_DECREF(nd_s);
	//Py_DECREF(nd_f);
	Py_DECREF(nd_kern);
	Py_DECREF(nd_kern);
	return nd_f;
}

static PyObject *meth_find_peaks(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"", "axis", "n", "thresh", NULL};
	PyArrayObject *nd_i;
	long axis=1L;
	long n=1L;
	double thresh=0.;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|lld", keywords,
		PyArray_Converter, &nd_i, &axis, &n, &thresh)) {
		return NULL;
	}
	if (PyArray_TYPE(nd_i) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array must be of dtype numpy.float64");
	}
	int ndim = PyArray_NDIM(nd_i);
	npy_intp *dims_i = PyArray_DIMS(nd_i);
	npy_intp dims_o[ndim];
	
	long numKeys = 11;
	
	if (ndim == 2) {
		dims_o[0] = dims_i[0];
		dims_o[1] = numKeys * n;
	} else if (ndim == 1) {
		dims_o[0] = numKeys * n;
	} else {
		PyErr_SetString(PyExc_ValueError, "Input array must be 1d or 2d.");
	}
	//dims_o[0] = dims_i[0];
	//dims_o[1] = 2;
	PyObject *nd_o = PyArray_EMPTY(ndim, dims_o, NPY_DOUBLE, NPY_CORDER);
	PyObject *optargs = PyTuple_Pack(2, PyLong_FromLong(n), PyFloat_FromDouble(thresh));
	
	Py_INCREF(nd_i);
	Py_INCREF(nd_o);
	rowbyrow_optargs(find_peaks_row, (PyObject *)nd_i, nd_o, axis, optargs);
	Py_DECREF(nd_i);
	Py_DECREF(optargs);
	
	/* Now parse this large 2d array into a dict with labels
	   The output array has 10*n columns.  Each 10 columns corresponds to the RQs
	   for one found pulse.  For example, if n=2, then:
	   The output array has 20 columns
	   columns 1-10 are the RQs for pulse 1
	   columsn 11-20 are the RQs for pulse 2
	   The current RQs, in order are:
	        Col:  0,  1,  2,   3,    4,    5,    6,   7,   8,   9,  10
	         RQ: pA, pM, pH, h0l, h50l, h100, h50r, h0r, a25, a50, a75
	*/
	//Initiate output dictionary
	PyObject *RQ_dict = PyDict_New();
	
	//Create the dict key names
	const char *keys[numKeys];
	keys[0]  = "pA";		//	Area of the pulse in [adcc*samples] (assuming input was adcc)
	keys[1]  = "pH";		//	Height of the pulse in [adcc]
	keys[2]  = "pM";		//	Minimum of the pulse [in whatever units input was given]
	keys[3]  = "h0_l_samp";	//	Starting sample of the pulse (relative to the start of event window)
	keys[4]  = "h50_l_samp";//	Sample that exceeds 50% of pulse height, on the left side of pulse
	keys[5]  = "h100_samp";	//	Sample where the pulse reaches its maximum value
	keys[6]  = "h50_r_samp";//	Sample that exceeds 50% of pulse height, on the right side of the pulse
	keys[7]  = "h0_r_samp";	//	End sample of the pulse 
	keys[8]  = "a25_samp";	//	Sample where the area of the pulse reaches 25% of its full value
	keys[9]  = "a50_samp";	//	Sample where the area of the pulse reaches 50% of its full value
	keys[10] = "a75_samp";	//	Sample where teh area of the uplse reaches 75% of its full value
	
	PyObject *sliceO;
	PyObject *slice_tuple;
	PyObject *RQ_array;
	int r;
	for (npy_intp k=0; k<numKeys; k++){
		sliceO = PySlice_New(PyLong_FromLong(k), NULL, PyLong_FromLong(numKeys));
		slice_tuple = PyTuple_Pack(2, Py_Ellipsis, sliceO);
		RQ_array = PyArray_Transpose((PyArrayObject *)PyObject_GetItem(nd_o, slice_tuple), NULL);
		r = PyDict_SetItemString(RQ_dict, keys[k], PyArray_Squeeze((PyArrayObject *)RQ_array));
	}
	
	//Py_DECREF(nd_i); // <-- over decrefs the input
	Py_DECREF(RQ_array);
	Py_DECREF(slice_tuple);
	Py_DECREF(sliceO);
	
	return RQ_dict;
}

/* ----------------- </MODULE FUNCTIONS> ----------------- */


PyDoc_STRVAR(
	avebox__doc__,
	"avebox(s_raw, n, axis=-1)\n--\n\n"
	"Apply a box average filter to a signal.\n"
	" s_raw: raw signal.  Numpy array either 1d, or 2d.  If 2d,\n"
	"        the signals will be filtered along axis.\n"
	"     n: The number of samples in the box. n must be an ODD number\n"
	"  axis: If s_raw is 2d, the filtering will occur along this axis.\n"
	"        Default is axis=1, which means along the ROWS.\n"
	"output: The filtered signal.  Will have the same size as s_raw.");
PyDoc_STRVAR(
	exp_filt__doc__,
	"exp_filt(s_raw, t0=100., axis=-1)\n--\n\n"
	"Apply an exponential filter to a signal.\n"
	" s_raw: raw signal.  Numpy array either 1d, or 2d.  If 2d,\n"
	"        the signals will be filtered along axis.\n"
	"    t0: The decay constant of the exponential filter\n"
	"        in units of samples.\n"
	"  axis: If s_raw is 2d, the filtering will occur along this axis.\n"
	"        Default is axis=1, which means along the ROWS.\n"
	"output: The filtered signal.  Will have the same size as s_raw.");

PyDoc_STRVAR(
	forwardconv__doc__,
	"forwardconv(s_raw, s_kern, axis=-1)\n--\n\n"
	"Forward-convolution a kernel signal with an observed waveform. This is\n"
	"useful when you have a template pulse you are searching for in a\n"
	"noisy waveform.  This differs from a convolution, in that a\n"
	"convolution time-reverses the template pulse (i.e. kernel), while a\n"
	"forward-convolution does not.\n"
	"Inputs:\n"
	"    s_raw: The raw waveform.  Must be a 1-d or 2-d numpy array of dtype\n"
	"           float.\n"
	"   s_kern: A 1-d numpy array (of dtype float) that represents the\n"
	"           pulse shape that is being searched for. The kernel must\n"
	"           have fewer elements than the waveform of a single event.\n"
	"     axis: If s_raw is 2-d, this specifies which dimension will be\n"
	"           considered as an event.  Typically, each row is an event,\n"
	"           which means axis should be 1 or -1 (-1 is default, which\n"
	"           just means the last dimension of the array).\n"
	"Outputs:\n"
	"   s_filt: The resulting forward-convolved signal. Same dimensions as\n"
	"           s_raw."
	);

PyDoc_STRVAR(
	find_peaks__doc__,
	"find_peaks(sig_in, axis=-1, n=1, thresh=0.)\n--\n\n"
	"Find n peaks above threshold.  Returns a dict whose keys describe "
	"different reduced quantities of each found pulse.");

static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{"exp_filt",(PyCFunction)meth_exp_filt,METH_VARARGS|METH_KEYWORDS, exp_filt__doc__},
	{"find_peaks",(PyCFunction)meth_find_peaks,METH_VARARGS|METH_KEYWORDS, find_peaks__doc__},
	{"forwardconv",(PyCFunction)meth_forwardconv,METH_VARARGS|METH_KEYWORDS, forwardconv__doc__},
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
