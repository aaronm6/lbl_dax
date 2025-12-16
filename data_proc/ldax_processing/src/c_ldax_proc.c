#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <numpy/npy_math.h>
#include <math.h>

#include "rowbyrow_funcs.h"

/* ----------------- <AUX FUNCTIONS> ----------------- */
PyObject *split_rowlist(PyObject *row_list) {
	// Takes something that was produced following get_pulse_quantities (given a
	// 2d array of waveforms) and produce a dict of numpy arrays.
	// These arrays are intended to be used to construct varrays.
	// For example, we may have 2 waveforms: the first waveform has one pulse
	// of pulse area 100. and pulse height 10.  The second waveform has two pulses
	// of pulse areas (200,220) and pulse heights (20,22).  The output of rowbyrow_list
	// will be: 
	//       [[(100., 10.)], [(200., 20.), (220., 22.)]].
	// We want to turn this into data arrays and a single shape arrays (i.e. the inputs
	// to varray's constructor).
	// pA_darray = np.array([100., 200., 220.])
	// pH_darray = np.array([10., 20., 22.])
	// sarray = np.array([1, 2])
	// This function takes that nested list above and produces a dict of arrays.
	//
	// The pulse quantities provided, and their order, is hard coded here to match
	// what has been produced by get_pulse_quantities_lrow.
	// Currently it is:
	//   pulse_start, pulse_max, pulse_stop, pulse_area, pulse_height);
	
	char *rq_names[] = {"p_start","p_max","p_stop","p_area","p_height","p_bs", NULL};
	int rq_types[] = {NPY_INT64, NPY_INT64, NPY_INT64, NPY_FLOAT64, NPY_FLOAT64, NPY_FLOAT64, -1};
	
	int num_rqs = 0;
	while (rq_names[num_rqs] != NULL) {
		num_rqs++;
	}
	
	Py_ssize_t num_events = PyList_Size(row_list);
	
	// Determine how many pulses there are:
	Py_ssize_t num_pulses = 0;
	//int row_size[num_events];
	int sarray_ndim = 1;
	npy_intp sarray_dims[1];
	sarray_dims[0] = num_events;
	PyObject *sarray = PyArray_EMPTY(sarray_ndim, sarray_dims, NPY_LONG, NPY_CORDER);
	npy_int64 *i_sarray;
	PyObject *evt_list;
	for (int i=0; i<num_events; i++) {
		evt_list = PyList_GetItem(row_list, i);
		num_pulses += PyList_Size(evt_list);
		//row_size[i] = PyList_Size(evt_list);
		i_sarray = (npy_int64 *)PyArray_GETPTR1(sarray, i);
		*i_sarray = PyList_Size(evt_list);
	}
	
	PyObject *out_dict = PyDict_New();
	//int r;
	PyObject *rq_array;
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = num_pulses;
	
	// initialize the npyarrays and put them into the dict
	for (int i=0; i<num_rqs; i++) {
		rq_array = PyArray_EMPTY(ndim, dims, rq_types[i], NPY_CORDER);
		//r = PyDict_SetItemString(out_dict, rq_names[i], rq_array);
		PyDict_SetItemString(out_dict, rq_names[i], rq_array);
		Py_DECREF(rq_array);
	}
	
	Py_ssize_t array_index=0;
	PyObject *pls_tuple; // tuple that will hold the per-pulse info
	PyObject *rq;
	npy_float64 *i_el_float;
	npy_int64 *i_el_int;
	// Fill the arrays.
	// Loop over events
	
	for (Py_ssize_t i_evt=0; i_evt<num_events; i_evt++) {
		// first, get the number of pulses
		i_sarray = (npy_int64 *)PyArray_GETPTR1(sarray, i_evt);
		// get the event list
		evt_list = PyList_GetItem(row_list, i_evt);
		// loop over pulses within the event i_evt
		for (Py_ssize_t i_pls=0; i_pls<*i_sarray; i_pls++) {
			pls_tuple = PyList_GetItem(evt_list, i_pls);
			// now loop over RQ
			for (Py_ssize_t i_rq=0; i_rq<num_rqs; i_rq++) {
				rq = PyTuple_GetItem(pls_tuple, i_rq);
				rq_array = PyDict_GetItemString(out_dict, rq_names[i_rq]);
				if (rq_types[i_rq] == NPY_FLOAT64) {
					i_el_float = (npy_float64 *)PyArray_GETPTR1(rq_array, array_index);
					*i_el_float = PyFloat_AsDouble(rq);
				} else {
					i_el_int = (npy_int64 *)PyArray_GETPTR1(rq_array, array_index);
					*i_el_int = PyLong_AsLong(rq);
				}
			}
			array_index++;
		}
		
	}
	PyDict_SetItemString(out_dict, "sarray", sarray);
	Py_DECREF(sarray);
	Py_DECREF(evt_list);
	return out_dict;
}
/* ----------------- <ROW-BY-ROW FUNCTIONS> ----------------- */

/* ----------------- <ARRAY LROW OPERATIONS> ----------------- */
PyObject *first_last_lrow(PyObject *args) {
	// Just take a 1d array and return a 2-element list object with the first and last el.
	PyArrayObject *nd_i;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &nd_i)) {
		PyErr_SetString(PyExc_ValueError, "Something went wrong with row unpacking");
	}
	npy_intp numel = PyArray_SIZE(nd_i);
	
	npy_float64 *i_el;
	
	PyObject *out_list = PyList_New(2);
	i_el = (npy_float64 *)PyArray_GETPTR1(nd_i, 0);
	PyList_SetItem(out_list, 0, PyFloat_FromDouble(*i_el));
	i_el = (npy_float64 *)PyArray_GETPTR1(nd_i, numel-1);
	PyList_SetItem(out_list, 1, PyFloat_FromDouble(*i_el));
	
	return out_list;
}

PyObject *get_pulse_quantities_lrow(PyObject *args) {
	Py_ssize_t args_len = PyTuple_Size(args);
	
	PyArrayObject *nd_i, *nd_b;
	nd_i = (PyArrayObject *)PyTuple_GetItem(args, 0);
	nd_b = (PyArrayObject *)PyTuple_GetItem(args, 1);
	npy_intp num_samp_baseline_avg = 0L;
	if (args_len > 2) {
		num_samp_baseline_avg = PyLong_AsLong(PyTuple_GetItem(args, 2));
	}
		
	npy_intp num_i = PyArray_SIZE(nd_i);
	npy_intp num_b = PyArray_SIZE(nd_b);
	if (num_i != num_b) {
		PyErr_SetString(PyExc_ValueError, "input data and binary array must have the same length.");
	}
	
	npy_float64 *i_el;
	npy_bool *b_el;
	npy_float64 pulse_area = 0.;
	npy_float64 pulse_height = -100000.;
	npy_float64 pulse_bs = 0.; // holds the baseline average of the pulse
	npy_intp pulse_start, pulse_max, pulse_stop;
	PyObject *pulse_area_list = PyList_New(0);
	PyObject *trace_quantities_list = PyList_New(0);
	PyObject *pulse_quantities;
	// Declare PyObjects that will hold pulse quantites
	PyObject *p_area, *p_height, *p_start, *p_max, *p_stop, *p_bs;
	for (npy_intp k=0; k<num_i; k++) {
		b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
		if (*b_el == NPY_TRUE) {
			pulse_area = 0.;
			pulse_height = -100000.;
			pulse_bs = 0.;
			i_el = (npy_float64 *)PyArray_GETPTR1(nd_i, k);
			
			// loop over the pre-pulse samples (if any) to establish a new baseline
			for (int b=0; b<num_samp_baseline_avg; b++) {
				pulse_bs += *i_el;
				k++;
				i_el = (npy_float64 *)PyArray_GETPTR1(nd_i, k);
			}
			pulse_start = k;
			pulse_bs /= num_samp_baseline_avg;
			b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
			
			// loop over samples of the pulse
			while (*b_el == NPY_TRUE) {
				pulse_area += *i_el - pulse_bs;
				if (*i_el > pulse_height) {
					pulse_height = *i_el - pulse_bs;
					pulse_max = k;
				}
				k++;
				b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
				i_el = (npy_float64 *)PyArray_GETPTR1(nd_i, k);
			}
			pulse_stop = k;
			p_start = PyLong_FromLong(pulse_start);
			p_max = PyLong_FromLong(pulse_max);
			p_stop = PyLong_FromLong(pulse_stop);
			p_area = PyFloat_FromDouble(pulse_area);
			p_height = PyFloat_FromDouble(pulse_height);
			p_bs = PyFloat_FromDouble(pulse_bs);
			pulse_quantities = PyTuple_Pack(6, p_start, p_max, p_stop, p_area, p_height, p_bs);
			Py_DECREF(p_start);
			Py_DECREF(p_max);
			Py_DECREF(p_stop);
			Py_DECREF(p_area);
			Py_DECREF(p_height);
			Py_DECREF(p_bs);
			PyList_Append(trace_quantities_list, pulse_quantities);
			Py_DECREF(pulse_quantities);
		}
	}
	return trace_quantities_list;
}


/* ----------------- <ARRAY ROW OPERATIONS> ----------------- */
void ngdd_filt_mask_row(PyObject *args) {
	// This takes the negative-gauss-double-derivative filter output, applies
	// a threshold, then constructs pulse intervals of time based on that.  It handles
	// overlaps fine, and returns a boolean array that is true where the pulse windows
	// exist.
	// The strategy is to find any point the filtered signal goes above threshold, find
	// the points to the left and right where it then crosses zero.  The left boundary is
	// that left zero-crossing point.  The right boundary is the right zero-crossing
	// point plus a few samples (given by post_samples_add).
	PyObject *nd_s, *nd_b;
	double thresh;
	long pre_samples_add;
	long post_samples_add;
	if (!PyArg_ParseTuple(args, "O&O&dll",
		PyArray_Converter, &nd_s,
		PyArray_Converter, &nd_b,
		&thresh, &pre_samples_add, &post_samples_add)) {
		PyErr_SetString(PyExc_ValueError, "Something went wrong with unpacking ngdd mask");
	}
	npy_intp numel_s = PyArray_SIZE(nd_s);
	npy_intp numel_b = PyArray_SIZE(nd_b);
	
	if (numel_s != numel_b) {
		PyErr_SetString(PyExc_IndexError, "Input and output rows must have the same length");
	}
	
	npy_float64 *s_el;
	npy_bool *b_el;
	npy_intp k_p, k_pre;
	for (npy_intp k=0; k<numel_s; k++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k);
		b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
		if (*s_el >= thresh) {
			k_p = k;
			while ((*s_el > 0.) & (k_p >= 0)) {
				//s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k_p);
				//b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k_p);
				*b_el = NPY_TRUE;
				k_p--;
				s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k_p);
				b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k_p);
			}
			k_pre = 0;
			while ((k_pre<pre_samples_add) & (k_p-k_pre>=0)) {
				b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k_p-k_pre);
				*b_el = NPY_TRUE;
				k_pre++;
			}
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k);
			b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
			while ((*s_el > 0.) & (k < numel_s)) {
				*b_el = NPY_TRUE;
				k++;
				s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k);
				b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
			}
			k_p = 0;
			while ((k_p<post_samples_add) & (k_p+k < numel_s)) {
				b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k+k_p);
				*b_el = NPY_TRUE;
				k_p++;
			}
			k += k_p;
		}
	}
}

void merge_islands_row(PyObject *args) {
	PyArrayObject *nd_b, *nd_v;
	long width; // gap size (or smaller) that should be merged
	if (!PyArg_ParseTuple(args, "O&O&l", 
		PyArray_Converter, &nd_b, 
		PyArray_Converter, &nd_v, 
		&width)) {
		PyErr_SetString(PyExc_ValueError, "Something went wrong unpacking inputs in merge_islands_row");
	}
	npy_intp numel = PyArray_SIZE(nd_b);
	long k_counter = 0;
	npy_bool gate = NPY_FALSE;
	npy_bool *b_el = (npy_bool *)PyArray_GETPTR1(nd_b, 0);
	npy_bool last = *b_el;
	
	for (long k=0; k<numel; k++) {
		b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
		if ((last==NPY_TRUE) & (*b_el == NPY_FALSE)) {
			gate = NPY_TRUE;
		}
		while ((gate==NPY_TRUE) & (k_counter<=(width+1)) & (*b_el==NPY_FALSE) & (k<numel)) {
			b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k);
			k_counter++;
			k++;
		}
		if ((gate==NPY_TRUE) & (k_counter <= (width+1)) & (k<numel)) {
			*b_el = NPY_TRUE;
		}
		while ((gate==NPY_TRUE) & (k_counter <= (width+1)) & (k_counter>=0) & (k<numel)) {
			k_counter--;
			b_el = (npy_bool *)PyArray_GETPTR1(nd_b, k-k_counter-1);
			*b_el = NPY_TRUE;
		}
		gate = NPY_FALSE;
		k_counter = 0;
		last = *b_el;
	}
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

void maxbox_row(PyObject *args) {
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
	//npy_float64 f_sum = 0.;
	npy_float64 f_max = -9999999.;
	npy_float64 n_dbl = (npy_float64)n;
	npy_intp n_half_floor = (npy_intp)(n/2);
	npy_intp n_half_ceil = n_half_floor + 1;
	npy_intp n_window_end;
	
	// Get the max of the first half-box elements
	for (npy_intp i=0; i<n_half_ceil; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		if (*s_el > f_max) { 
			f_max = *s_el;
		}
		//f_sum += *s_el;
	}
	
	// Now do the actual filtering
	// First loop covers elements whose indices are less than half the width of the box
	for (npy_intp i=0; i<n_half_floor; i++) {
		s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, i);
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		//*f_el = f_sum / n_dbl;
		//*f_el = f_sum / ((npy_float64)(n_half_ceil + i));
		*f_el = f_max;
		//f_sum += *((npy_float64 *)PyArray_GETPTR1(nd_s, i+n_half_ceil));
		if (*s_el > f_max) {
			f_max = *s_el;
		}
	}
	
	// Second for loop covers the main array (including the end bit)
	//for (npy_intp i=n_half_floor; i<(numel-n_half_ceil); i++) {
	for (npy_intp i=n_half_floor; i<numel; i++) {
		f_max = -99999.;
		n_window_end = intp_min(i+n_half_ceil, numel);
		//for (npy_intp k=i-n_half_floor; k<i+n_half_ceil; k++) {
		for (npy_intp k=i-n_half_floor; k<n_window_end; k++) {
			s_el = (npy_float64 *)PyArray_GETPTR1(nd_s, k);
			if (*s_el > f_max) {
				f_max = *s_el;
			}
		}
		f_el = (npy_float64 *)PyArray_GETPTR1(nd_f, i);
		*f_el = f_max;
	}
	Py_DECREF(nd_s);
	Py_DECREF(nd_f);
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


/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_get_pulse_quantities(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"","","pulse_bs_avg","axis", NULL};
	PyArrayObject *nd_i, *nd_b;
	long axis = -1;
	long pulse_bs_avg = 0; // Use this many samples at the beginning of each PULSE (not evt) to
	                       // recalculate the baseline average.
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|ll", keywords,
		PyArray_Converter, &nd_i,
		PyArray_Converter, &nd_b,
		&pulse_bs_avg, &axis)) {
		return NULL;
	}
	PyObject *aux_arrays = PyTuple_New(1);
	Py_INCREF(nd_b);
	PyTuple_SetItem(aux_arrays, 0, (PyObject *)nd_b);
	PyObject *optargs = PyTuple_New(1);
	PyTuple_SetItem(optargs, 0, PyLong_FromLong(pulse_bs_avg));
	PyObject *list_out;
	list_out = rowbyrow_list(get_pulse_quantities_lrow, nd_i, axis, aux_arrays, optargs);
	PyObject *dict_out = split_rowlist(list_out);
	Py_DECREF(optargs);
	Py_DECREF(aux_arrays);
	Py_DECREF(nd_i);
	Py_DECREF(nd_b);
	Py_DECREF(list_out);
	//return list_out;
	return dict_out;
}

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

//ngdd_filt_mask_row
static PyObject *meth_ngdd_filt_mask(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"","thresh","pre_samples_add","post_samples_add","axis", NULL};
	PyArrayObject *nd_s;
	npy_double thresh = 8.;
	long pre_samples_add = 0;
	long post_samples_add = 10;
	long axis=-1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|dlll", keywords,
		PyArray_Converter, &nd_s,
		&thresh,
		&pre_samples_add,
		&post_samples_add,
		&axis)) {
		return NULL;
	}
	if (PyArray_TYPE(nd_s) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array 's_raw' must be of dtype numpy.float64");
	}
	//	npy_intp numel_s = PyArray_SIZE(nd_s);
	//npy_intp numel = PyArray_SIZE(nd_s);
	int ndim = PyArray_NDIM(nd_s);
	npy_intp *dims = PyArray_DIMS(nd_s);
	PyObject *nd_b = PyArray_ZEROS(ndim, dims, NPY_BOOL, 0);
	
	PyObject *optargs = PyTuple_Pack(
		3, 
		PyFloat_FromDouble(thresh), 
		PyLong_FromLong(pre_samples_add),
		PyLong_FromLong(post_samples_add));
	Py_INCREF(nd_s);
	Py_INCREF(nd_b);
	rowbyrow_optargs(ngdd_filt_mask_row, (PyObject *)nd_s, nd_b, axis, optargs);
	Py_DECREF(nd_s);
	Py_DECREF(optargs);
	
	return nd_b;
}

static PyObject *meth_merge_islands(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"", "width", "axis", NULL};
	PyArrayObject *nd_b;
	long width=10;
	long axis=-1;
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&|ll", keywords,
		PyArray_Converter, &nd_b,
		&width, &axis)) {
		return NULL;
	}
	if ((long)PyArray_TYPE(nd_b) != NPY_BOOL) {
		PyErr_SetString(PyExc_TypeError, "merge_islands can only accept boolean arrays");
	}
	PyObject *nd_x = PyArray_NewLikeArray(nd_b, NPY_ANYORDER, NULL, 0);
	PyObject *optargs = PyTuple_Pack(1, PyLong_FromLong(width));
	Py_INCREF(nd_b);
	Py_INCREF(nd_x);
	rowbyrow_optargs(merge_islands_row, (PyObject *)nd_b, nd_x, axis, optargs);
	Py_DECREF(nd_b);
	Py_DECREF(nd_x);
	Py_DECREF(optargs);
	
	Py_RETURN_NONE;
}

static PyObject *meth_avebox(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "n", "axis", NULL};
	PyArrayObject *nd_s;
	//PyObject *n;
	long n=1;
	long axis=-1;
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

static PyObject *meth_maxbox(PyObject *self, PyObject *args, PyObject *kwargs) {
	static char *keywords[] = {"signal", "n", "axis", NULL};
	PyArrayObject *nd_s;
	//PyObject *n;
	long n=1;
	long axis=-1;
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
	rowbyrow_optargs(maxbox_row, (PyObject *)nd_s, nd_f, axis, optargs);
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
	long axis=-1;
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
	//int r;
	for (npy_intp k=0; k<numKeys; k++){
		sliceO = PySlice_New(PyLong_FromLong(k), NULL, PyLong_FromLong(numKeys));
		slice_tuple = PyTuple_Pack(2, Py_Ellipsis, sliceO);
		RQ_array = PyArray_Transpose((PyArrayObject *)PyObject_GetItem(nd_o, slice_tuple), NULL);
		//r = PyDict_SetItemString(RQ_dict, keys[k], PyArray_Squeeze((PyArrayObject *)RQ_array));
		PyDict_SetItemString(RQ_dict, keys[k], PyArray_Squeeze((PyArrayObject *)RQ_array));
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
	maxbox__doc__,
	"maxbox(s_raw, n, axis=-1)\n--\n\n"
	"Apply a box maximum filter to a signal.\n"
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

PyDoc_STRVAR(
	ngdd_filt__doc__,
	"ngdd_filt_mask(s_in, thresh=8., pre_samples_add=0, post_samples_add=10, axis=-1)\n--\n\n"
	"Take the filtered result of a multidimensional array of raw data and\n"
	"return a boolean array (of the same dimensions) that is True where a\n"
	"sample is within a pulse window and False otherwise.\n"
	"Inputs:\n"
	"  s_in: filtered data in.\n"
	"  thresh: threshold over which to find pulses\n"
	"  pre_samples_add: beginning of the pulse is extended by this many samples\n"
	"  post_samples_add: the tail of the pulse is extended by this many samples\n"
	"  axis: axis of s_in that constitutes individual channels and events.");

PyDoc_STRVAR(
	merge_islands__doc__,
	"merge_islands(b_in, width=10, axis=-1)\n--\n\n"
	"An 'island' is a series of Trues in a boolean area surrounded by Falses.\n"
	"If two islands are separated by a less than a certain amount, then the\n"
	"Falses between them are set to true, i.e. the islands are merged.\n"
	"Inputs:\n"
	"b_in: boolean array, multidimensional\n"
	"width: (int) If two islands are separated by this number of elements or\n"
	"       fewer, then the two islands will be merged.\n"
	"axis: (int) The axis of b_in that is considered a row (default: last axis\n"
	"\n"
	"Returns: nothing (acts on the input boolean array in place)");

PyDoc_STRVAR(
	get_pqs__doc,
	"get_pulse_quantities(a, b, pulse_bs_avg=0, axis=-1)\n--\n\n"
	"Get pulse quantities (e.g. area, height, etc.)\n"
	"Inputs:\n"
	"            a: Numpy array of dtype('float64') of raw waveforms.  Can be\n"
	"               either a 1d (single) waveform) or 2d, where every\n"
	"               individual waveform is along the axis given by the 'axis'\n"
	"               keyword\n"
	"            b: Numpy array of dtype('bool'), the same shape as 'a', that\n"
	"               specifies the regions of pulses that have been found in\n"
	"               input 'a'.  That is, in an individual waveform, 'b' will\n"
	"               be mostly False, but a series of contiguous Trues\n"
	"               indicates a pulse was found.  There may be several such\n"
	"               regions (or none) in a given waveform.  This boolean array\n"
	"               was probably made with function 'ngdd_filt_mask' in this\n"
	"               module.\n"
	" pulse_bs_avg: int, default=0.  The number of samples at the start of\n"
	"               every pulse that is considered baseline, and is used to \n"
	"               recalculate the baseline average for that individual pulse\n"
	"               This must the same as what was given as the\n"
	"               'pre_samples_add' keyword to 'ngdd_filt_mask'\n"
	"         axis: int, default=-1.  The axis of the array along which\n"
	"               individual waveforms run.  axis=-1 (the default) means the\n"
	"               last dimension.");

static PyMethodDef ldax_methods[] = {
	{"avebox", (PyCFunction)meth_avebox,METH_VARARGS|METH_KEYWORDS, avebox__doc__},
	{"maxbox", (PyCFunction)meth_maxbox,METH_VARARGS|METH_KEYWORDS, maxbox__doc__},
	{"exp_filt",(PyCFunction)meth_exp_filt,METH_VARARGS|METH_KEYWORDS, exp_filt__doc__},
	{"find_peaks",(PyCFunction)meth_find_peaks,METH_VARARGS|METH_KEYWORDS, find_peaks__doc__},
	{"forwardconv",(PyCFunction)meth_forwardconv,METH_VARARGS|METH_KEYWORDS, forwardconv__doc__},
	{"ngdd_filt_mask",(PyCFunction)meth_ngdd_filt_mask, METH_VARARGS|METH_KEYWORDS, ngdd_filt__doc__},
	{"merge_islands",(PyCFunction)meth_merge_islands, METH_VARARGS|METH_KEYWORDS, merge_islands__doc__},
	{"get_pulse_quantities",(PyCFunction)meth_get_pulse_quantities,METH_VARARGS|METH_KEYWORDS, get_pqs__doc},
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
