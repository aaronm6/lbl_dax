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
/* ----------------- </AUX FUNCTIONS> ----------------- */

/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_condense_intervals(PyObject *self, PyObject *args) {
	/*
	condense_intervals(starts_stops, min_dist)
	Inputs:
		starts_stops: 2xN array, where each column represents an interval,
		              and all intervals are assumed to be sequential. The
		              first row are the starting times or indices of the
		              intervals, the second row are the stopping times or 
		              indicies of the intervals.  Here this array must be
		              of dtype int64 (long in C)
		    min_dist: The minimum difference between the stop of one 
		              interval and the start of the next.  If this 
		              difference is less than min_dist, then the two 
		              intervals will be condensed into one. 
		              min_dist must be a scalar of type [python]int, which
		              is C long.
	Output:
		cond_str_stp: A copy of starts_stops input but with nearby 
		              intervals condensed.
	*/
	//printf("CCCCCCCCC:  entering condense_intervals\n");
	PyArrayObject *nd_stsp;
	long md;
	if (!PyArg_ParseTuple(args, "O&l",
			PyArray_Converter, &nd_stsp,
			&md)) {
		return NULL;
	}
	npy_intp nel_stsp = PyArray_SIZE(nd_stsp);
	int ndim_stsp = PyArray_NDIM(nd_stsp);
	npy_intp *dims_stsp = PyArray_DIMS(nd_stsp);
	long num_intvl = (long)(dims_stsp[1]);
	long *stsp = (long *)PyArray_DATA(nd_stsp);
	
	int stsp_c_con = PyArray_IS_C_CONTIGUOUS(nd_stsp);
	if (stsp_c_con!=1) {
		printf("Input array must be C-contiguous\n");
		return NULL;
	}
	
	/*
	ar_temp is going to hold the new start stops, but for convenience it will not be
	a pyarray with dimensions the same as before.  Elements will alternate between
	starts and stops.  That is, its format is:
	[start0,stop0,start1,stop1,start2,stop2,...], where these are the starts and stops
	of the new, condensed intervals.
	*/
	long ar_temp[nel_stsp];
	long ic = 0L;
	ar_temp[0] = stsp[0];
	//int array_ind(int n, int m, npy_intp *dims,int C_con) {
	for (long i=1L; i<num_intvl; i++) {
		if ((stsp[i] - stsp[i-1L+num_intvl]) > md) {
			ar_temp[ic + 1] = stsp[i-1L+num_intvl];
			ic += 2L;
			ar_temp[ic] = stsp[i];
		}
	}
	ar_temp[ic+1] = stsp[2*num_intvl-1L];
	ic += 2L;
	long num_cintvls = ic/2;
	int ndim_out = 2;
	npy_intp dims_out[ndim_out];
	dims_out[0] = 2;
	dims_out[1] = num_cintvls;
	int ft = NPY_CORDER;
	int tp = NPY_LONG;
	PyObject *nd_out = PyArray_EMPTY(ndim_out, dims_out, tp, ft);
	long *out = (long *)PyArray_DATA(nd_out);
	for (long i=0L;i<num_cintvls;i++) {
		out[i] = ar_temp[2*i];
		out[i + num_cintvls] = ar_temp[2*i+1L];
	}
	//printf("                        <<<<CC about to decref\n");
	Py_DECREF(nd_stsp);
	//printf("CCCCC: GOT TO THE END OF CONDENSE INTERVALS\n");
	return nd_out;
}

static PyObject *meth_findpeaks(PyObject *self, PyObject *args) {
	/*
	Forget about time array
	Inputs: 
		d: ndarray (1D) of the smoothed voltage data; dtype: 12(i.e. float64)
		pod_starts: ndarray (1D) of the start samples; dtype: 7(i.e. int64)
		pod_stops:  ndarray (1D) of the stop samples (must be of the same size and dtype as pod_starts)
	Output:
		pulsepara: 3xN (2D) array: first row is position of peak starts
		                            second row is position of peak maxs
		                            third row is position of peak ends
		                            dtype: 7(i.e. int64)
	*/
	
	PyArrayObject *d, *pd_starts, *pd_stops;
	
	if (!PyArg_ParseTuple(args, "O&O&O&", 
			PyArray_Converter, &d, 
			PyArray_Converter, &pd_starts,
			PyArray_Converter, &pd_stops)) {
		return NULL;
	}
	
	npy_intp num_samps = PyArray_SIZE(d);
	npy_intp num_pods = PyArray_SIZE(pd_starts);
	int ndim = 2;
	npy_intp dims[ndim];
	dims[0] = 3;
	dims[1] = num_pods;
	
	PyObject *pulsepara = PyArray_ZEROS(ndim, dims, NPY_LONG, NPY_CORDER);
	double *d_ptr = (double *)PyArray_DATA(d);
	long *pd_starts_ptr = (long *)PyArray_DATA(pd_starts);
	long *pd_stops_ptr = (long *)PyArray_DATA(pd_stops);
	long *pulsepara_ptr = (long *)PyArray_DATA(pulsepara);
	
	double max_val; // = -1000.;
	long max_pos,pbegin, pend, pod_len, i_w;
	// loops over pods 
	for (npy_intp ip=0; ip<num_pods; ip++){
		max_val = -1000.;
		// first, find the max value and its position 
		pod_len = pd_stops_ptr[ip] - pd_starts_ptr[ip] + 1;
		//if (pd_stops_ptr[ip] >= num_samps) {
		//	printf("******>>>>>>>>>>>>> PROBLEM HERE HERE HERE <<<<<<<<<<<<<<<<<<***********\n");
		//}
		for (long i=pd_starts_ptr[ip];i<pd_stops_ptr[ip];i++) {
			if (d_ptr[i] > max_val) {
				max_val = d_ptr[i];
				max_pos = i;
			}
		}
		// Now start at the max and step left until reaching 5% of max 
		i_w = max_pos;
		while ((d_ptr[i_w] > (0.05*max_val))&&(i_w>=0)) {
			i_w--;
		}
		pbegin = i_w;
		// Now start at the max again and step right until reaching 5% of max
		i_w = max_pos;
		while ((d_ptr[i_w] > (0.05*max_val))&&(i_w<(long)num_samps)) {
			i_w++;
		}
		pend = i_w;
		// Now fill the pulsepara array 
		pulsepara_ptr[ip] = pbegin;
		pulsepara_ptr[ip+num_pods] = max_pos;
		pulsepara_ptr[ip+2*num_pods] = pend;
	}
	Py_DECREF(d);
	Py_DECREF(pd_starts);
	Py_DECREF(pd_stops);
	return pulsepara;
}


static PyObject *meth_highpass(PyObject *self, PyObject *args) {
	PyArrayObject *nd_t, *nd_v;
	double f_c;
	if (!PyArg_ParseTuple(args, "O&O&d", 
		PyArray_Converter, &nd_t, 
		PyArray_Converter, &nd_v,
		&f_c)) {
		return NULL;
	}
	
	int t_ndim = PyArray_NDIM(nd_t);
	int v_ndim = PyArray_NDIM(nd_v);
	if ((t_ndim != 1) || (v_ndim != 1)) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must be 1-dimensional");
		return NULL;
	}
	int t_dtype = PyArray_TYPE(nd_t);
	int v_dtype = PyArray_TYPE(nd_v);
	if ((t_dtype != NPY_FLOAT64) || (v_dtype != NPY_FLOAT64)) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must have dtype of np.float64");
	}
	
	npy_intp t_nel = PyArray_SIZE(nd_t);
	npy_intp v_nel = PyArray_SIZE(nd_v);
	if (t_nel != v_nel) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must be of equal length.");
	}
	
	npy_intp *t_st_bytes = PyArray_STRIDES(nd_t);
	npy_intp *v_st_bytes = PyArray_STRIDES(nd_v);
	
	npy_intp el_bytes = PyArray_ITEMSIZE(nd_t); // should be the same as 'sizeof(npy_double)'
	
	npy_intp t_st = *t_st_bytes / el_bytes; // t_st is the strides in units of array elements, not bytes
	npy_intp v_st = *v_st_bytes / el_bytes;
	
	PyObject *nd_v_out = PyArray_NewLikeArray(nd_v, NPY_CORDER, NULL, 1);
	npy_intp *vo_st_bytes = PyArray_STRIDES(nd_v_out); // shouldn't be necessary but doing this just to be careful
	npy_intp vo_st = *vo_st_bytes / el_bytes;
	
	npy_double *t = (npy_double *)PyArray_DATA(nd_t);
	npy_double *v = (npy_double *)PyArray_DATA(nd_v);
	npy_double *v_out = (npy_double *)PyArray_DATA(nd_v_out);
	npy_double dt = t[1] - t[0];
	npy_double alph = 1 / (2 * NPY_PI * dt * f_c + 1);
	
	v_out[0] = v[0];
	for (npy_intp i=1;i<t_nel;i++) {
		v_out[i*vo_st] = alph * (v_out[(i-1)*vo_st] + v[i*v_st] - v[(i-1)*v_st]);
	}
	Py_DECREF(nd_t);
	Py_DECREF(nd_v);
	return nd_v_out;
}

static PyObject *meth_lowpass(PyObject *self, PyObject *args) {
	PyArrayObject *nd_t, *nd_v;
	double f_c;
	if (!PyArg_ParseTuple(args, "O&O&d", 
		PyArray_Converter, &nd_t, 
		PyArray_Converter, &nd_v,
		&f_c)) {
		return NULL;
	}
	
	int t_ndim = PyArray_NDIM(nd_t);
	int v_ndim = PyArray_NDIM(nd_v);
	if ((t_ndim != 1) || (v_ndim != 1)) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must be 1-dimensional");
		return NULL;
	}
	int t_dtype = PyArray_TYPE(nd_t);
	int v_dtype = PyArray_TYPE(nd_v);
	if ((t_dtype != NPY_FLOAT64) || (v_dtype != NPY_FLOAT64)) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must have dtype of np.float64");
	}
	
	npy_intp t_nel = PyArray_SIZE(nd_t);
	npy_intp v_nel = PyArray_SIZE(nd_v);
	if (t_nel != v_nel) {
		PyErr_SetString(PyExc_TypeError, "Input arrays must be of equal length.");
	}
	
	npy_intp *t_st_bytes = PyArray_STRIDES(nd_t);
	npy_intp *v_st_bytes = PyArray_STRIDES(nd_v);
	
	npy_intp el_bytes = PyArray_ITEMSIZE(nd_t); // should be the same as 'sizeof(npy_double)'
	
	npy_intp t_st = *t_st_bytes / el_bytes; // t_st is the strides in units of array elements, not bytes
	npy_intp v_st = *v_st_bytes / el_bytes;
	
	PyObject *nd_v_out = PyArray_NewLikeArray(nd_v, NPY_CORDER, NULL, 1);
	npy_intp *vo_st_bytes = PyArray_STRIDES(nd_v_out); // shouldn't be necessary but doing this just to be careful
	npy_intp vo_st = *vo_st_bytes / el_bytes;
	
	npy_double *t = (npy_double *)PyArray_DATA(nd_t);
	npy_double *v = (npy_double *)PyArray_DATA(nd_v);
	npy_double *v_out = (npy_double *)PyArray_DATA(nd_v_out);
	npy_double dt = t[1] - t[0];
	npy_double RC_const = 2 * NPY_PI * dt * f_c;
	npy_double alph = RC_const / (RC_const + 1.);
	
	v_out[0] = v[0];
	for (npy_intp i=1;i<t_nel;i++) {
		v_out[i*vo_st] = v_out[(i-1)*vo_st] + alph*(v[i*v_st] - v_out[(i-1)*vo_st]);
	}
	Py_DECREF(nd_t);
	Py_DECREF(nd_v);
	return nd_v_out;
}

static PyObject *meth_avebox(PyObject *self, PyObject *args) {
	// avebox(input_array_double, number_of_iterations_longint) -> filtered_array_double
	PyArrayObject *nd_d;
	long n;
	if (!PyArg_ParseTuple(args, "O&l", PyArray_Converter, &nd_d, &n)) {
		return NULL;
	}
	PyObject *nd_d_out = PyArray_NewLikeArray(nd_d, NPY_CORDER, NULL, 1);
	npy_double *d     = (npy_double *)PyArray_DATA(nd_d);
	npy_double *d_out = (npy_double *)PyArray_DATA(nd_d_out);
	npy_intp numEl = PyArray_SIZE(nd_d);
	npy_double f_sum = 0.;
	npy_intp n_half_floor = (npy_intp)(n/2);
	npy_intp n_half_ceil = n_half_floor + 1;
	for (npy_intp i=0;i<n_half_ceil;i++) {
		f_sum += d[i];
	}
	npy_double n_double = (npy_double)n;
	//Breaking up the filtering into three for loops. It can be done with a single for loop but
	//requires a bunch of if statements each iteration.  Breaking it up allows these extra 
	//each-iteration steps to be removed, thereby speeding up the execution.
	
	// First for loop covers the elements whose indices are less than half the width of the box
	for (npy_intp i=0; i<n_half_floor;i++) {
		d_out[i] = f_sum / n_double;
		f_sum += d[i+n_half_ceil];
	}
	// Second for loop covers the main array
	for (npy_intp i=n_half_floor; i<(numEl-n_half_ceil); i++) {
		d_out[i] = f_sum / n_double;
		f_sum += d[i+n_half_ceil];
		f_sum -= d[i-n_half_floor];
	}
	// Third for loop covers the elements whos elements are close to the end of the array than
	// half the width of the box
	for (npy_intp i=(numEl-n_half_ceil); i<numEl; i++) {
		d_out[i] = f_sum / n_double;
		f_sum -= d[i-n_half_floor];
	}
	Py_DECREF(nd_d);
	return nd_d_out;
}

static PyObject *meth_S2filter(PyObject *self, PyObject *args, PyObject *kwargs) {
	PyArrayObject *nd_d;
	long n1 = 300/2;
	long n2 = 1500/2;
	static char *keywords[] = {"","n1","n2", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, kwargs, "O&|ll", keywords, PyArray_Converter, &nd_d, &n1, &n2)) {
		return NULL;
	}
	/*if (!PyArg_ParseTuple(args, "O&ll", PyArray_Converter, &nd_d, &n1, &n2)) {
		return NULL;
	}*/
	// Create the new arrays
	PyObject *nd_d_out = PyArray_NewLikeArray(nd_d, NPY_CORDER, NULL, 1);
	PyObject *nd_d_s1  = PyArray_NewLikeArray(nd_d, NPY_CORDER, NULL, 1);
	PyObject *nd_d_s2  = PyArray_NewLikeArray(nd_d, NPY_CORDER, NULL, 1);
	
	// Grab the pointers to the data of the arrays
	double *d     = (double *)PyArray_DATA(nd_d);
	double *d_s1  = (double *)PyArray_DATA(nd_d_s1);
	double *d_s2  = (double *)PyArray_DATA(nd_d_s2);
	double *d_out = (double *)PyArray_DATA(nd_d_out);
	
	npy_intp numEl = PyArray_SIZE(nd_d);
	// First create the S1-size and S2-size box-filtered signals.
	// S1 ~ 300 ns, S2 ~ 1.5 us
	// d is in samples (horizontal) and 1 sample = 2 ns
	//npy_intp n1 = 300 / 2;
	if ((n1 % 2) == 0) {
		n1 += 1;
	}
	//npy_intp n2 = 1500 / 2;
	if ((n2 % 2) == 0) {
		n2 += 1;
	}
	
	// ------------------ S1-sized box filter ----------------//
	double f_sum = 0.;
	npy_intp n1_half_floor = (npy_intp)(n1/2);
	npy_intp n1_half_ceil = n1_half_floor + 1;
	for (npy_intp i=0;i<n1_half_ceil;i++) {
		f_sum += d[i];
	}
	//Breaking up the filtering into three for loops. It can be done with a single for loop but
	//requires a bunch of if statements each iteration.  Breaking it up allows these extra 
	//each-iteration steps to be removed, thereby speeding up the execution.
	
	// First for loop covers the elements whose indices are less than half the width of the box
	for (npy_intp i=0; i<n1_half_floor;i++) {
		d_s1[i] = f_sum;
		f_sum += d[i+n1_half_ceil];
	}
	// Second for loop covers the main array
	for (npy_intp i=n1_half_floor; i<(numEl-n1_half_ceil); i++) {
		d_s1[i] = f_sum;
		f_sum += d[i+n1_half_ceil];
		f_sum -= d[i-n1_half_floor];
	}
	// Third for loop covers the elements whos elements are close to the end of the array than
	// half the width of the box
	for (npy_intp i=(numEl-n1_half_ceil); i<numEl; i++) {
		d_s1[i] = f_sum;
		f_sum -= d[i-n1_half_floor];
	}
	// ------------------ /S1-sized box filter ----------------//
	
	// ------------------ S2-sized box filter ----------------//
	f_sum = 0.;
	npy_intp n2_half_floor = (npy_intp)(n2/2);
	npy_intp n2_half_ceil = n2_half_floor + 1;
	for (npy_intp i=0;i<n2_half_ceil;i++) {
		f_sum += d[i];
	}
	//Breaking up the filtering into three for loops. It can be done with a single for loop but
	//requires a bunch of if statements each iteration.  Breaking it up allows these extra 
	//each-iteration steps to be removed, thereby speeding up the execution.
	
	// First for loop covers the elements whose indices are less than half the width of the box
	for (npy_intp i=0; i<n2_half_floor;i++) {
		d_s2[i] = f_sum;
		f_sum += d[i+n2_half_ceil];
	}
	// Second for loop covers the main array
	for (npy_intp i=n2_half_floor; i<(numEl-n2_half_ceil); i++) {
		d_s2[i] = f_sum;
		f_sum += d[i+n2_half_ceil];
		f_sum -= d[i-n2_half_floor];
	}
	// Third for loop covers the elements whos elements are close to the end of the array than
	// half the width of the box
	for (npy_intp i=(numEl-n2_half_ceil); i<numEl; i++) {
		d_s2[i] = f_sum;
		f_sum -= d[i-n2_half_floor];
	}
	// ------------------ /S2-sized box filter ----------------//

	// ------------------ Find S1-max for each S2 box ----------------//
	double s1_max;
	npy_intp jmin, jmax;
	for (npy_intp i=0; i<numEl; i++) {
		s1_max = -9999.;
		jmin = intp_max(i-n2_half_floor+n1_half_floor,0);
		jmax = intp_min(i+n2_half_ceil-n1_half_floor, numEl);
		for (npy_intp im=jmin; im<jmax; im++) {
			if (d_s1[im] > s1_max) {
				s1_max = d_s1[im];
			}
		}
		d_out[i] = d_s2[i] - s1_max;
	}
	// ------------------ /Find S1-max for each S2 box ----------------//
	
	//npy_intp intp_max(npy_intp a, npy_intp b) {
	//Relinquish the reference for the arrays that are not returned by the function.
	Py_DECREF(nd_d);
	Py_DECREF(nd_d_s1);
	Py_DECREF(nd_d_s2);
	return nd_d_out;
}


static PyObject *meth_arbfilt(PyObject *self, PyObject *args) {
	PyArrayObject *nd_d, *nd_filt;
	if (!PyArg_ParseTuple(args, "O&O&",
		PyArray_Converter, &nd_d,
		PyArray_Converter, &nd_filt)) {
		return NULL;
	}
	PyObject *nd_d_out = PyArray_NewLikeArray(nd_d, NPY_CORDER, NULL, 1);
	double *d     = (double *)PyArray_DATA(nd_d);
	double *filt  = (double *)PyArray_DATA(nd_filt);
	double *d_out = (double *)PyArray_DATA(nd_d_out);
	npy_intp numEl = PyArray_SIZE(nd_d);
	npy_intp numFilt = PyArray_SIZE(nd_filt);
	double f_sum;
	npy_intp i_max, i_min;
	/*
	for (npy_intp i0=0;i0<numEl;i0++) {
		f_sum = 0.;
		i_max = intp_min(i0+numFilt, numEl) - i0;
		for (npy_intp i=0;i<i_max;i++) {
			f_sum += filt[numFilt-1-i] * d[i+i0];
		}
		d_out[i0] = f_sum;
	}
	*/
	
	npy_intp i_mid = numFilt/2;
	for (npy_intp i0=0; i0<numEl; i0++) {
		f_sum = 0.;
		i_min = intp_max(i0-i_mid, 0) - i0 + i_mid;
		i_max = intp_min(i0+numFilt, numEl) - i0;
		for (npy_intp i=i_min; i<i_max; i++) {
			f_sum += filt[numFilt-1-i] * d[i + i0 - i_mid];
		}
		d_out[i0] = f_sum;
	}
	Py_DECREF(nd_d);
	Py_DECREF(nd_filt);
	return nd_d_out;
}
/*
i0 = 50
i_mid = 4
i0 - i_mid = 46 (>0)
max(i0-i_mid,0) - i0 + i_mid = 46 - 50 + 4 = 0

i0 = 6
i_mid = 10
i0 - i_mid = -4 < 0
max(i0-i_mid,0) - i0 + i_mid = 0 - 6 + 10 = 4

              o o o o o o o o o o o o o o 
      * * * * * * * * * * ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 

i0 = 1
i_mid = 10
i0 - i_mid = -9 < 0
max(i0-i_mid,0) - i0 + i_mid = 0 - 1 + 10 = 9
                        o o o o o o o o o o o o o o 
      * * * * * * * * * * ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ 
*/
static PyObject *meth_get_pulseRQs(PyObject *self, PyObject *args) {
	/*
	get_pulseRQs(d_raw, ch_a, pls_strstp)
	d_raw is the full trace; should be dtype=double(float64) of raw waveform data (baseline subtracted)
	ch_a is the full trace that has been box-average filtered
	pls_strstop is the 2xN array of pulse starts (1st row) and stops (2nd row)
		should be of type numpy.int64 or long in C
	*/
	
	// First, get input arrays
	PyArrayObject *nd_d, *nd_ch_a, *nd_ch_h, *nd_stsp;
	if (!PyArg_ParseTuple(args, "O&O&O&", 
			PyArray_Converter, &nd_d, 
			PyArray_Converter, &nd_ch_a, 
			PyArray_Converter, &nd_stsp)) {
		return NULL;
	}
	
	// Now unpack the input arrays
	npy_intp num_samps = PyArray_SIZE(nd_d);
	double *d    = (double *)PyArray_DATA(nd_d);
	double *ch_a = (double *)PyArray_DATA(nd_ch_a);
	double *ch_h = (double *)PyArray_DATA(nd_ch_h);
	//long *stsp   = PyArray_DATA(nd_stsp);
	long *stsp   = (long *)PyArray_DATA(nd_stsp);
	int ndim_stsp = PyArray_NDIM(nd_stsp);
	npy_intp *dims_stsp = PyArray_DIMS(nd_stsp);
	npy_intp num_pulses = dims_stsp[1];
	
	// Now initialize output arrays and get pointers to their data
	int ft = 0; // say no to Fortran memordering
	int tp_l = NPY_LONG;  // type code for numpy.int64    (i.e. long)
	int tp_d = NPY_DOUBLE; // type code for numpy.float64  (i.e. double)
	int ndim = 1;
	npy_intp dims[ndim];
	dims[0] = num_pulses;
	
	PyObject *nd_pA_raw   = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_pA_av    = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_pA_av_fw = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_pH_raw   = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_pH_av    = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_pM       = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_aft_10   = PyArray_EMPTY(ndim, dims, tp_l, ft);
	PyObject *nd_aft_50   = PyArray_EMPTY(ndim, dims, tp_l, ft);
	PyObject *nd_aft_90   = PyArray_EMPTY(ndim, dims, tp_l, ft);
	PyObject *nd_hft_10l  = PyArray_EMPTY(ndim, dims, tp_l, ft); // time when height is 10% of max on left
	PyObject *nd_t_max    = PyArray_EMPTY(ndim, dims, tp_l, ft);
	PyObject *nd_hft_10r  = PyArray_EMPTY(ndim, dims, tp_l, ft); // time when height is 10% of max on right
	PyObject *nd_w_rms    = PyArray_EMPTY(ndim, dims, tp_d, ft);
	PyObject *nd_b_av_l   = PyArray_EMPTY(ndim, dims, tp_d, ft); // avg of 100 samples to left of pulse bound
	PyObject *nd_b_av_r   = PyArray_EMPTY(ndim, dims, tp_d, ft); // avg of 100 samples to right of pulse bound
	
	double *pA_raw   = (double *)PyArray_DATA(nd_pA_raw);
	double *pA_av    = (double *)PyArray_DATA(nd_pA_av);
	double *pA_av_fw = (double *)PyArray_DATA(nd_pA_av_fw);
	double *pH_raw   = (double *)PyArray_DATA(nd_pH_raw);
	double *pH_av    = (double *)PyArray_DATA(nd_pH_av);
	double *pM       = (double *)PyArray_DATA(nd_pM);
	long   *aft_10   =   (long *)PyArray_DATA(nd_aft_10);
	long   *aft_50   =   (long *)PyArray_DATA(nd_aft_50);
	long   *aft_90   =   (long *)PyArray_DATA(nd_aft_90);
	long   *hft_10l  =   (long *)PyArray_DATA(nd_hft_10l);
	long   *t_max    =   (long *)PyArray_DATA(nd_t_max);
	long   *hft_10r  =   (long *)PyArray_DATA(nd_hft_10r);
	double *w_rms    = (double *)PyArray_DATA(nd_w_rms);
	double *b_av_l   = (double *)PyArray_DATA(nd_b_av_l);
	double *b_av_r   = (double *)PyArray_DATA(nd_b_av_r);
	
	// Now iterate over pulses
	double max_val_raw, max_val_av, min_val;  //, max_val_h
	long max_pos = 0L;
	double pulse_a_raw, pulse_a_av, pulse_a_pos;
	long p_srt, p_stp;
	double m1, m2; // first and second moments
	npy_intp b_wing_samps = 100;
	double b_av_wings;
	npy_intp ib_min, ib_max, i_w;
	npy_intp fw_rise=50; // samples, 100 ns
	npy_intp fw_tail=100; // samples, 200 ns
	for (npy_intp ip=0;ip<num_pulses;ip++) {
		// find total area, max height, max position
		pulse_a_raw = 0.;
		pulse_a_av = 0.;
		pulse_a_pos = 0.;
		p_srt = stsp[ip];
		p_stp = stsp[ip + (long)num_pulses];
		m1 = 0.; // first moment, with respect to pulse start
		max_val_raw = -10000.;
		max_val_av = -10000.;
		//max_val_h = -10000.;
		min_val = 10000.;
		for(long i=p_srt;i<p_stp;i++) {
			pulse_a_raw += d[i];
			pulse_a_av += ch_a[i];
			if (ch_a[i] >= 0.) {
				pulse_a_pos += ch_a[i];
				m1 += ch_a[i] * (double)(i-p_srt);
			}
			if (d[i] > max_val_raw) {
				max_val_raw = d[i];
			}
			if (ch_a[i] > max_val_av) {
				max_val_av = ch_a[i];
				max_pos = i;
			}
			if (ch_a[i] < min_val) {
				min_val = ch_a[i];
			}
		}
		pA_raw[ip] = pulse_a_raw;
		pA_av[ip] = pulse_a_av;
		pH_raw[ip] = max_val_raw;
		pH_av[ip] = max_val_av;
		pM[ip] = min_val;
		t_max[ip] = max_pos - p_srt;
		m1 = m1 / pulse_a_pos; // Now m1 is normalized correctly
		
		// now calculate the area fraction times and the wrms
		pulse_a_av = 0.;
		m2 = 0.;
		for (long i=p_srt;i<p_stp;i++) {
			if (ch_a[i] >= 0.) {
				m2 += ch_a[i] * pow(((double)(i-p_srt)-m1),2.);
			}
			pulse_a_av += ch_a[i];
			if (pulse_a_av < (0.1*pA_av[ip])) {
				aft_10[ip] = i - p_srt;
			}
			if (pulse_a_av < (0.5*pA_av[ip])) {
				aft_50[ip] = i - p_srt;
			}
			if (pulse_a_av < (0.9*pA_av[ip])) {
				aft_90[ip] = i - p_srt;
			}
		}
		//m2 = sqrt(m2 / pulse_a_pos);
		m2 = sqrt(m2 / pulse_a_av);
		w_rms[ip] = m2;
		
		// find hft_10l
		i_w = max_pos;
		while ((ch_a[i_w] > (0.1*max_val_av))&&(i_w>=0)) {
			i_w--;
		}
		hft_10l[ip] = i_w - p_srt;
		// find hft_10r
		i_w = max_pos;
		while ((ch_a[i_w] > (0.1*max_val_av))&&(i_w<num_samps)) {
			i_w++;
		}
		hft_10r[ip] = i_w - p_srt;
		// Now get the averages of the left pulse wing
		b_av_wings = 0.;
		ib_min = intp_max(p_srt-b_wing_samps,0);
		ib_max = p_srt;
		for (npy_intp ib=ib_min; ib<ib_max; ib++) {
			b_av_wings += ch_a[ib];
		}
		b_av_wings /= (double)b_wing_samps;
		b_av_l[ip] = b_av_wings;
		// Now get the averages of the left pulse wing
		b_av_wings = 0.;
		ib_min = p_stp;
		ib_max = intp_min(p_stp+b_wing_samps, num_samps);
		for (npy_intp ib=ib_min; ib<ib_max; ib++) {
			b_av_wings += ch_a[ib];
		}
		b_av_wings /= (double)b_wing_samps;
		b_av_r[ip] = b_av_wings;
		
		// Calculate area of ch_a in a fixed window fw_rise
		ib_min = intp_max(max_pos-fw_rise, 0);
		ib_max = intp_min(max_pos+fw_tail, num_samps);
		pulse_a_av = 0.;
		for (npy_intp ifw=ib_min; ifw<ib_max; ifw++) {
			pulse_a_av += ch_a[ifw];
		}
		pA_av_fw[ip] = pulse_a_av;
	}
	
	// Now create the output dict and fill it
	PyObject *out_dict = PyDict_New();
	
	const char *key = "pA_raw";
	int r = PyDict_SetItemString(out_dict, key, nd_pA_raw);
	Py_DECREF(nd_pA_raw);
	
	key = "pA_av";
	r = PyDict_SetItemString(out_dict, key, nd_pA_av);
	Py_DECREF(nd_pA_av);
	
	key = "pA_av_fw";
	r = PyDict_SetItemString(out_dict, key, nd_pA_av_fw);
	Py_DECREF(nd_pA_av_fw);
	
	key = "pH_raw";
	r = PyDict_SetItemString(out_dict, key, nd_pH_raw);
	Py_DECREF(nd_pH_raw);
	
	key = "pH_av";
	r = PyDict_SetItemString(out_dict, key, nd_pH_av);
	Py_DECREF(nd_pH_av);
	
	key = "pM";
	r = PyDict_SetItemString(out_dict, key, nd_pM);
	Py_DECREF(nd_pM);
	
	key = "aft_10";
	r = PyDict_SetItemString(out_dict, key, nd_aft_10);
	Py_DECREF(nd_aft_10);
	
	key = "aft_50";
	r = PyDict_SetItemString(out_dict, key, nd_aft_50);
	Py_DECREF(nd_aft_50);
	
	key = "aft_90";
	r = PyDict_SetItemString(out_dict, key, nd_aft_90);
	Py_DECREF(nd_aft_90);
	
	key = "hft_10l";
	r = PyDict_SetItemString(out_dict, key, nd_hft_10l);
	Py_DECREF(nd_hft_10l);
	
	key = "t_max";
	r = PyDict_SetItemString(out_dict, key, nd_t_max);
	Py_DECREF(nd_t_max);
	
	key = "hft_10r";
	r = PyDict_SetItemString(out_dict, key, nd_hft_10r);
	Py_DECREF(nd_hft_10r);
	
	key = "w_rms";
	r = PyDict_SetItemString(out_dict, key, nd_w_rms);
	Py_DECREF(nd_w_rms);
	
	key = "b_av_l";
	r = PyDict_SetItemString(out_dict, key, nd_b_av_l);
	Py_DECREF(nd_b_av_l);
	
	key = "b_av_r";
	r = PyDict_SetItemString(out_dict, key, nd_b_av_r);
	Py_DECREF(nd_b_av_r);
	
	Py_DECREF(nd_d);
	Py_DECREF(nd_ch_a);
	Py_DECREF(nd_stsp);
	
	return out_dict;
}

static PyObject *meth_get_nfold(PyObject *self, PyObject *args) {
	// function signature:
	// get_nfold(p_starts, p_stops, p_starts_i, p_stops_i, pA_av_i)
	PyArrayObject *nd_srt;
	PyArrayObject *nd_stp;
	PyArrayObject *nd_srt_i;
	PyArrayObject *nd_stp_i;
	PyArrayObject *nd_pA_av_i;
	if(!PyArg_ParseTuple
	(args, "O&O&O&O&O&", 
		PyArray_Converter, &nd_srt,
		PyArray_Converter, &nd_stp,
		PyArray_Converter, &nd_srt_i,
		PyArray_Converter, &nd_stp_i,
		PyArray_Converter, &nd_pA_av_i)) {
		return NULL;
	}
	npy_intp numPulses = PyArray_SIZE(nd_srt);
	npy_intp numAltPulse = PyArray_SIZE(nd_srt_i);
	
	int ft = 0;
	int tp_l = NPY_LONG; // type long int
	int tp_d = NPY_DOUBLE; // type double
	int ndim = 1;
	npy_intp dims[ndim];
	dims[0] = numPulses;
	PyObject *nd_nCoinc = PyArray_EMPTY(ndim, dims, tp_l, ft);
	PyObject *nd_pA_coin = PyArray_ZEROS(ndim, dims, tp_d, ft);
	
	long *srt       = (long *)PyArray_DATA(nd_srt); // Channel under test
	long *p_srt     = (long *)PyArray_DATA(nd_srt_i); // Alternate channel
	long *stp       = (long *)PyArray_DATA(nd_stp);
	long *p_stp     = (long *)PyArray_DATA(nd_stp_i);
	double *pA_av_i = (double *)PyArray_DATA(nd_pA_av_i);
	long *nCoinc    = (long *)PyArray_DATA(nd_nCoinc);
	double *pA_coin = (double *)PyArray_DATA(nd_pA_coin);
	
	long num_coinc;
	long num_while_iters;
	npy_intp ip = 0;
	for (npy_intp i0=0; i0<numPulses; i0++) {
		num_while_iters = 0L;
		num_coinc = 0L;
		while ((p_srt[ip] < stp[i0])&&(ip<numAltPulse)&&(num_while_iters<1000)) {
			if ((p_srt[ip]<=srt[i0])&&(p_stp[ip]>srt[i0])) {
				num_coinc += 1L;
				pA_coin[i0] = pA_av_i[ip];
			}
			else if ((p_srt[ip]<stp[i0])&&(p_stp[ip]>=stp[i0])) {
				//num_coinc += 2L; --> for testing purposes
				num_coinc += 1L;
				pA_coin[i0] = pA_av_i[ip];
			}
			else if ((srt[i0]<=p_srt[ip])&&(stp[i0]>p_srt[ip])) {
				//num_coinc += 4L; --> for testing purposes
				num_coinc += 1L;
				pA_coin[i0] = pA_av_i[ip];
			}
			ip++;
		}
		
		if (ip == numAltPulse) {
			ip -= 1L;
		}
		if (num_coinc > 1L) {
			num_coinc = 1L;
		}
		if (num_while_iters>998) {
			nCoinc[i0] = -1;
		} else {
			nCoinc[i0] = num_coinc;
		}
	}
	
	Py_DECREF(nd_srt);
	Py_DECREF(nd_stp);
	Py_DECREF(nd_srt_i);
	Py_DECREF(nd_stp_i);
	Py_DECREF(nd_pA_av_i);
	
	// create a tuple and fill it with the two ndarrays that were created above
	PyObject *out_tuple = PyTuple_New(2);
	PyTuple_SetItem(out_tuple, 0, nd_nCoinc);
	PyTuple_SetItem(out_tuple, 1, nd_pA_coin);
	//Py_DECREF(nd_outarr1); --->  DO NOT DECREF THIS (UNLIKE IN A DICT)
	
	return out_tuple;
}
static PyObject *meth_pulse_bool(PyObject *self, PyObject *args) {
	PyArrayObject *nd_waveform;
	PyArrayObject *nd_pstarts;
	PyArrayObject *nd_pstops;
	if(!PyArg_ParseTuple
	(args, "O&O&O&", 
		PyArray_Converter, &nd_waveform,
		PyArray_Converter, &nd_pstarts,
		PyArray_Converter, &nd_pstops)) {
		return NULL;
	}
	npy_intp numSamps = PyArray_SIZE(nd_waveform);
	npy_intp numPulses = PyArray_SIZE(nd_pstarts);
	
	double *waveform = (double *)PyArray_DATA(nd_waveform);
	long *pstarts = (long *)PyArray_DATA(nd_pstarts);
	long *pstops = (long *)PyArray_DATA(nd_pstops);
	
	// Create the bool array to output
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = numSamps;
	int ft = 0;
	int tp = NPY_BOOL;
	PyObject *nd_outarr = PyArray_EMPTY(ndim, dims, tp, ft);
	npy_bool *outarr = (npy_bool *)PyArray_DATA(nd_outarr);
	
	npy_bool in_pulse = NPY_FALSE;
	npy_intp ip = 0;
	for (npy_intp iw=0; iw<numSamps; iw++) {
		if (iw == pstarts[ip]) {
			in_pulse = NPY_TRUE;
		}
		outarr[iw] = in_pulse;
		if (iw == pstops[ip]) {
			in_pulse = NPY_FALSE;
			if (ip < (numPulses-1)) {
				ip++;
			}
		}
	}
	
	Py_DECREF(nd_waveform);
	Py_DECREF(nd_pstarts);
	Py_DECREF(nd_pstops);
	return nd_outarr;
}
/*
      (       )                   [         ]     
    (   )                            [          ]
    
*/
static PyObject *meth_sayhello(PyObject *self, PyObject *Py_UNUSED(b)) {
	printf("Hello!\n");
	Py_RETURN_NONE;
}
/* ----------------- </MODULE FUNCTIONS> ----------------- */

/* ----------------- <DOC STRINGS> ----------------- */
PyDoc_STRVAR(
	findpeaks__doc__,
	"findpeaks(data,pod_starts,pod_stops)\n--\n\n"
	"Given a 1D array data (dtype np.float64), and a 1D array of pod_start \n"
	"indices (dtype np.int64), and a 1D array of pod_stops (dtype np.int64), \n"
	"this returns a 2D array (3xN, where N is the number of pods), where the\n"
	"first row is the pulse 5% begin (the index), second row is the pulse max\n"
	"index, third row is the pulse 5% end (the index).  The output 3xN array\n"
	"is of dtype np.int64\n");

PyDoc_STRVAR(
	condense_intervals__doc__,
	"condense_intervals(starts_stops, min_dist)\n--\n\n"
	"Inputs:\n"
	"    starts_stops: 2xN array, where each column represents an interval,\n"
	"                  and all intervals are assumed to be sequential. The\n"
	"                  first row are the starting times or indices of the\n"
	"                  intervals, the second row are the stopping times or\n"
	"                  indicies of the intervals.  Here this array must be\n"
	"                  of dtype int64 (long in C)\n"
	"        min_dist: The minimum difference between the stop of one\n"
	"                  interval and the start of the next.  If this\n"
	"                  difference is less than min_dist, then the two\n"
	"                  intervals will be condensed into one.\n"
	"                  min_dist must be a scalar of type [python]int, which\n"
	"                  is C long.\n"
	"Output:\n"
	"    cond_str_stp: A copy of starts_stops input but with nearby\n"
	"                  intervals condensed.\n");

PyDoc_STRVAR(
	get_pulseRQs__doc__,
	"get_pulseRQs(d_raw, ch_a, pls_strstp)\n--\n\n"
	"d_raw is the full trace; should be dtype=double(float64) of raw waveform data\n"
	"    (baseline subtracted)\n"
	"ch_a is the full trace that has been box-average filtered\n"
	"ch_h is ch_a that has had a hat filter (like -1,-1,+1,+1,+1,+1,-1,-1)\n"
	"pls_strstop is the 2xN array of pulse starts (1st row) and stops (2nd row)\n"
	"        should be of type numpy.int64 or long in C\n"
	"Spits out (each a numpy array, compiled into a dict):\n"
	"    pA_raw: full pulse area from the raw waveform\n"
	"    pA_av: full pulse area from the averaged waveform\n"
	"    pH_raw: pulse max of the raw waveform\n"
	"    pH_av: pulse max of the averaged waveform\n"
	"    pH_h: pulse max of the hat-filtered waveform\n"
	"    pM: pulse min\n"
	"    aft_10: samples from pulse start to reach 10% of pulse area (of ch_a)\n"
	"    aft_50: samples from pulse start to reach 50% of pulse area (of ch_a)\n"
	"    aft_90: samples from pulse start to reach 90% of pulse area (of ch_a)\n"
	"    t_max: samples from pulse start to reach max height (of ch_a)\n"
	"    w_rms: width as quantified by the 2nd central moment (of ch_a)\n");

PyDoc_STRVAR(
	highpass__doc__,
	"highpass(t, v, f_c)\n--\n\n"
	"t is the time array, in seconds (or in units of 1/f_c, at least)\n"
	"v is the voltage-signal array, which will get filtered\n"
	"f_c is the high-pass filter cutoff frequency, f_c = 1 / (2pi * RC)\n"
	"Gives a 1D numpy as output, which is the same size and dtype as v\n");

PyDoc_STRVAR(
	lowpass__doc__,
	"lowpass(t, v, f_c)\n--\n\n"
	"t is the time array, in seconds (or in units of 1/f_c, at least)\n"
	"v is the voltage-signal array, which will get filtered\n"
	"f_c is the low-pass filter cutoff frequency, f_c = 1 / (2pi * RC)\n"
	"Gives a 1D numpy as output, which is the same size and dtype as v\n");

PyDoc_STRVAR(
	arbfilt__doc__,
	"arbfilt(d, filt)\n--\n\n"
	"Filter a signal d with a filter 'filt'\n");

PyDoc_STRVAR(
	sayhello__doc__,
	"sayhello()\n--\n\n"
	"Say Hello\n");

PyDoc_STRVAR(
	avebox__doc__,
	"avebox(d, n)\n--\n\n"
	"d is the raw data, n is the width of the box filter in samples.\n"
	"n should be an ODD number, but this is not checked\n");

PyDoc_STRVAR(
	S2filter__doc__,
	"S2filter(d, n1=150, n2=750)\n--\n\n"
	"Give a waveform, summed over channels, (preferably already avebox filtered),\n"
	"and return a single trace that gives the S2 filtered trace (i.e. high response\n"
	"for pulses that are S2-shaped, very small response for pulses that are\n"
	"S1-shaped).  n1 and n2 are optional inputs, must be python ints.");

PyDoc_STRVAR(
	get_nfold__doc__,
	"get_nfold(starts_0, stops_0, starts_i, stops_i, pA_av_i)\n--\n\n"
	"Testing one channel of pulse starts and stops against an array of other pulse\n"
	"starts and stops.  Inputs and types:\n"
	"   starts_0           (ndarray, dtype=int or np.int64)\n"
	"   stops_0            (ndarray, dtype=int or np.int64) same dimensions/size as p_starts\n"
	"   starts_i  (python list, each element is a ndarray, dtype int or np.int64)\n"
	"   stops_i   (like the other list)\n"
	"   pA_av_i    array of pulse areas of the start_i/stop_i");

PyDoc_STRVAR(
	pulse_bool__doc__,
	"pulse_bool(waveform, pstarts, pstops)\n--\n\n"
	"waveform is a 1D array of waveform data\n"
	"    (type C double or py float or numpy float64)\n"
	"ptarts are the starting indices of pulses\n"
	"    (type C long or py int or numpy int64)\n"
	"pstops are the stopping indices of pulses\n"
	"    (type C long or py int or numpy int64)\n"
	"Produces a 1D bool array, same size as waveform, where the elements are False\n"
	"for elements of waveform that are outside of a pulse, and True for elements\n"
	"of waveform that are inside a pulse.\n");
/* ----------------- </DOC STRINGS> ----------------- */

static PyMethodDef ProcessMethods[] = {
	{"findpeaks",meth_findpeaks,METH_VARARGS,findpeaks__doc__},
	{"condense_intervals",meth_condense_intervals,METH_VARARGS,condense_intervals__doc__},
	{"get_pulseRQs",meth_get_pulseRQs,METH_VARARGS, get_pulseRQs__doc__},
	{"highpass",meth_highpass, METH_VARARGS, highpass__doc__},
	{"lowpass",meth_lowpass, METH_VARARGS, lowpass__doc__},
	{"arbfilt",meth_arbfilt, METH_VARARGS, arbfilt__doc__},
	{"avebox",meth_avebox, METH_VARARGS, avebox__doc__},
	{"S2filter",(PyCFunction)meth_S2filter, METH_VARARGS|METH_KEYWORDS, S2filter__doc__},
	{"get_nfold",meth_get_nfold, METH_VARARGS, get_nfold__doc__},
	{"pulse_bool",meth_pulse_bool, METH_VARARGS, pulse_bool__doc__},
	{"sayhello",meth_sayhello,METH_NOARGS,sayhello__doc__},
	{NULL, NULL, 0, NULL}
};

/* the following struct defines properties of the module itself */
static struct PyModuleDef cprocess_module = {
	PyModuleDef_HEAD_INIT,
	"c_process",
	"Low-level pulse-finding/characterizing functions and signal filtering.",
	-1,
	ProcessMethods
};

/*  NOTE: in the function below, 'import_array()' must be included, which does not exist in the other
   other examples that use python-only API functions and variable types.

The name of the function of type PyMODINIT_FUNC has to be "PyInit_{name}", where {name} is the name
of the module as it will be imported from python, and has to match the secend element of the module
struct defined above.
 */
PyMODINIT_FUNC PyInit_c_process(void) {
	import_array();
	return PyModule_Create(&cprocess_module);
}
/* compile on theo with:
gcc -shared -o c_processmodule.so -fPIC c_processmodule.c 
*/