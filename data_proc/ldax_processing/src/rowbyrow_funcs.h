#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>

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

npy_float64 float64_max(npy_float64 a, npy_float64 b) {
	if (a > b) {
		return a;
	}
	return b;
}
npy_float64 float64_min(npy_float64 a, npy_float64 b) {
	if (a < b) {
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

int all_iters(int num_iters, NpyIter **iter_arr, NpyIter_IterNextFunc **next_arr) {
	// When there are multiple iterators and one wishes to iterate over
	// them simultaneously, the iterators (and their NextFunc's) can be
	// each put into an array and this function will tackle them all
	// at once.  It returns 1 if they all are still active, or 0 if
	// any are finished.
	int all = 1;
	for (int k=0; k<num_iters; k++) {
		all *= next_arr[k](iter_arr[k]);
	}
	return all;
}

PyObject *rowbyrow_list(
	PyObject* (*f)(PyObject *args), 
	PyArrayObject *nd_i, 
	long axis, 
	PyObject *aux_arrays, 
	PyObject *optargs) {
	
	Py_INCREF(nd_i);
	int ndim = PyArray_NDIM(nd_i);
	/*if (ndim > 2) {
		PyErr_SetString(PyExc_ValueError, "Main array must be 1d or 2d");
	}*/
	
	npy_intp *dims = PyArray_DIMS(nd_i);
	
	Py_ssize_t auxarr_length = py_ssize_t_max(PyTuple_Size(aux_arrays), 0);
	Py_ssize_t optarg_length = py_ssize_t_max(PyTuple_Size(optargs), 0);
	
	PyObject *passargs = PyTuple_New(1 + auxarr_length + optarg_length);
	
	PyObject *temp_item;
	for (int k=0; k<optarg_length; k++) {
		temp_item = PyTuple_GetItem(optargs, k);
		Py_INCREF(temp_item);
		PyTuple_SetItem(passargs, 1+auxarr_length+k, temp_item);
	}
	
	long raxis = get_axis(axis, ndim);
	
	npy_intp dims_prod = 1;
	for (int i=0; i<ndim; i++) {
		if (i != raxis) {
			dims_prod *= dims[i];
		}
	}
	PyObject *big_list;
	if (ndim == 1) {
		Py_INCREF(nd_i);
		PyTuple_SetItem(passargs, 0, (PyObject *)nd_i);
		for (int k=0; k<auxarr_length; k++) {
			temp_item = PyTuple_GetItem(aux_arrays, k);
			Py_INCREF(temp_item);
			PyTuple_SetItem(passargs, 1+k, temp_item);
		}
		big_list = f(passargs);
	} else {
		big_list = PyList_New(dims_prod);
		PyObject *slice_full = PySlice_New(NULL, NULL, NULL);
		PyObject *slices_1d = PyTuple_New(ndim);
		PyObject *slices_nd = PyTuple_New(ndim);
		for (int k=0; k<ndim; k++) {
			Py_INCREF(slice_full);
			if (k == raxis) {
				PyTuple_SetItem(slices_1d, k, slice_full);
				PyTuple_SetItem(slices_nd, k, PyLong_FromLong(0));
			} else {
				PyTuple_SetItem(slices_1d, k, PyLong_FromLong(0));
				PyTuple_SetItem(slices_nd, k, slice_full);
			}
		}
		
		PyArrayObject *aslice_rows[1 + auxarr_length];
		PyArrayObject *aslice_mats[1 + auxarr_length];
		
		aslice_rows[0] = (PyArrayObject *)PyObject_GetItem((PyObject *)nd_i, slices_1d);
		aslice_mats[0] = (PyArrayObject *)PyObject_GetItem((PyObject *)nd_i, slices_nd);
		Py_INCREF(aslice_rows[0]);
		PyTuple_SetItem(passargs, 0, (PyObject *)aslice_rows[0]);
		
		PyObject *temp_aux;
		for (int i=0; i<auxarr_length; i++) {
			temp_aux = PyTuple_GetItem(aux_arrays, i);
			aslice_rows[i+1] = (PyArrayObject *)PyObject_GetItem(temp_aux, slices_1d);
			Py_INCREF(aslice_rows[i+1]);
			PyTuple_SetItem(passargs, i+1, (PyObject *)aslice_rows[i+1]);
			aslice_mats[i+1] = (PyArrayObject *)PyObject_GetItem(temp_aux, slices_nd);
		}
		
		// Initialize and create array of numpy iterators
		NpyIter *iters[1+auxarr_length];
		for (int i=0; i<(1+auxarr_length); i++) {
			iters[i] = NpyIter_New(aslice_mats[i], NPY_ITER_READONLY, NPY_CORDER, NPY_NO_CASTING, NULL);
		}
		
		// Initialize and create array of iterator-next functions
		NpyIter_IterNextFunc *iternexts[1+auxarr_length];
		for (int i=0; i<(1+auxarr_length); i++) {
			iternexts[i] = NpyIter_GetIterNext(iters[i], NULL);
			if (iternexts[i] == NULL) {
				printf("NULL ITERATOR--------------------\n");
				NpyIter_Deallocate(iters[i]);
			}
		}
		
		// Initialize and create array of data-pointer arrays
		char **dataptrs[1+auxarr_length];
		for (int i=0; i<(auxarr_length+1); i++) {
			dataptrs[i] = NpyIter_GetDataPtrArray(iters[i]);
		}
		
		// Initialize array of data pointers (which will access the data as iterator iterates)
		char *datas[1+auxarr_length];
		
		// Iterate over the arrays and pass slices to the given lrow function
		PyObject *row_list;
		long k=0L;
		do {
			for (int i=0; i<(1+auxarr_length); i++) {
				datas[i] = *dataptrs[i];
				aslice_rows[i]->data = (void *)datas[i];
			}
			row_list = f(passargs);
			PyList_SetItem(big_list, k, row_list);
			// PyList_SetItem steals the reference, so we should not decref row_list
			//Py_DECREF(row_list);
			k++;
		} while (all_iters(1+auxarr_length, iters, iternexts));
		
		for (npy_intp i=0; i<(1+auxarr_length); i++) {
			NpyIter_Deallocate(iters[i]);
		}
		
		Py_DECREF(slices_1d);
		Py_DECREF(slices_nd);
		Py_DECREF(slice_full);
		for (int i=0; i<(1+auxarr_length); i++) {
			Py_DECREF(aslice_rows[i]);
			Py_DECREF(aslice_mats[i]);
		}
	}
	Py_DECREF(nd_i);
	Py_DECREF(passargs);
	return big_list;
}

void rowbyrow_optargs(void (*f)(PyObject *args), PyObject *nd_i, PyObject *nd_o, long axis, PyObject *optargs) {
	int ndim = PyArray_NDIM(nd_i);
	//npy_intp *dims = PyArray_DIMS(nd_i);
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
