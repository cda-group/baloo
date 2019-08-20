import numpy
import numpy as np
import typing

from .cache import Cache
from .convertors import numpy_to_weld_type
from .convertors.utils import to_weld_vec
from .pyweld.types import WeldStruct, WeldVec, WeldLong, WeldDouble, WeldBit, WeldInt, WeldFloat, WeldChar
from .pyweld.weldobject import WeldObject
from .weld_aggs import weld_aggregate, weld_count
from .weld_utils import weld_cast_array, create_weld_object


class LazyResult(object):
    """Wrapper class around a yet un-evaluated Weld result.

    Attributes
    ----------
    weld_expr : WeldObject or numpy.ndarray
        Expression that needs to be evaluated.
    weld_type : WeldType
        Type of the output.
    ndim : int
        Dimensionality of the output.

    """
    _cache = Cache()

    def __init__(self, weld_expr, weld_type, ndim):
        self.weld_expr = weld_expr
        self.weld_type = weld_type
        self.ndim = ndim

    def __repr__(self):
        return "{}(weld_type={}, ndim={})".format(self.__class__.__name__,
                                                  self.weld_type,
                                                  self.ndim)

    def __str__(self):
        return str(self.weld_expr)

    @property
    def values(self):
        """The internal data representation.

        Returns
        -------
        numpy.ndarray or WeldObject
            The internal data representation.

        """
        return self.weld_expr

    def is_raw(self):
        return not isinstance(self.weld_expr, WeldObject)

    def generate(self):
        return self.weld_expr.generate()
        # to_weld_vec(self.weld_type, self.ndim)

    def evaluate(self, verbose=False, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=True):
        """Evaluate the stored expression.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print output for each Weld compilation step.
        decode : bool, optional
            Whether to decode the result
        passes : list, optional
            Which Weld optimization passes to apply
        num_threads : int, optional
            On how many threads to run Weld
        apply_experimental_transforms : bool
            Whether to apply the experimental Weld transforms.

        Returns
        -------
        numpy.ndarray
            Output of the evaluated expression.

        """
        if isinstance(self.weld_expr, WeldObject):
            old_context = dict(self.weld_expr.context)

            for key in self.weld_expr.context.keys():
                if LazyResult._cache.contains(key):
                    self.weld_expr.context[key] = LazyResult._cache.get(key)

            evaluated = self.weld_expr.evaluate(to_weld_vec(self.weld_type,
                                                            self.ndim),
                                                verbose,
                                                decode,
                                                passes,
                                                num_threads,
                                                apply_experimental_transforms)

            self.weld_expr.context = old_context

            return evaluated
        else:
            return self.weld_expr


# TODO: not really happy having functionality here; maybe have e.g. LazyArray(LazyArrayResult) adding the functionality?
class LazyArrayResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyArrayResult, self).__init__(weld_expr, weld_type, 1)

    @property
    def empty(self):
        if self.is_raw():
            return len(self.weld_expr) == 0
        else:
            return False

    def _aggregate(self, operation):
        return LazyScalarResult(weld_aggregate(self.weld_expr,
                                               self.weld_type,
                                               operation),
                                self.weld_type)

    def min(self):
        """Returns the minimum value.

        Returns
        -------
        LazyScalarResult
            The minimum value.

        """
        return self._aggregate('min')

    def max(self):
        """Returns the maximum value.

        Returns
        -------
        LazyScalarResult
            The maximum value.

        """
        return self._aggregate('max')

    def _lazy_len(self):
        return LazyLongResult(weld_count(self.weld_expr))

    def __len__(self):
        """Eagerly get the length.

        Note that if the length is unknown (such as for a WeldObject stop),
        it will be eagerly computed by evaluating the data!

        Returns
        -------
        int
            Length.

        """
        if self._length is None:
            self._length = self._lazy_len().evaluate()

        return self._length

    def _astype(self, dtype):
        return weld_cast_array(self.values,
                               self.weld_type,
                               numpy_to_weld_type(dtype))


# could make all subclasses but seems rather unnecessary atm
class LazyScalarResult(LazyResult):
    def __init__(self, weld_expr, weld_type):
        super(LazyScalarResult, self).__init__(weld_expr, weld_type, 0)

    def op(self, op, other=None, return_type=None, right=False):
        lhs = self.weld_expr.obj_id
        result = LazyScalarResult(WeldObject(None, None), self.weld_type if return_type is None else return_type)
        if other is None:
            result.weld_expr.weld_code = "{}{}".format(op, lhs)
            result.weld_expr.dependencies[lhs] = self.weld_expr
        else:
            result.weld_expr.dependencies[lhs] = self.weld_expr
            if hasattr(other, 'weld_expr'):
                result.weld_expr.dependencies[other.weld_expr.obj_id] = other.weld_expr
                if self.weld_type == other.weld_type:
                    rhs = other.weld_expr.obj_id
                else:
                    rhs = "{}({})".format(self.weld_type, other.weld_expr.obj_id)
            else:
                weld_type = python_expr_to_weld_type(other)
                if self.weld_type == weld_type:
                    rhs = other
                else:
                    rhs = "{}({})".format(self.weld_type, other)
            if right:
                result.weld_expr.weld_code = "{} {} {}".format(rhs, op, lhs)
            else:
                result.weld_expr.weld_code = "{} {} {}".format(lhs, op, rhs)
        return result

    def __add__(self, other):
        return self.op('+', other)

    def __sub__(self, other):
        return self.op('-', other)

    def __mul__(self, other):
        return self.op('*', other)

    def __truediv__(self, other):
        return self.op('/', other)

    def __eq__(self, other):
        return self.op('==', other, return_type=WeldBit())

    def __ne__(self, other):
        return self.op('!=', other, return_type=WeldBit())

    def __or__(self, other):
        return self.op('||', other, return_type=WeldBit())

    def __and__(self, other):
        return self.op('&&', other, return_type=WeldBit())

    def __pow__(self, power, modulo=None):
        ...

    def __gt__(self, other):
        return self.op('>', other, return_type=WeldBit())

    def __ge__(self, other):
        return self.op('<=', other, return_type=WeldBit())

    def __lt__(self, other):
        return self.op('<', other, return_type=WeldBit())

    def __le__(self, other):
        return self.op('<=', other, return_type=WeldBit())

    def __mod__(self, other):
        self.op('%', other)

    def __xor__(self, other):
        self.op('^', other)

    def __cmp__(self, other):
        ...

    def __abs__(self):
        ...

    def __neg__(self, other):
        return self.op('-')

    # r ops

    def __radd__(self, other):
        return self.op('+', other, right=True)

    def __rsub__(self, other):
        return self.op('-', other, right=True)

    def __rmul__(self, other):
        return self.op('*', other, right=True)

    def __rtruediv__(self, other):
        return self.op('/', other, right=True)

    def __ror__(self, other):
        return self.op('||', other, return_type=WeldBit(), right=True)

    def __rand__(self, other):
        return self.op('&&', other, return_type=WeldBit(), right=True)

    def __rpow__(self, power, modulo=None):
        ...

    def __rmod__(self, other):
        self.op('%', other, right=True)

    def toInt(self):
        result = LazyScalarResult(WeldObject(None, None), WeldLong())
        result.weld_expr.weld_code = "i64({})".format(self.weld_expr.obj_id)
        result.weld_expr.dependencies[self.weld_expr.obj_id] = self.weld_expr
        return result

    def toFloat(self):
        result = LazyScalarResult(WeldObject(None, None), WeldDouble())
        result.weld_expr.weld_code = "f64({})".format(self.weld_expr.obj_id)
        result.weld_expr.dependencies[self.weld_expr.obj_id] = self.weld_expr
        return result


class LazyLongResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldLong(), 0)


class LazyDoubleResult(LazyScalarResult):
    def __init__(self, weld_expr):
        super(LazyScalarResult, self).__init__(weld_expr, WeldDouble(), 0)


class LazyStructResult(LazyResult):
    # weld_types should be a list of the Weld types in the struct
    def __init__(self, weld_expr, weld_types):
        super(LazyStructResult, self).__init__(weld_expr, WeldStruct(weld_types), 0)

    def __getitem__(self, key):
        if isinstance(key, slice):
            lazy = lazify(WeldStruct(self.weld_type.field_types[key]))
            fields = ','.join(
                ["{}.${}".format(self.weld_expr.obj_id, k) for k in
                 range(key.start, key.stop, 1 if key.step is None else key.step)])
            lazy.weld_expr.weld_code = '{{ {} }}'.format(fields)
        else:
            lazy = lazify(self.weld_type.field_types[key])
            lazy.weld_expr.weld_code = "{}.${}".format(self.weld_expr.obj_id, key)
        lazy.weld_expr.dependencies[self.weld_expr.obj_id] = self.weld_expr
        return lazy


class LazyStructOfVecResult(LazyStructResult):
    # weld_types should be a list of the Weld types in the struct
    def __init__(self, weld_expr, weld_types):
        weld_vec_types = [WeldVec(weld_type) for weld_type in weld_types]

        super(LazyStructOfVecResult, self).__init__(weld_expr, weld_vec_types)


# Creates a lazy value out of a Weld type
def lazify(weld_type, weld_code=""):
    weld_expr = WeldObject(None, None)
    weld_expr.weld_code = weld_code
    if isinstance(weld_type, (WeldLong, WeldDouble, WeldBit)):
        return LazyScalarResult(weld_expr, weld_type)
    elif isinstance(weld_type, WeldVec):
        return LazyArrayResult(weld_expr, weld_type.elemType)
    elif isinstance(weld_type, WeldStruct):
        if all(map(lambda ty: isinstance(ty, WeldVec), weld_type.field_types)):  # Tuple of arrays
            weld_types = [ty.elemType for ty in weld_type.field_types]
            return LazyStructOfVecResult(weld_expr, weld_types)
        elif all(map(lambda ty: isinstance(ty, (WeldStruct, WeldVec, WeldLong, WeldDouble, WeldBit)),
                     weld_type.field_types)):  # Mixed tuple
            weld_types = [ty for ty in weld_type.field_types]
            return LazyStructResult(weld_expr, weld_types)
        else:
            raise TypeError('Unsupported type in struct {}', weld_type)
    else:
        raise TypeError('Unsupported type {}', weld_type)


# Translates a Python type to a Weld type, e.g. Tuple[int] to WeldStruct(WeldInt())
def python_type_to_weld_type(python_type):
    if python_type is int:
        return WeldLong()
    elif python_type is float:
        return WeldDouble()
    elif python_type is bool:
        return WeldBit()
    elif python_type is str:
        return WeldVec(WeldChar)
    elif hasattr(python_type, '__origin__'):
        origin = python_type.__origin__
        if origin is list:
            elem_type = python_type_to_weld_type(python_type.__args__[0])
            return WeldVec(elem_type)
        elif origin is tuple:
            field_types = [python_type_to_weld_type(arg) for arg in python_type.__args__]
            return WeldStruct(field_types)
    else:
        raise TypeError('Unsupported type {}'.format(python_type))


# Translates a Python expr to a Weld expr, e.g. (1,2) to WeldStruct(WeldInt())
def python_expr_to_weld_expr(python_expr):
    if isinstance(python_expr, LazyResult):
        return python_expr
    elif isinstance(python_expr, tuple):
        lazy_fields = [field if isinstance(field, LazyResult) else python_expr_to_weld_expr(field) for field in
                       python_expr]
        weld_expr = WeldObject(None, None)
        for lazy_field in lazy_fields:
            weld_expr.dependencies[lazy_field.weld_expr.obj_id] = lazy_field.weld_expr
        lazy_field_types = [lazy_field.weld_type for lazy_field in lazy_fields]
        weld_expr.weld_code = '{{ {} }}'.format(','.join([lazy_field.weld_expr.obj_id for lazy_field in lazy_fields]))
        return LazyStructResult(weld_expr, lazy_field_types)
    elif isinstance(python_expr, int):
        weld_expr = WeldObject(None, None)
        weld_expr.weld_code = str(python_expr) + "L"
        return LazyScalarResult(weld_expr, WeldLong())
    elif isinstance(python_expr, float):
        weld_expr = WeldObject(None, None)
        weld_expr.weld_code = str(python_expr)
        return LazyScalarResult(weld_expr, WeldFloat())
    elif isinstance(python_expr, bool):
        weld_expr = WeldObject(None, None)
        weld_expr.weld_code = "true" if python_expr else "false"
        return LazyScalarResult(weld_expr, WeldBit())
    else:
        raise TypeError('Cannot convert python expr {} to weld expr'.format(python_expr))


# Translates a Python expr to a Weld expr, e.g. (1,2) to WeldStruct(WeldInt())
def python_expr_to_weld_type(python_expr):
    if isinstance(python_expr, int):
        return WeldLong()
    elif isinstance(python_expr, float):
        return WeldInt()
    elif isinstance(python_expr, bool):
        return WeldBit()
    elif isinstance(python_expr, list):
        return WeldVec(python_expr_to_weld_type(python_expr[0]))
    elif isinstance(python_expr, tuple):
        return WeldStruct([python_expr_to_weld_type(field) for field in python_expr])
    elif isinstance(python_expr, LazyResult):
        return python_expr.weld_type
    else:
        raise TypeError('Cannot convert python expr {} to weld expr'.format(python_expr))


def weld_to_numpy_type(weld_type):
    if isinstance(weld_type, WeldLong):
        return numpy.dtype(numpy.long)
    elif isinstance(weld_type, WeldDouble):
        return numpy.dtype(numpy.double)
    elif isinstance(weld_type, WeldBit):
        return numpy.dtype(numpy.bool)
    elif isinstance(weld_type, WeldStruct):
        print(weld_type.field_types)
        field_types = [weld_to_numpy_type(field).type for field in weld_type.field_types]
        print(field_types)
        return numpy.dtype(field_types)
    else:
        raise TypeError('Cannot convert {} to numpy'.format(weld_type))
