##########################################################################
#  This file is part of the deModel library, a Python package for using
#  Python to model fixed point arithmetic algorithms.
#
#  Copyright (C) 2007 Dillon Engineering, Inc.
#  http://www.dilloneng.com
#
#  The deModel library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public License as
#  published by the Free Software Foundation; either version 2.1 of the
#  License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#  License along with this library. 
#  If not, see <http://www.gnu.org/licenses/>
##########################################################################

'''FxInt class
'''

__author__ = "$Author: guenter $"
__revision__ = "$Revision: 431 $"
__date__ = "$Date: 2007-09-19 19:16:58 +0200 (Wed, 19 Sep 2007) $"


from numpy import sqrt
import math

import numpy
from decimal import Decimal

def arrayFxInt(intWidth, fractWidth, N, value=None):
	'''Create a Numpy array of length N with FxInt instances of
	representation A(intWidth, fractWidth).
	If value is specified the instances are set with the specified value.

	@type   intWidth    : integer
	@param  intWidth    : used bit width for interger part

	@type   fractWidth  : integer
	@param  fractWidth  : used bit width for fractional part

	@type   N           : integer, list, or array
	@param  N           : if N is an integer the value specifies the length of the
												array to be created. If N is a list or array, an
												array of same length is created with the values
												used to initialize the returned array.

	@type   value       : integer or float
	@param  value       : initialize the array with the given value. This
												parameter is ommitted if N is a list

	@rtype              : numpy array
	@return             : numpy array with N instances of FxInt()
	'''

	if(isinstance(N, int)):
 
		retA = numpy.array([FxInt(intWidth, fractWidth) for i in range(N)])

		if(value):
			for i, item in enumerate(retA):
				item.value = value

	elif(isinstance(N, (list, numpy.ndarray))):
		retA = numpy.array([FxInt(intWidth, fractWidth, value) \
				for value in N])

	else:
		raise TypeError("type(N) = '%s' not supported" %type(N))

	return retA



class FxIntOverflowError(OverflowError):
	"Used to indicate that a set value exceeds the specified width of FxInt."


class FxInt(object):
	''' 
	Fixed point fractional arithmetic data type
	===========================================

	Introduction
	------------

	The class is an abstract data type that can be used to perform fixed-
	point arithmetic. The data type keeps track of the decimal point and 
	arithmetic operations affect the position of the decimal point based
	on the fundamental rules of fixed-point arithmetic.

	The data type is for signed numbers. It can be initiated with an
	integer number, then the value is just assigned to the data type. It
	also can be initiated with a floating point number, which is scaled
	based on the fractional width.

	The data type always needs to be initiated with an integer width and
	a fractional width. The integer width specifies how many bits are used
	to represent the integer part of the value. The fractional width
	specifies how many bits represent the fractional part of the value. As
	the value is always considered a signed number, the overall width is
	M{width = integer width + fractional width + 1}.

	There are different nomenclatures used to specify fixed point
	fractional data types. One commonly used one is the s-number
	representation. For example s0.9 specifies a signed fixed point number
	with 0 bits being used to represent the integer width and 9 bits are
	used to represent the fractional width of the number. In this
	documentation we use a second type of representation which is
	A(0,9). The capital 'A' specifies the number to be a signed number,
	with the first number in the parenthesis being the integer bit width and
	the second number after the comma being the fractional bit width. Note
	that due to the fact that both representations show a signed number,
	the overall width of this number is 10 bit.


	Fundamental fixed point arithmetic operations
	---------------------------------------------

	The class follows the fundamental fixed point arithmetic rules as
	described in the document "Fixed Point Arithmetic: An Introduction" by
	Randy Yates. Availble from this page: 
	
	http://www.digitalsignallabs.com/fp.pdf


	Basic usage 
	-----------

	This section describes the basic usage of the class. For further
	details refer to the respective documentation of member functions.

	>>> from deModel import FxInt
	
	>>> a = FxInt(8,2, 2.5)
	>>> print a
	<10 (2.500) A(8,2)>
	
	>>> b = FxInt(8,2, 3.75)
	>>> print b
	<15 (3.750) A(8,2)>

	>>> c = a + b

	>>> print c
	<25 (6.250) A(9,2)>

	>>> d = a * b
	>>> print d
	<150 (9.375) A(17,4)>

	Here some examples in connection with numpy

	>>> from deModel import arrayFixedInt
	>>> a = arrayFixedInt(8,2, [4.5, 1.25, 3.75, 2.0])
	>>> print a
	[<18 (4.500) A(8,2)> <5 (1.250) A(8,2)> <15 (3.750) A(8,2)>
	<8 (2.000) A(8,2)>]

	>>> b = arrayFixedInt(8,2, [2.25, 3.0, 1.5, 3.75])
	>>> print b
	[<9 (2.250) A(8,2)> <12 (3.000) A(8,2)> <6 (1.500) A(8,2)>
	<15 (3.750) A(8,2)>]
	>>> c = a + b
	>>> print c
	[<27 (6.750) A(9,2)> <17 (4.250) A(9,2)> <21 (5.250) A(9,2)>
	<23 (5.750) A(9,2)>]
	
	Internals
	---------

	The class specifies only a few private variables and to save memory
	they are fixed via the __slots__ member variable. There are two
	notable effects of this. One is that only assignments to member
	variables are allowed that are listed in the __slots__ variable.
	Another is that by default no weak reference is supported for an
	instance of this class. For further details on this refer to: 
	http://docs.python.org/ref/slots.html#l2h-218

	The stored data are all set as private data and if necessary can be
	accessed via properties. For example the actual value is stored in the
	variable self.__value and can be accessed via the value property. This
	allows for the set property for example to test the data type and in
	case of a float value to convert the float to integer, depending on
	the specified integer and fractional width.

	Integer and fractional width are values that can be specified when
	instantiating the class and their values are later read only. This is
	due to the fact that they are changed indirect by operations applied
	to the actual value of the class.

	The class supports the read only property width, which returns the
	used bit width. The bit width  is integer width + fractional width + 1. 
	

	'''

	__slots__ = ('__intWidth', '__fractWidth', '__roundMode', '__overflowMode', 
		'__value', '__ideal', '__uncertainty')

	def __init__(self, intWidth=0, fractWidth=15, value=0, 
		roundMode='round_even', overflowMode='saturate', ideal=None, uncertainty=None):

		'''    
		@type   intWidth    : unsigned integer number
		@param  intWidth    : Number of bits used to store the integer part of the 
													value. As the class stores signed numbers the resulting 
													bit width is intWidth + fractWidth + 1

		@type   fractWidth  : unsigned integer number
		@param  fractWidth  : Number of bits that are used to store the fractional
													part of the value. The fractional width determines
													the scaling that is applied to floating point values.
													The maximum value allowed is 1 bit smaller than width,
													as FxInt is storing signed numbers.
		
		@type   value       : integer or floating point number
		@param  value       : Assigns the initial value to the data type. If the value
													is of integer type the value is just assigned as is. If 
													the value is of float type the value is scaled up,
													depending on the fractWidth value.

		@type   roundMode   : string
		@param  roundMode   : Specifies the way rounding is done for operations 
													with this data type. The setting affects the rounding
													done when converting a floating point value to fixed 
													point representation
													Possible settings:
													'trunc'       - truncate the result
													'round_even'  - round the result to the nearest even value
													'round'       - round the result

		@type   overflowMode : string
		@param  overflowMode : Specifies the way overflow is handled. The setting 
													affects assignment of an integer value.
													Possible settings:
													'saturate' - saturate to max or min
													'wrap'  - round the result to the nearest even value
		'''
	 
		# Test for proper parameter
		# Setting the value will be tested through the property function
		if(intWidth < 0):
			raise ValueError( "Integer width needs to be >= 0!")
		if(fractWidth < 0):
			raise ValueError( "Fractional width needs to be >= 0!")

		if( (roundMode != 'trunc') and
				(roundMode != 'round_even') and
				(roundMode != 'round')):
			raise ValueError( "Round mode '%s' not supported!" % roundMode)

		if( (overflowMode != 'saturate') and
				(overflowMode != 'wrap')):
			raise ValueError( "Overflow mode '%s' not supported!" % overflowMode)

		self.__intWidth = intWidth
		self.__fractWidth = fractWidth
		self.__roundMode = roundMode
		self.__overflowMode = overflowMode
		self._setValue(value)
		
	 
	###################################################################### 
	# properties
	###################################################################### 

	def _getValue(self):
		'''
		Return the value
		'''
		return self.__value

	def _setValue(self, value, ideal=None, uncertainty=None):
		'''
		Allow to set the value
		@type     value : integer or float
		@param    value : Set the value. An integer will be set as is. A
											float value will be scaled based on the fractional
											width
		'''
		if(isinstance(value, (Decimal, str))):
			#print "decimal value"
			self._fromDecimal(value)
			self.__ideal = ideal if ideal else Decimal(value)

		elif(isinstance(value, float)):
			#print "float value"
			self._fromFloat(value)
			self.__ideal = ideal if ideal else Decimal(value)

		elif(isinstance(value, int)):
			#print "int value"
			self.__value = value
			self.__ideal = ideal if ideal else Decimal(self._toFloat())

		else:
			print("unkown type: ", type(value))

		self.__uncertainty = uncertainty if uncertainty else Decimal(2**(-self.__fractWidth))

		self._overflowCheck()


	value = property(_getValue, _setValue)



	def _getUncertainty(self):
		return self.__uncertainty
	uncertainty = property(_getUncertainty)


	def _getIdealValue(self):
		return self.__ideal
	ideal = property(_getIdealValue)

	def _getError(self):
		return Decimal(self._toFloat()) - self.__ideal
	error = property(_getError)


	def _getFloatValue(self):
		return self._toFloat()
	fValue = property(_getFloatValue)
	
	def _getIntWidth(self):
		return self.__intWidth
	intWidth = property(_getIntWidth)

	def _getFractWidth(self):
		return self.__fractWidth
	fractWidth = property(_getFractWidth)

	def _getWidth(self):
		'''width property'''
		return  self.__intWidth + self.__fractWidth + 1
	width = property(_getWidth)

	def _getRep(self):
		'''Return the representation of the fixed point number as string'''
		return "A(%d,%d)" % (self.intWidth, self.fractWidth)
	rep = property(_getRep)
 
	###################################################################### 
	# overloaded functions
	###################################################################### 

	def __copy__(self):
		return copy.deepcopy(self)

	def __getitem__(self, key):
		'''Allow to access a bit or slice of bits

		For bit access the respective bit is returned as integer type. For
		slicing a FxInt instance is returned with the value set to the
		sliced bits and intWidth/fractWidth being adjusted based on the
		slice.

		When the slice includes the sign bit it is taken over to the return
		value. If the sign bit is excluded the bits are taken as is with the
		sign bit set to 0.

		For example using the 4-bit number -6 = b1010, slicing bits 3:1 -->
		b101 includes the sign bit, the result is -3.

		Now using the 4-bit number -3 = b1101, slicing bits 2:1 --> b10,
		however, the slice excludes the sign bit, hence the result is 2.

		The same is true for a positive 4-bit number like 5 = b0101. Slicing
		bits 2:1 --> b10. As the sign bit is not included in the slice the
		result is again 2. Notice that even though the msb of the slice is 1
		the result is not negative.

		@type   key : Integer or slice
		@param  key : Index value 0 ... len-1 will return bits lsb ... msb. 
									Negative numbers -1 ... -len will return the bits 
									msb ... lsb.

									For a slice the bits are specified in the order 
									[msb:lsb]. With msb > lsb. The msb bit is not included
									in the slice. For example, the slice [4:] will return
									4 bits, namely bits 3, 2, 1, and 0. The slice [4:2]
									will return 4-2=2 bits, namely bits 3 and 2.

		@rtype  : Integer or FxInt for slice
		@return : Bit or slice specified by key
		'''


		newWidth = msb - lsb + 1
		mask = 2**newWidth - 1

		
		newValue = self.value >> lsb
		if msb > self.width:
			newValue &= mask




		if(isinstance(key, int)):
			i = key
			if(i >= self.width or i < (-self.width)):
				raise IndexError( "list index %d out of range %d ... %d" % \
													(i, -self.width, (self.width-1)))

			if(i < 0):
				shift = self.width + i
			else:
				shift = i
		
			return ((self.value >> shift) & 0x1)

		elif(isinstance(key, slice)):
			msb, lsb = key.start, key.stop

			# first determine the new value
			if(lsb == None):
				lsb = 0
			if(lsb < 0):
				raise ValueError("FxInt[msb:lsb] requires lsb >= 0\n" \
							"            lsb == %d" % lsb)
			if(msb == None or msb == self.width):
				if(msb == None):
					msb = self.width
				newValue = (self.value >> lsb)
			else:
				newValue = None

			if(msb <= lsb):
				raise ValueError( "FxInt[msb:lsb] requires msb > lsb\n" \
							"            [msb:lsb] == [%d:%d]" % (msb, lsb))

			if(msb > self.width):
				raise ValueError("FxInt[msb:lsb] requires msb <= %d\n" \
							"            msb == %d" % (self.width, msb))

			if(not newValue):
				newValue = (self.value & (1 << msb)-1) >> lsb


			# then the new intWidth and fractWidth
			if(lsb < self.fractWidth):
				if(msb > self.fractWidth):
					newFractWidth = self.fractWidth - lsb

					if(msb > self.intWidth + self.fractWidth):
						newIntWidth = self.intWidth
					else:
						newIntWidth = msb - self.fractWidth
				
				else:
					newIntWidth = 0
					newFractWidth = msb - lsb

			else:
				newFractWidth = 0

				if(msb > (self.intWidth + self.fractWidth)):
					newIntWidth = msb - lsb - 1
				else:
					newIntWidth = msb - lsb

			# create new instance and return it
			retValue = FxInt(newIntWidth, newFractWidth, newValue)

			return retValue

		else:
			raise TypeError( "FxInt item/slice index must be integer")

		

	def __repr__(self):
		str = "<%d" % (self.__value)
		str += " (%.6f)" % (self.fValue)
		str += " A(%d,%d)>" % (self.__intWidth, self.__fractWidth)
		return str

	def __str__(self):
		str = "<%d" % (self.__value)
		str += " (%.6f)" % (self.fValue)
		str += " A(%d,%d)>" % (self.__intWidth, self.__fractWidth)
		return str


	def bit_length(self):
		return self.width

	@property
	def twoComp(self):
		v = self.value
		n = self.width
		return v if v >= 0 else v + (1<<n)

	def __index__(self):
		return self.twoComp

	def __hash__(self):
		return hash((self.value, self.__intWidth, self.__fractWidth, self.__ideal, self.__uncertainty))

	def __hex__(self):
		'''Return the hex representation of the value.

		The number is represented with the minimum number of nibbles as 
		needed based on the width.
		Negative numbers are represented as two's complement.
		'''
		width = self.width
		mask = 2** width -1
		fStr = '0x%%.%dX'%(int(math.ceil(width / 4)))
		return fStr % (self.value & mask)

	def __int__(self):
		return self.value

	def __float__(self):
		return self.fValue



	def __mul__(self, other):
		'''Fixed Point multiplication

		Fixed point representation is calculated based on:

		A(a1, b1) * A(a2, b2) = A(a1+a2+1, b1+b2)

		@type other   : - FxInt
										- int;        will be first converted to FxInt based on 
																	operand A intWidth/fractWidth
										- float;      will be scaled and converted to FxInt based
																	on intWidth/fractWidth of operand A
										
		@param other  : Operand B
		
		@rtype  : FxInt
		@return : A * B
		'''
		retValue = FxInt()

		if(isinstance(other, FxInt)):
			
			#print "__mult__: other is FxInt"
			retValue.__intWidth = self.__intWidth + other.__intWidth + 1
			retValue.__fractWidth = self.__fractWidth + other.__fractWidth
			retValue.__roundMode = self.__roundMode
			retValue.__overflowMode = self.__overflowMode

			retValue.value = self.value * other.value
			retValue.__ideal = self.ideal * other.ideal

			# If any of the values are zero - relative uncertainty becomes meaningless
			# Use lsb uncertainty instead.
			if (retValue.value == 0 or self.value == 0 or other.value == 0):
				retValue.__uncertainty = Decimal(2**(-retValue.__fractWidth))
			else:
				retValue.__uncertainty = retValue.ideal * sqrt( (self.__uncertainty / self.ideal)**2 + (other.__uncertainty / other.ideal)**2 )

		elif(isinstance(other, (int, float, str, Decimal))):
			
			#print "__mult__: other is '%s' "% type(other)
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self * b
		
		else:
			msg = "'%s' not supported as operator for FxInt multiplication"%type(other)
			raise TypeError(msg)

		return retValue


	def __div__(self, other):
	#  '''Fixed point division

	#  Fixed pont representation is calculated based on:

	#  A(a1, b1) / A(a2, b2) = A(a1+b2+1, a2+b1)

	#  @type other   : - FxInt
	#                  - int;        will be first converted to FxInt based on 
	#                                operand A intWidth/fractWidth
	#                  - float;      will be scaled and converted to FxInt based
	#                                on intWidth/fractWidth of operand A
										
	#  @param other  : Operand B

	#  @rtype  : FxInt
	#  @return : A / B
	#  '''
		retValue = FxInt()

		if(isinstance(other, FxInt)):
			
			#print "__mult__: other is FxInt"
			retValue.__intWidth = self.__intWidth + other.__intWidth + 1
			retValue.__fractWidth = self.__fractWidth + other.__fractWidth
			retValue.__roundMode = self.__roundMode
			retValue.__overflowMode = self.__overflowMode

			retValue.value = self.value / other.value
			retValue.__ideal = self.ideal / other.ideal
			retValue.__uncertainty = retValue.__ideal * sqrt( (self.__uncertainty / self.__ideal)**2 + (other.__uncertainty / other.__ideal)**2 )

		elif(isinstance(other, (int, float, str, Decimal))):
			
			#print "__mult__: other is '%s' "% type(other)
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self * b
		
		else:
			msg = "'%s' not supported as operator for FxInt division" % type(other)
			raise TypeError(msg)

		return retValue




	def __add__(self, other):
		'''Scale operand b to the representation of operand a and add them
		A(a, b) + A(a, b) = A(a+1, b)

		@type   other : FxInt
		@param  other : Operand B

		@rtype  : FxInt
		@return : A + B
		'''
		retValue = FxInt()

		if(isinstance(other, FxInt)):

			retValue = FxInt(
				intWidth     = max(self.__intWidth, other.__intWidth) + 1,
				fractWidth   = max(self.__fractWidth, other.__fractWidth),
				value        = self.value + other.value,
				roundMode    = self.__roundMode,
				overflowMode = self.__overflowMode,
				ideal        = self.ideal + other.ideal,
				uncertainty  = sqrt( self.__uncertainty**2 + other.__uncertainty**2 )
			);

		elif(isinstance(other, (int, float, str, Decimal))):
			
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self + b
		
		else:
			msg = "'%s' not supported as operator for FxInt addition" % type(other)
			raise TypeError(msg)

		return retValue

	def __radd__(self, other):
		return other + self;

	def __iadd__(self, other):
		'''Scale operand b to the representation of operand a and add them
		A(a, b) + A(a, b) = A(a+1, b)

		@type   other : FxInt
		@param  other : Operand B

		@rtype  : FxInt
		@return : A + B
		'''
		retValue = FxInt()

		if(isinstance(other, FxInt)):

			retValue = FxInt(
				intWidth     = self.__intWidth,
				fractWidth   = self.__fractWidth,
				value        = self.value + other.value,
				roundMode    = self.__roundMode,
				overflowMode = self.__overflowMode,
				ideal        = self.ideal + other.ideal,
				uncertainty  = sqrt( self.__uncertainty**2 + other.__uncertainty**2 )
			);

		elif(isinstance(other, (int, float, str, Decimal))):
			
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self + b
		
		else:
			msg = "'%s' not supported as operator for FxInt addition" % type(other)
			raise TypeError(msg)

		return retValue


	def __sub__(self, other):
		'''Scale operand b to the representation of operand a and subtract them.
		A(a, b) - A(a, b) = A(a+1, b)

		@type   other : FxInt
		@param  other : Operand B

		@rtype    : FxInt
		@return   : A - B
		'''

		retValue = FxInt()

		if(isinstance(other, FxInt)):

			retValue = FxInt(
				intWidth     = max(self.__intWidth, other.__intWidth) + 1,
				fractWidth   = max(self.__fractWidth, other.__fractWidth),
				value        = self.value - other.value,
				roundMode    = self.__roundMode,
				overflowMode = self.__overflowMode,
				ideal        = self.ideal - other.ideal,
				uncertainty  = sqrt( self.__uncertainty**2 + other.__uncertainty**2 )
			);

		elif(isinstance(other, (int, float, str, Decimal))):
			
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self - b
		
		else:
			msg = "'%s' not supported as operator for FxInt subtraction" % type(other)
			raise TypeError(msg)

		return retValue

	def __rsub__(self, other):
		return other - self

	def __isub__(self, other):
		retValue = FxInt()

		if(isinstance(other, FxInt)):

			retValue = FxInt(
				intWidth     = self.__intWidth,
				fractWidth   = self.__fractWidth,
				value        = self.value - other.value,
				roundMode    = self.__roundMode,
				overflowMode = self.__overflowMode,
				ideal        = self.ideal - other.ideal,
				uncertainty  = sqrt( self.__uncertainty**2 + other.__uncertainty**2 )
			);

		elif(isinstance(other, (int, float, str, Decimal))):
			
			b = FxInt(self.__intWidth, self.__fractWidth, other, self.__roundMode, self.__overflowMode)
			retValue = self - b
		
		else:
			msg = "'%s' not supported as operator for FxInt subtraction" % type(other)
			raise TypeError(msg)

		return retValue
		

	def __lshift__(self, other):
		'''Left shift operation
		Shift left the value by the specified amount of bits, without 
		changing intWidth/fractWidth

		@type   other : Integer
		@param  other : Number of bits to shift
		'''
		
		# check for the other value, only support 'int' type
		if(not isinstance(other, int)):
			msg = "unsupported operand type(s) for <<: 'FxInt' and '%s'"% type(other)
			raise TypeError(msg)
		
		if(other < 0):
			raise ValueError("negative shift count")

		return FxInt(
			intWidth     = self.__intWidth,
			fractWidth   = self.__fractWidth,
			value        = self.value << other,
			roundMode    = self.__roundMode,
			overflowMode = self.__overflowMode,
			ideal        = ideal * 2**other,
			uncertainty  = self.__uncertainty * 2**other
			);

		
	
	def __rshift__(self, other):
		'''Right shift operation
		Shift the value by the specified amount of bits, without changing
		intWidth/fractWidth.

		The result will be adjusted based on the selected rounding mode.

		@type   other : integer
		@param  other : Number of bits to shift the value right
		'''
		if(not isinstance(other, int)):
			msg = "unsupported operand type(s) for <<: 'FxInt' and '%s'"% type(other)
			raise TypeError(msg)
		 
		if(other < 0):
			raise ValueError("negative shift count")

		if(other == 0):
			return self

		if(self.__roundMode == 'round'):
			roundBit = self[other-1] # take the msb that would get lost
			shr = (self.value >> other) + roundBit # and add it
			
		elif(self.__roundMode == 'round_even'):
			newBitZero = self[other]
			msbTrunc = self[other-1]
			remainTrunc = self[other-1:0]

			# TODO: should the 'not' work just for FxInt?
			if(msbTrunc and not remainTrunc.value):  # truncing 100..-> round even
				shr =  (self.value >> other) + (newBitZero & msbTrunc)

			else: # not .500.. case, round normal
				shr =  (self.value >> other) + msbTrunc
			
		else:   # __roundMode == 'trunc'
			shr = self.value >> other

		return FxInt(
			intWidth     = self.__intWidth,
			fractWidth   = self.__fractWidth,
			value        = shr,
			roundMode    = self.__roundMode,
			overflowMode = self.__overflowMode,
			ideal        = ideal / 2**other,
			uncertainty  = self.__uncertainty / 2**other
			);



	def __abs__(self):
		"""Modulus"""
		if self.value < 0:
			return -self
		else:
			return self




	def __neg__(self):
		"""Change sign"""
		return FxInt(family=self.family,
					 scaled_value=-self.scaledval)

	def __pos__(self):
		"""Identity operation"""
		return self


	###################################################################### 
	# private methods
	###################################################################### 

	def _fromFloat(self, value):
		'''Convert float value to fixed point'''
		self.__value = self.round(value * 2.0**self.__fractWidth )

	def _fromDecimal(self, value):
		'''Convert decimal value to fixed point'''
		self.__value = self.round(float(value) * 2.0**self.__fractWidth )

	def _toFloat(self):
		'''Convert fixed point value to floating point number'''
		return (self.__value  / (2.0 ** self.__fractWidth))

	def _overflowCheck(self):
		'''Verify that the set value does not exceed the specified width'''
		maxNum = 2 ** (self.width - 1) - 1
		minNum = - 2 ** (self.width - 1)
	
		if(self.__value > maxNum):
			if (self.__overflowMode == "saturate"):
				self.__value = maxNum
				print("WARNING (deModel): Saturating value to max %d %f" % (self.value, self.fValue)) 
			else:
				self.__value = minNum + (self.value - maxNum - 1)
				print("WARNING (deModel): Value overflow, wrapping to %d %f" % (self.value, self.fValue)) 

		if(self.value < minNum):
			if (self.__overflowMode == "saturate"):
				self.__value = minNum
				print("WARNING (deModel): Saturating value to min %d %f" % (self.value, self.fValue))
			else:
				self.__value = maxNum + (self.value - minNum + 1)
				print("WARNING (deModel): Value underflow, wrapping to %d %f" % (self.value, self.fValue)) 


	###################################################################### 
	# public methods (interface)
	###################################################################### 

	def isOverflowing(self, intWidth, fractWidth):
		'''Return True if the stored value exceeds the specified width
		
			This function allows to test whether a value would fit in an 
			instance with different width.

			@type   intWidth    : integer
			@param  intWidth    : integer width
			@type   fractWidth  : integer
			@param  fractWidth  : fractional width

			@rtype              : Boolean
			@return             : True if self.__value is overflowing A(intWidth, fractWidth)
														False if self.__value is not overflowing the specified parameters.
		'''
		maxNum = 2 ** (intWidth + fractWidth) - 1
		minNum = - 2 ** (intWidth + fractWidth)

		retValue = False
	
		if(self.value > maxNum or self.value < minNum):
			retValue = True

		return retValue


	def newRep(self, intWidth, fractWidth, mode='trunc'):
		'''Change the fixed point representation to the specified representation.

		The operation changes the intWidth and fractWidth based on the given
		parameter. The value of the instance is changed by this operation,
		however the representing floating point number stays the same,
		except for rounding issues when reducing the fractional width.

		If the number does not fit the new representation a FxIntOverflowError
		exception is called.

		@type   intWidth    : integer
		@param  intWidth    : new integer width

		@type   fractWidth  : integer
		@param  fractWidth  : new fractional representation

		@type   mode        : string
		@param  mode        : How to handle changes in binary point position
							  Possible settings:
							  'trunc' - No rescaling intent. Ideal value unchanged. Integer bits are added/discarded, overflow handled if necessary.
							  'rescale' - Rescaling intended. Ideal value shifted along with binary point, overflow handled if necessary.
		'''

		def shifter(v, n):
			if isinstance(v, int):
				if n < 0:
					return v >> abs(n)
				elif n > 0:
					return v << n
				else:
					return v

			elif isinstance(v, Decimal):
				return v * Decimal(2.0**n)

			else:
				raise ValueError( "Type '%s' not supported!".format(type(v)))

		ideal = self.ideal
		value = self.value;
		uncertainty = self.uncertainty;

		nFractional = fractWidth - self.fractWidth
		nInteger = intWidth - self.intWidth

		if (mode == 'trunc'):
			value = shifter(value, nFractional)
			ideal = shifter(ideal, 0)
			uncertainty = shifter(uncertainty, 0)

		elif (mode == 'scale'):
			value = shifter(value, nFractional + nInteger)
			ideal = shifter(ideal, nInteger)
			uncertainty = shifter(uncertainty, nInteger)

		else:
			raise ValueError( "Mode '%s' not supported!".format(mode))
		
		self.__fractWidth = fractWidth
		self.__intWidth = intWidth

		self.value = value    # overflow checks performed by _setValue()
		self.__ideal = ideal
		#print('checking uncertainty, self={0}, calc={1}'.format(self.__uncertainty, uncertainty))
		self.__uncertainty = max(self.__uncertainty, uncertainty)



	def round(self, value):
		'''Return the floating point value as int, rounded depending on the 
		roundMode setting.

		@type   value : float
		@param  value : Value to be rounded based on the set self.__roundMode

		@rtype        : float
		@return       : Based on the set self.__roundMode rounded number
		'''
		if(self.__roundMode == 'trunc'):
			retVal = int(value)

		elif(self.__roundMode == 'round_even'):
			# if value is .50 round to even, if not, round normal
			fract, integer = math.modf(value)
			absIValue = int(abs(integer))
			if(int(integer) < 0):
				sign = -1
			else:
				sign = 1

			# TODO: look for a better way to compare here for 0.500
			# floating point compare does not seem to be so good
			if((abs(fract) - 0.5) == 0.0):
				if((absIValue%2) == 0):  # even
					retVal = absIValue * sign
				else:                 # odd
					retVal = (absIValue + 1) * sign
			else:
				retVal = round(value)

		elif(self.__roundMode == 'round'):
			retVal = round(value)

		else:
			raise "ERROR: FxInt.round(): '%s' not supported round mode!" % \
					self.__roundMode

		return int(retVal)


	
	def showRange(self):
		'''
		Print out the possible value range of the number.
		'''
		min = -2**self.intWidth
		max = 2**self.intWidth - 1.0 / 2.0**self.fractWidth
		print( "A(%d, %d): " %(self.intWidth, self.fractWidth))
		print( "%f ... %f" % (min, max))
 

	def showValueRange(self):
		'''Print out the integer # and its floating point representation'''
		fract = 2**self.fractWidth
		min = -2**self.intWidth
		for i in range(2**self.width):
			print( "i: %d --> %f" %(i, (min+ i/ 2.0**self.fractWidth)))

	def bin(self):
		return ('0b{0:0>' + str(self.width) + 'b}').format(self.twoComp)


###################################################################### 
# 
# main()
# 
if __name__ == '__main__':

	a = FxInt()
	a.value = 1

	print( "Showing range:")
	a.showRange()
	print( "printing a: ", a)

	a = FxInt(8, 0, 1)
	print("Showing range:")
	a.showRange()
	print( "printing a: ", a)

	a = FxInt(8, 3, 1.2)
	print( "Showing range: ")
	a.showRange()
	print("printig a: ", a)
	
	a = FxInt(8, 2)
	print("Representation a: ", a.rep)
	
	b = FxInt(8, 0)
	print("Representation b: ", b.rep)

	c = a + b
	print("Representation c: ", c.rep)

	a = 1.25
	b = 2.0
	c = a + b
	print(c)








