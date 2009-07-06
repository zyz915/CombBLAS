/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_MAT_H
#define _SP_MAT_H

#include <iostream>
#include <vector>
#include <utility>
#include <tr1/tuple>
#include "SpDefs.h"
#include "promote.h"
#include "LocArr.h"

using namespace std;
using namespace std::tr1;

// Forward declaration (required since a friend function returns a SpTuples object)
template <class IU, class NU>	
class SpTuples;


/**
 ** The abstract base class for all derived sequential sparse matrix classes
 ** Contains no data members, hence no copy constructor/assignment operator
 ** Uses static polymorphism through curiously recurring templates (CRTP)
 ** Template parameters: IT (index type), NT (numerical type), DER (derived class type)
 **/
template < class IT, class NT, class DER >
class SpMat
{
public:
	//! Standard destructor, copy ctor and assignment are generated by compiler, they all do nothing !
	//! Default constructor also exists, and does nothing more than creating Base<Derived>() and Derived() objects
	//! One has to call one of the overloaded create functions to get an nonempty object
	void Create(const vector<IT> & essentials)
	{
		static_cast<DER*>(this)->CreateImpl(essentials);
	}

	void Create(IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples)
	{
		static_cast<DER*>(this)->CreateImpl(size, nRow, nCol, mytuples);
	}
	
	
	SpMat< IT,NT,DER >  operator() (const vector<IT> & ri, const vector<IT> & ci) const;
	
	template <typename SR>
	void SpGEMM( SpMat< IT,NT,DER > & A, SpMat< IT,NT,DER > & B, bool isAT, bool isBT);

	// ABAB: A semiring elementwise operation with automatic type promotion is required for completeness (should cover +/- and .* ?)
	// ABAB: A neat version of ConvertNumericType should be in base class (an operator SpMat<NIT,NNT,NDER>())

	void Split( SpMat< IT,NT,DER > & partA, SpMat< IT,NT,DER > & partB); 
	void Merge( SpMat< IT,NT,DER > & partA, SpMat< IT,NT,DER > & partB); 

	Arr<IT,NT> GetArrays() const
	{
		return static_cast<const DER*>(this)->GetArrays();
	}
	vector<IT> GetEssentials() const
	{
		return static_cast<const DER*>(this)->GetEssentials();
	}

	void Transpose()
	{
		static_cast<DER*>(this)->Transpose();
	}
		
	ofstream& put(ofstream& outfile) const;
	ifstream& get(ifstream& infile);
	
	bool isZero() const { return static_cast<const DER*>(this)->isZero(); }
	IT getnrow() const { return static_cast<const DER*>(this)->getnrow(); }
	IT getncol() const { return static_cast<const DER*>(this)->getncol(); }
	IT getnnz() const  { return static_cast<const DER*>(this)->getnnz(); }

protected:

	template < typename UIT, typename UNT, typename UDER >
	friend ofstream& operator<< (ofstream& outfile, const SpMat< UIT,UNT,UDER > & s);	

	template < typename UIT, typename UNT, typename UDER >
	friend ifstream& operator>> (ifstream& infile, SpMat< UIT,UNT,UDER > & s);

	//! Returns a pointer to SpTuples, in order to avoid temporaries
	//! It is the caller's responsibility to delete the returned pointer afterwards
	template< class SR, class IU, class NU1, class NU2, class DER1, class DER2 >
	friend SpTuples< IU, typename promote_trait<NU1,NU2>::T_promote > *
	MultiplyReturnTuples (const SpMat< IU, NU1, DER1 > & A, const SpMat< IU, NU2, DER2 > & B, bool isAT, bool isBT);

};

#include "SpMat.cpp"	
#endif

