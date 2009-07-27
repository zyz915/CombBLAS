/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#ifndef _SP_DCCOLS_H
#define _SP_DCCOLS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "Isect.h"
#include "Semirings.h"
#include "MemoryPool.h"
#include "LocArr.h"


template <class IT, class NT>
class SpDCCols: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	// Constructors :
	SpDCCols ();
	SpDCCols (IT size, IT nRow, IT nCol, IT nzc, MemoryPool * mpool = NULL);
	SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool = NULL);
	SpDCCols (const SpDCCols<IT,NT> & rhs);					// Actual copy constructor		
	~SpDCCols();

	template <typename NNT> operator SpDCCols<IT,NNT> () const;		//!< NNT: New numeric type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> operator() (const vector<IT> & ri, const vector<IT> & ci) const;
	bool operator== (const SpDCCols<IT, NT> & rhs) const
	{
		if(nnz != rhs.nnz || m != rhs.m || n != rhs.n)
			return false;
		return ((*dcsc) == (*(rhs.dcsc)));
	}

	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		if(nnz > 0)
			dcsc->Apply(__unary_op);	
	}

	template <typename _BinaryOperation>
	void UpdateDense(NT ** array, _BinaryOperation __binary_op) const
	{
		if(nnz > 0 && dcsc != NULL)
			dcsc->UpdateDense(array, __binary_op);
	}

	void EWiseScale(NT ** scaler, IT m_scaler, IT n_scaler);
	void EWiseMult (const SpDCCols<IT,NT> & rhs, bool exclude);
	
	void Transpose();				//!< Mutator version, replaces the calling object 
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	void CreateImpl(const vector<IT> & essentials);
	void CreateImpl(IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);

	Arr<IT,NT> GetArrays() const;
	vector<IT> GetEssentials() const;
	const static IT esscount = static_cast<IT>(4);

	bool isZero() const { return (nnz == zero); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	
	ofstream& put(ofstream& outfile) const;
	ifstream& get(ifstream& infile);
	void PrintInfo() const;

	template <typename SR> 
	int PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);
	
	template <typename SR>
	int PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);

private:
	void CopyDcsc(Dcsc<IT,NT> * source);
	SpDCCols<IT,NT> ColIndex(const vector<IT> & ci) const;	//!< col indexing without multiplication	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdOutProdMult(const SpDCCols<IT,NTR> & rhs) const;	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdColByCol(const SpDCCols<IT,NTR> & rhs) const;	
	
	SpDCCols (IT size, IT nRow, IT nCol, const vector<IT> & indices, bool isRow);	// Constructor for indexing
	SpDCCols (IT size, IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc);			// Constructor for multiplication

	// Private member variables
	Dcsc<IT, NT> * dcsc;

	IT m;
	IT n;
	IT nnz;
	const static IT zero;
	
	//! store a pointer to the memory pool, to transfer it to other matrices returned by functions like Transpose
	MemoryPool * localpool;

	template <class IU, class NU>
	friend class SpDCCols;		// Let other template instantiations (of the same class) access private members
	
	template <class IU, class NU>
	friend class SpTuples;

	template<typename IU, typename NU1, typename NU2>
	friend SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseMult (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, bool exclude);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);
};

// At this point, complete type of of SpDCCols is known, safe to declare these specialization (but macros won't work as they are preprocessed)
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,int> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,int> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,int> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,unsigned> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,unsigned> >       
    {                                           
        typedef SpDCCols<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,double> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,bool> , SpDCCols<int,float> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,double> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,double> T_promote;                    
    };
template <> struct promote_trait< SpDCCols<int,float> , SpDCCols<int,bool> >       
    {                                           
        typedef SpDCCols<int,float> T_promote;                    
    };



#include "SpDCCols.cpp"
#endif

