#ifndef PY_SP_PAR_MAT_Obj2_H
#define PY_SP_PAR_MAT_Obj2_H

#include "pyCombBLAS.h"

//INTERFACE_INCLUDE_BEGIN
class pySpParMatObj2 {
//INTERFACE_INCLUDE_END
protected:

	typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
	typedef SpParMat < int64_t, int, SpDCCols<int64_t,int> > PSpMat_Int;
	typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;

public:
	typedef int64_t INDEXTYPE;
	typedef SpDCCols<INDEXTYPE,Obj2> DCColsType;
	typedef SpParMat < INDEXTYPE, Obj2, DCColsType > PSpMat_Obj2;
	typedef PSpMat_Obj2 MatType;
	
public:
	
	pySpParMatObj2(MatType other);
	pySpParMatObj2(const pySpParMatObj2& copyFrom);

public:
	MatType A;

/////////////// everything below this appears in python interface:
//INTERFACE_INCLUDE_BEGIN
public:
	pySpParMatObj2();
	pySpParMatObj2(int64_t m, int64_t n, pyDenseParVec* rows, pyDenseParVec* cols, pyDenseParVecObj2* vals);

public:
	//int64_t getnnz();
	int64_t getnee();
	int64_t getnrow();
	int64_t getncol();
	
public:	
	void load(const char* filename);
	void save(const char* filename);
	
	//double GenGraph500Edges(int scale, pyDenseParVec* pyDegrees = NULL, int EDGEFACTOR = 16);
	//double GenGraph500Edges(int scale, pyDenseParVec& pyDegrees);
	
public:
	pySpParMatObj2 copy();
	//pySpParMatObj2& operator+=(const pySpParMatObj2& other);
	pySpParMatObj2& assign(const pySpParMatObj2& other);
	pySpParMat     SpGEMM(pySpParMat&     other, op::SemiringObj* sring);
	pySpParMatObj2 SpGEMM(pySpParMatObj2& other, op::SemiringObj* sring);
	pySpParMatObj1 SpGEMM(pySpParMatObj1& other, op::SemiringObj* sring);
	//pySpParMatObj2 operator*(pySpParMatObj2& other);
#define NOPARMATSUBSREF
	pySpParMatObj2 SubsRef(const pyDenseParVec& rows, const pyDenseParVec& cols);
	pySpParMatObj2 __getitem__(const pyDenseParVec& rows, const pyDenseParVec& cols);
	
	int64_t removeSelfLoops();
	
	void Apply(op::UnaryFunctionObj* f);
	void DimWiseApply(int dim, const pyDenseParVecObj2& values, op::BinaryFunctionObj* f);
	void Prune(op::UnaryPredicateObj* pred);
	int64_t Count(op::UnaryPredicateObj* pred);
	
	// Be wary of identity value with min()/max()!!!!!!!
	pyDenseParVecObj2 Reduce(int dim, op::BinaryFunctionObj* f, Obj2 identity = Obj2());
	pyDenseParVecObj2 Reduce(int dim, op::BinaryFunctionObj* bf, op::UnaryFunctionObj* uf, Obj2 identity = Obj2());
	
	void Transpose();
	//void EWiseMult(pySpParMatObj2* rhs, bool exclude);

	void Find(pyDenseParVec* outrows, pyDenseParVec* outcols, pyDenseParVecObj2* outvals) const;
public:
/*
	pySpParVec SpMV_PlusTimes(const pySpParVec& x);
	pySpParVec SpMV_SelMax(const pySpParVec& x);
	void SpMV_SelMax_inplace(pySpParVec& x);
*/
	pySpParVec     SpMV(const pySpParVec&     x, op::SemiringObj* sring);
	pySpParVecObj2 SpMV(const pySpParVecObj2& x, op::SemiringObj* sring);
	pySpParVecObj1 SpMV(const pySpParVecObj1& x, op::SemiringObj* sring);
	pyDenseParVec     SpMV(const pyDenseParVec&     x, op::SemiringObj* sring);
	pyDenseParVecObj2 SpMV(const pyDenseParVecObj2& x, op::SemiringObj* sring);
	pyDenseParVecObj1 SpMV(const pyDenseParVecObj1& x, op::SemiringObj* sring);
//	void SpMV_inplace(pySpParVec& x, op::SemiringObj* sring);
//	void SpMV_inplace(pyDenseParVec& x, op::SemiringObj* sring);

public:
	static int Column() { return ::Column; }
	static int Row() { return ::Row; }
};

//pySpParMat EWiseMult(const pySpParMat& A1, const pySpParMat& A2, bool exclude);
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj2& B, op::BinaryFunctionObj *bf, bool notB = false, Obj2 defaultBValue = Obj2());
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMatObj1& B, op::BinaryFunctionObj *bf, bool notB = false, Obj1 defaultBValue = Obj1());
pySpParMatObj2 EWiseApply(const pySpParMatObj2& A, const pySpParMat&     B, op::BinaryFunctionObj *bf, bool notB = false, double defaultBValue = 0);

//INTERFACE_INCLUDE_END


// From CombBLAS/promote.h:
/*
template <class T1, class T2>
struct promote_trait  { };

#define DECLARE_PROMOTE(A,B,C)                  \
    template <> struct promote_trait<A,B>       \
    {                                           \
        typedef C T_promote;                    \
    };
*/
DECLARE_PROMOTE(pySpParMatObj2::MatType, pySpParMatObj2::MatType, pySpParMatObj2::MatType)
DECLARE_PROMOTE(pySpParMatObj2::DCColsType, pySpParMatObj2::DCColsType, pySpParMatObj2::DCColsType)

template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,Obj1> >
    {                                           
        typedef SpDCCols<int64_t,Obj1> T_promote;
    };
template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,doubleint> >
    {                                           
        typedef SpDCCols<int64_t,doubleint> T_promote;
    };
///////
template <> struct promote_trait< SpDCCols<int64_t,Obj2> , SpDCCols<int64_t,bool> >       
    {                                           
        typedef SpDCCols<int64_t,Obj2> T_promote;                    
    };

template <> struct promote_trait< SpDCCols<int64_t,bool> , SpDCCols<int64_t,Obj2> >       
    {                                           
        typedef SpDCCols<int64_t,Obj2> T_promote;                    
    };

// Based on what's in CombBLAS/SpDCCols.h:
template <class NIT, class NNT>  struct create_trait< SpDCCols<int64_t, Obj2> , NIT, NNT >
    {
        typedef SpDCCols<NIT,NNT> T_inferred;
    };


#endif
