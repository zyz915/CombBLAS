#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#ifdef NOTR1
        #include <boost/tr1/tuple.hpp>
#else
        #include <tr1/tuple>
#endif
#include "../SpParVec.h"
#include "../SpTuples.h"
#include "../SpDCCols.h"
#include "../SpParMat.h"
#include "../DenseParMat.h"
#include "../DenseParVec.h"


using namespace std;
#define ITERATIONS 10

// Simple helper class for declarations: Just the numerical type is templated 
// The index type and the sequential matrix type stays the same for the whole code
// In this case, they are "int" and "SpDCCols"
template <class NT>
class PSpMat 
{ 
public: 
	typedef SpDCCols < int, NT > DCCols;
	typedef SpParMat < int, NT, DCCols > MPI_DCCols;
};


int main(int argc, char* argv[])
{
	MPI::Init(argc, argv);
	int nprocs = MPI::COMM_WORLD.Get_size();
	int myrank = MPI::COMM_WORLD.Get_rank();

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./MultTest <Matrix> <S> <STranspose>" << endl;
			cout << "<Matrix>,<S>,<STranspose> are absolute addresses, and files should be in triples format" << endl;
		}
		MPI::Finalize(); 
		return -1;
	}				
	{
		string Aname(argv[1]);		
		string Sname(argv[2]);
		string STname(argv[3]);		

		ifstream inputA(Aname.c_str());
		ifstream inputS(Sname.c_str());
		ifstream inputST(STname.c_str());

		MPI::COMM_WORLD.Barrier();
		typedef PlusTimesSRing<double, double> PTDOUBLEDOUBLE;	

		PSpMat<double>::MPI_DCCols A, S, ST;	// construct objects
		
		A.ReadDistribute(inputA, 0);
		S.ReadDistribute(inputS, 0);
		ST.ReadDistribute(inputST, 0);
		SpParHelper::Print("Data read\n");

		// force the calling of C's destructor
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(S, C);
			SpParHelper::Print("Warmed up for DoubleBuff (right evaluate)\n");
		}	
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(1,"SpGEMM_DoubleBuff_right");
		double t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(S, C);
		}
		MPI::COMM_WORLD.Barrier();
		double t2 = MPI::Wtime(); 	
		MPI_Pcontrol(-1,"SpGEMM_DoubleBuff_right");
		if(myrank == 0)
		{
			cout<<"Double buffered multiplications (right evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(C, ST);
			SpParHelper::Print("Warmed up for DoubleBuff (left evaluate)\n");
		}	
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(1,"SpGEMM_DoubleBuff_left");
		t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE>(C, ST);
		}
		MPI::COMM_WORLD.Barrier();
		t2 = MPI::Wtime(); 	
		MPI_Pcontrol(-1,"SpGEMM_DoubleBuff_left");
		if(myrank == 0)
		{
			cout<<"Double buffered multiplications (left evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{	
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(S, C);
		}
		SpParHelper::Print("Warmed up for Synch (right evaluate)\n");
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(1,"SpGEMM_Synch_right");
		t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(A, ST);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(S, C);
		}
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(-1,"SpGEMM_Synch_right");
		t2 = MPI::Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Synchronous multiplications (right evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}

		// force the calling of C's destructor
		{	
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(C, ST);
		}
		SpParHelper::Print("Warmed up for Synch (left evaluate)\n");
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(1,"SpGEMM_Synch_left");
		t1 = MPI::Wtime(); 	// initilize (wall-clock) timer
		for(int i=0; i<ITERATIONS; i++)
		{
			PSpMat<double>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(S, A);
			PSpMat<double>::MPI_DCCols D = Mult_AnXBn_Synch<PTDOUBLEDOUBLE>(C, ST);
		}
		MPI::COMM_WORLD.Barrier();
		MPI_Pcontrol(-1,"SpGEMM_Synch_left");
		t2 = MPI::Wtime(); 	
		if(myrank == 0)
		{
			cout<<"Synchronous multiplications (left evaluate) finished"<<endl;	
			printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
		}
		inputA.clear();
		inputA.close();
		inputS.clear();
		inputS.close();
		inputST.clear();
		inputST.clear();
	}
	MPI::Finalize();
	return 0;
}

