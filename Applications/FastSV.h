#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include <string>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/SpHelper.h"

/**
 ** Connected components based on Shiloach-Vishkin algorithm
 **/

namespace combblas {

template <typename T1, typename T2>
struct Select2ndMinSR
{
	typedef typename promote_trait<T1,T2>::T_promote T_promote;
	static T_promote id(){ return std::numeric_limits<T_promote>::max(); };
	static bool returnedSAID() { return false; }
	static MPI_Op mpi_op() { return MPI_MIN; };

	static T_promote add(const T_promote & arg1, const T_promote & arg2) {
		return std::min(arg1, arg2);
	}

	static T_promote multiply(const T1 & arg1, const T2 & arg2) {
		return static_cast<T_promote> (arg2);
	}

	static void axpy(const T1 a, const T2 & x, T_promote & y) {
		y = add(y, multiply(a, x));
	}
};

template<typename T>
class BinaryMin {
public:
	BinaryMin() = default;
	T operator()(const T &a, const T &b) {
		return std::min(a, b);
	}
};

template <typename IT>
IT LabelCC(FullyDistVec<IT, IT> & father, FullyDistVec<IT, IT> & cclabel)
{
	cclabel = father;
	cclabel.ApplyInd([](IT val, IT ind){return val==ind ? -1 : val;});
	FullyDistSpVec<IT, IT> roots (cclabel, bind2nd(std::equal_to<IT>(), -1));
	roots.nziota(0);
	cclabel.Set(roots);
	cclabel = cclabel(father);
	return roots.getnnz();
}

template <class IT, class NT>
int ReduceAssign(FullyDistVec<IT,IT> &ind, FullyDistVec<IT,NT> &val, 
		std::vector<std::vector<NT>> &reduceBuffer, NT MAX_FOR_REDUCE)
{
	auto commGrid = ind.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
	int myrank;
	MPI_Comm_rank(World,&myrank);

	std::vector<int> sendcnt (nprocs,0);
	std::vector<int> recvcnt (nprocs);
	std::vector<std::vector<IT>> indBuf(nprocs);
	std::vector<std::vector<NT>> valBuf(nprocs);

	int loclen = ind.LocArrSize();
	const IT *indices = ind.GetLocArr();
	const IT *values  = val.GetLocArr();
	for(IT i = 0; i < loclen; ++i) {
		IT locind;
		int owner = ind.Owner(indices[i], locind);
		if(reduceBuffer[owner].size() == 0) {
			indBuf[owner].push_back(locind);
			valBuf[owner].push_back(values[i]);
			sendcnt[owner]++;
		}
	}

	MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);
	IT totrecv = std::accumulate(recvcnt.begin(),recvcnt.end(), static_cast<IT>(0));
	double reduceCost = ind.MyLocLength() * log2(nprocs); // bandwidth cost
	IT reducesize = 0;
	std::vector<IT> reducecnt(nprocs,0);
	
	int nreduce = 0;
	if(reduceCost < totrecv)
		reducesize = ind.MyLocLength();
	MPI_Allgather(&reducesize, 1, MPIType<IT>(), reducecnt.data(), 1, MPIType<IT>(), World);
	
	for(int i = 0; i < nprocs; ++i)
		if (reducecnt[i] > 0) nreduce++;
	
	if(nreduce > 0) {
		MPI_Request* requests = new MPI_Request[nreduce];
		MPI_Status* statuses = new MPI_Status[nreduce];

		int ireduce = 0;
		for (int i = 0; i < nprocs; ++i) {
			if(reducecnt[i] > 0) {
				reduceBuffer[i].resize(reducecnt[i], MAX_FOR_REDUCE); // this is specific to LACC
				for (int j = 0; j < sendcnt[i]; j++)
					reduceBuffer[i][indBuf[i][j]] = std::min(reduceBuffer[i][indBuf[i][j]], valBuf[i][j]);
				if (myrank == i) // recv
					MPI_Ireduce(MPI_IN_PLACE, reduceBuffer[i].data(), reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
				else // send
					MPI_Ireduce(reduceBuffer[i].data(), NULL, reducecnt[i], MPIType<NT>(), MPI_MIN, i, World, &requests[ireduce++]);
			}
		}
		MPI_Waitall(nreduce, requests, statuses);
		delete [] requests;
		delete [] statuses;
	}
	return nreduce;
}

template <class IT, class NT>
FullyDistSpVec<IT, NT> Assign(FullyDistVec<IT, IT> &ind, FullyDistVec<IT, NT> &val)
{
	IT globallen = ind.TotalLength();
	auto commGrid = ind.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int nprocs = commGrid->GetSize();
	int * rdispls = new int[nprocs+1];
	int * recvcnt = new int[nprocs];
	int * sendcnt = new int[nprocs](); // initialize to 0
	int * sdispls = new int[nprocs+1];
	
	std::vector<std::vector<NT> > reduceBuffer(nprocs);
	NT MAX_FOR_REDUCE = static_cast<NT>(globallen);
	int nreduce = ReduceAssign(ind, val, reduceBuffer, MAX_FOR_REDUCE);
	
	std::vector<std::vector<IT> > indBuf(nprocs);
	std::vector<std::vector<NT> > valBuf(nprocs);

	int loclen = ind.LocArrSize();
	const IT *indices = ind.GetLocArr();
	const IT *values  = val.GetLocArr();
	for(IT i = 0; i < loclen; ++i) {
		IT locind;
		int owner = ind.Owner(indices[i], locind);
		if(reduceBuffer[owner].size() == 0) {
			indBuf[owner].push_back(locind);
			valBuf[owner].push_back(values[i]);
			sendcnt[owner]++;
		}
	}

	MPI_Alltoall(sendcnt, 1, MPI_INT, recvcnt, 1, MPI_INT, World);
	sdispls[0] = 0;
	rdispls[0] = 0;
	for(int i = 0; i < nprocs; ++i) {
		sdispls[i + 1] = sdispls[i] + sendcnt[i];
		rdispls[i + 1] = rdispls[i] + recvcnt[i];
	}
	IT totsend = sdispls[nprocs];
	IT totrecv = rdispls[nprocs];
	
	std::vector<IT> sendInd(totsend);
	std::vector<NT> sendVal(totsend);
	for(int i=0; i < nprocs; ++i) {
		std::copy(indBuf[i].begin(), indBuf[i].end(), sendInd.begin()+sdispls[i]);
		std::vector<IT>().swap(indBuf[i]);
		std::copy(valBuf[i].begin(), valBuf[i].end(), sendVal.begin()+sdispls[i]);
		std::vector<NT>().swap(valBuf[i]);
	}
	std::vector<IT> recvInd(totrecv);
	std::vector<NT> recvVal(totrecv);

 	MPI_Alltoallv(sendInd.data(), sendcnt, sdispls, MPIType<IT>(), recvInd.data(), recvcnt, rdispls, MPIType<IT>(), World);
	MPI_Alltoallv(sendVal.data(), sendcnt, sdispls, MPIType<IT>(), recvVal.data(), recvcnt, rdispls, MPIType<IT>(), World);
	DeleteAll(sdispls, rdispls, sendcnt, recvcnt);

	int myrank;
	MPI_Comm_rank(World, &myrank);
	if(reduceBuffer[myrank].size() > 0)
		for(int i = 0; i<reduceBuffer[myrank].size(); i++)
			if(reduceBuffer[myrank][i] < MAX_FOR_REDUCE) {
				recvInd.push_back(i);
				recvVal.push_back(reduceBuffer[myrank][i]);
			}
	
	FullyDistSpVec<IT, NT> indexed(commGrid, globallen, recvInd, recvVal, false, false);
	return indexed;
}

template<typename IT>
struct ReqInfo {
	int owner;
	IT locid;
	IT index; // index in the parent's local array
	IT fetch; // index in the recvVal array

	ReqInfo() = default;
	ReqInfo(int o, IT l, IT i):owner(o), locid(l), index(i), fetch(0) {}

	bool operator<(const ReqInfo &r) const {
		return (owner != r.owner ? owner < r.owner : locid < r.locid);
	}
	bool operator!=(const ReqInfo &r) const {
		return (locid != r.locid || owner != r.owner);
	}
};

template<typename IT>
class RequestRespond {
private:
	std::vector<int> sendcnt, recvcnt;
	std::vector<int> sdispls, rdispls;
	std::vector<ReqInfo<IT> > reqs;
	std::vector<IT> recvInd;

public:
	void request(FullyDistVec<IT, IT> &ind);
	FullyDistVec<IT, IT> respond(FullyDistVec<IT, IT> &val);
};

template<typename IT>
void RequestRespond<IT>::request(FullyDistVec<IT, IT> &ind)
{
	auto commGrid = ind.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int myrank = commGrid->GetRank();
	int nprocs = commGrid->GetSize();
	int nthreads = 1;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}

	IT length = ind.LocArrSize();
	const IT *p = ind.GetLocArr();
	IT **arr2D = SpHelper::allocate2D<IT>(nthreads, nprocs);
	for (int i = 0; i < nthreads; i++)
		std::fill(arr2D[i], arr2D[i] + nprocs, 0);

	std::vector<IT> range(nthreads + 1, 0);
	for (int i = 0; i < nthreads; i++)
		range[i + 1] = range[i] + (length + i) / nthreads;

	reqs.resize(length + 1);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if (range[id + 1] > range[id]) {
			// copy array
			for (int i = range[id]; i < range[id + 1]; i++) {
				IT locid;
				int owner = ind.Owner(p[i], locid);
				reqs[i] = ReqInfo<IT>(owner, locid, i);
			}
			// sort & sendcnt
			std::sort(reqs.begin() + range[id], reqs.begin() + range[id + 1]);
			IT *scnt = arr2D[id];
			IT i = range[id];
			scnt[reqs[i].owner] = 1;
			for (++i; i < range[id + 1]; ++i)
				if (reqs[i - 1] != reqs[i])
					scnt[reqs[i].owner] += 1;
		}
	}
	// empty value
	reqs[length].owner = -1;
	reqs[length].locid = -1;
	// sendcnt
	sendcnt.resize(nprocs, 0);
	recvcnt.resize(nprocs, 0);
	int totsend = 0;
	// offset
	for (int i = 0; i < nprocs; i++)
		for (int j = 0; j < nthreads; j++) {
			sendcnt[i] += arr2D[j][i];
			IT t = arr2D[j][i];
			arr2D[j][i] = totsend;
			totsend += t;
		}

	std::vector<IT> sendInd(totsend);
	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		if (range[id + 1] > range[id]) {
			IT i = range[id];
			for (int r = 0; r < nprocs; r++)
				if (reqs[i].owner == r) {
					IT c = arr2D[id][r];
					reqs[i].fetch = c;
					sendInd[c] = reqs[i].locid;
					for (++i; reqs[i].owner == r; ++i) {
						if (reqs[i].locid != reqs[i - 1].locid)
							sendInd[++c] = reqs[i].locid;
						reqs[i].fetch = c;
					}
				}
		}
	}
	SpHelper::deallocate2D(arr2D, nthreads);

	MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, World);

	sdispls.resize(nprocs + 1, 0);
	rdispls.resize(nprocs + 1, 0);
	for (int i = 0; i < nprocs; i++) {
		sdispls[i + 1] = sdispls[i] + sendcnt[i];
		rdispls[i + 1] = rdispls[i] + recvcnt[i];
	}

	int totrecv = std::accumulate(recvcnt.begin(), recvcnt.end(), 0);
	recvInd.resize(totrecv);

	MPI_Alltoallv(
			sendInd.data(), sendcnt.data(), sdispls.data(), MPIType<IT>(),
			recvInd.data(), recvcnt.data(), rdispls.data(), MPIType<IT>(),
			World);
	sendInd.clear();
}

template<typename IT>
FullyDistVec<IT, IT> RequestRespond<IT>::respond(FullyDistVec<IT, IT> &val)
{
	auto commGrid = val.getcommgrid();
	MPI_Comm World = commGrid->GetWorld();
	int totsend = std::accumulate(sendcnt.begin(), sendcnt.end(), 0);
	int totrecv = std::accumulate(recvcnt.begin(), recvcnt.end(), 0);
	std::vector<IT> respVal(totrecv);
	int length = val.LocArrSize();
	const IT *p = val.GetLocArr();
	#pragma omp parallel for 
	for (int i = 0; i < totrecv; ++i)
		respVal[i] = p[recvInd[i]];
	recvInd.clear();

	std::vector<IT> recvVal(totsend);
	MPI_Alltoallv(
			respVal.data(), recvcnt.data(), rdispls.data(), MPIType<IT>(),
			recvVal.data(), sendcnt.data(), sdispls.data(), MPIType<IT>(),
			World);
	respVal.clear();

	std::vector<IT> q(length);
	#pragma omp parallel for
	for (int i = 0; i < length; i++)
		q[reqs[i].index] = recvVal[reqs[i].fetch];

	return FullyDistVec<IT, IT>(q, commGrid);
}

template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> SV(SpParMat<IT,NT,DER> & A, IT & nCC)
{
	FullyDistVec<IT, IT> D(A.getcommgrid());
	D.iota(A.getnrow(), 0); // D[i] <- i
	FullyDistVec<IT, IT> gp(D);  // grandparent
	FullyDistVec<IT, IT> dup(D); // duplication of grandparent
	FullyDistVec<IT, IT> mnp(D); // minimum neighbor grandparent
	FullyDistVec<IT, IT> mod(D.getcommgrid(), A.getnrow(), 1);
	IT diff = D.TotalLength();
	for (int iter = 1; diff != 0; iter++) {
double t0 = MPI_Wtime();
double t1 = MPI_Wtime();
		int spmv = 0;
		if (diff * 50 > A.getnrow()) {
			mnp = SpMV<Select2ndMinSR<IT, IT> >(A, D); // minimum of neighbors' parent
		} else {
			spmv = 1;
			FullyDistSpVec<IT, IT> SpMod(mod, [](IT modified){ return modified; });
			FullyDistSpVec<IT, IT> SpD = EWiseApply<IT>(SpMod, D,
					[](IT m, IT p) { return p; },
					[](IT m, IT p) { return true; },
					false, static_cast<IT>(0));
			FullyDistSpVec<IT, IT> hooks(A.getcommgrid(), A.getnrow());
			SpMV<Select2ndMinSR<IT, IT> >(A, SpD, hooks, false);
			mnp.EWiseApply(hooks, BinaryMin<IT>(),
					[](IT a, IT b){ return true; }, false, A.getnrow());
		}
double t2 = MPI_Wtime();
		FullyDistSpVec<IT, IT> finalhooks = Assign(D, mnp);
		D.EWiseApply(finalhooks, BinaryMin<IT>(),
				[](IT a, IT b){ return true; }, false, A.getnrow());
double t3 = MPI_Wtime();
		RequestRespond<IT> reqresp;
		reqresp.request(D);
		gp = reqresp.respond(D);
		D = gp;
		D.EWiseOut(dup, [](IT a, IT b) { return static_cast<IT>(a != b); }, mod);
		diff = static_cast<IT>(mod.Reduce(std::plus<IT>(), static_cast<IT>(0)));
		dup = D;
double t4 = MPI_Wtime();
		char out[100];
		sprintf(out, "Iteration %d: diff %ld, spmv %d\n", iter, diff, spmv);
		SpParHelper::Print(out);
		sprintf(out, "total %.3f, GP %.3f, SpMV %.3f, Hooking %.3f, Others %.3f\n",
				t4-t0, t1-t0, t2-t1, t3-t2, t4-t3);
		SpParHelper::Print(out);
	}
	FullyDistVec<IT, IT> cc(D.getcommgrid());
	nCC = LabelCC(D, cc);
	return cc;
} /* SV() */

} /* namespace combblas */
