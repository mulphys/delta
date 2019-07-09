/*! \file delta.cc
 * \brief # Implementation of the back-propagation algorithm for Neural Networks. #
 * Source: <a href="www.webpages.ttu.edu/dleverin/neural_network/neural_networks.html">www.webpages.ttu.edu/dleverin/neural_network/neural_networks.html</a>
 *
 */

#include <math.h>

#define TEST
#define DEBUG	
#define REAL	double
#define RAND	(REAL)rand() / (REAL) RAND_MAX


REAL linear ( REAL x ) {
	return x;
}

REAL linear1 ( REAL x ) {
	return 1.0;
}

REAL sigmoid ( REAL x ) {
	return (REAL) 1./(1.+exp(-x));
}

REAL sigmoid1 ( REAL x ) {
	REAL s = sigmoid(x);
	return s*(1.-s);
}

REAL tanh1 ( REAL x ) {
	REAL t = tanh(x);
	return 1.0 - t * t;
}



#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <limits>
using namespace std;

//! ## Activation function. ##

REAL 
	(*act) ( REAL x ),
	(*act1) ( REAL x );

//-REAL act( REAL x ) { 
//-#ifdef TEST
//-	return x; 
//-#else
//-	return (REAL) tanh(x);
//-#endif
//-}

//! ## Derivative of the activation function. ##
//-REAL act1( REAL x ) { 
//-	REAL tx = (REAL) tanh(x);
//-#ifdef TEST
//-	return 1.0; //-
//-#else
//-	return 1.0 - tx * tx;
//-#endif
//-}

/*! 
 * # Main routine. #
 */
int main (int argc, char *argv[]) {

act = sigmoid;
act1 = sigmoid1;

#ifdef DEBUG
printf("Activation function:");
for (int i=-10; i < 10; i++) {
	REAL x = (REAL) i;
	printf("activation(%f) = %g\n",x,(*act)(x));
}
#endif

	const int //! ## Declare variables ##
		nlh = 0,       // number of hidden levels
		nll = nlh + 2, // number of all levels
		nnh = 3;       // number of nodes on each hidden level

	printf("Number of layers: nll=%d\n",nll);
	printf("Number of hidden layers: nlh=%d\n",nlh);
	printf("Number of nodes on each hidden layer: nnh=%d",nnh);

	const int //TODO: read from a file
		nts = 4, // number of training sets
		nni = 4, // number of input nodes in each set
		nno = 1; // number of output nodes

	//! ## Define the traning data sets ##
	// generally, this is read from a file
	REAL 
		I[nts][nni] = { // input set
			{+1.0, -1.0, +1.0, -1.0}, 
			{+1.0, +1.0, +1.0, +1.0}, 
			{+1.0, +1.0, +1.0, -1.0}, 
			{+1.0, -1.0, -1.0, +1.0} 
		},
		O[nts][nno] = { // output set
			{+1.0}, 
			{+1.0}, 
			{-1.0}, 
			{-1.0}
		};

	printf ("Number of training sets: %d\n",nts);
	printf ("Number of input nodes in each set: %d\n",nni);
	printf ("Number of output nodes: %d\n",nno);

#ifdef DEBUG
//! ## Output the training sets ##
for (int its=0; its<nts; its++) {
	printf("Set: %d:",its);
	for (int ini=0; ini<nni; ini++) {
		printf(" %4.1f",I[its][ini]);
	}
	printf(" -> ");
	for (int ino=0; ino<nno; ino++) {
		printf(" %4.1f",O[its][ino]);
	}
	printf("\n");
}
#endif

	//! ## Allocate Nodes and Weights ##
	int *N = new int[nll]; // number of nodes on each level
	REAL 
		**V = new REAL*[nll], // values at the nodes 
		**B = new REAL*[nll-1], // biases
		***W= new REAL**[nll-1]; // connection weights

	//! ## Initialize Nodes ##
	// input - output layers:
	N[0] = nni;
	N[nll-1] = nno;
	V[0] = new REAL[nni];
	for (int i=0; i<nni; i++) V[0][i] = 0.0;
	V[nll-1] = new REAL[nno];
	for (int i=0; i<nno; i++) V[nll-1][i] = 0.0;
	// hidden layers:
	for (int l=1; l<nlh+1; l++) {
		N[l] = nnh; // generally, this can vary for each layer
		V[l] = new REAL[nnh];
		for (int i=0; i<nnh; i++) V[l][i]=0.0;
	}


#ifdef DEBUG	
for (int l=0; l<nll; l++) {
	printf("V[%d]:",l,V[l]);
	for (int i=0; i<N[l]; i++) 
		printf(" %g",V[l][i]);
	printf("\n");
}
printf("N[%d]:",nll);
for (int l=0; l<nll; l++) 
	printf(" %d",N[l]);
printf("\n");

printf("Initializing weights");
#endif

	//! ## Initialize weights ##

	unsigned int rand_seed = (unsigned int)time(NULL);
#ifdef DEBUG
printf("Random number seed: %d",rand_seed);
#endif

	srand(rand_seed);

	for (int l=0; l<nll-1; l++) {
		int 
			n0 = N[l],
			n1 = N[l+1];

		printf("N[%d]=%d, N[%d]=%d\n",l,n0,l+1,n1);
		B[l] = new REAL[n1];
		W[l] = new REAL*[n1];
		for (int i1=0; i1<n1; i1++) {
			B[l][i1] = 2.*RAND-1.;
#ifdef DEBUG
B[l][i1] = 0.0;
#endif
			W[l][i1] = new REAL[n0];
			for (int i0=0; i0<n0; i0++) {
				W[l][i1][i0] = 2.*RAND-1.;
#ifdef DEBUG
W[l][i1][i0] = 0.0;//-
printf("W[%d][%d][%d]=%lg\n",l,i1,i0,W[l][i1][i0]);
#endif
			}
		}
	}

	int max_iter = 40;
	REAL
		rel = 0.25, // relaxation factor (learning rate)
		eps = 1e-3, // termination criterion
		err = numeric_limits<REAL>::max();

	//! \brief ## Learning loop. ##
	//! Looping over traning sets and computing the weights.
	//
	for (int iter=0; iter<max_iter && err > eps; iter++) {

		int ne = 0; // number of error nodes computd
		err = 0.0;

#ifdef DEBUG
printf ("****** Iteration %d of %d\n",iter+1,max_iter);
#endif
		for (int its = 0; its<nts; its++) {
			printf("*** Traning set: its=%d\n",its);
#ifdef DEBUG
printf("Old Input:");
for (int i=0; i<nni; i++) printf(" %g",V[0][i]);
printf("\n");
#endif
			//! ### Retrieve new training set ###
			for (int i=0; i<nni; i++) {
				V[0][i] = I[its][i];
			}
#ifdef DEBUG
printf("New Input:");
for (int i=0; i<nni; i++) printf(" %g",V[0][i]);
printf("\n");
printf("Forward propagation\n");
#endif

			//! ### Forward propagation ####
			for (int l=0; l<nll-1; l++) {
				for (int i=0; i<N[l+1]; i++) {
					REAL v = 0.0;
					for (int j=0; j<N[l]; j++) {
						v += (*act)(V[l][j] * W[l][i][j]) + B[l][i];
					}
					V[l+1][i] = v;
				}
#ifdef DEBUG
printf("V[%d]:",l+1);
for (int i=0; i<N[l+1]; i++) 
	printf(" %g",V[l+1][i]);
printf("\n");
#endif
			} // endfor l: forward propagation

			//! ### Compute the output error ###

#ifdef DEBUG
printf("Computing the error on the output layer (%d):\n",nll);
#endif

			REAL e = 0.0;
			for (int i=0; i<N[nll-1]; i++) {
				REAL da = (*act)(V[nll-1][i]) - (*act)(O[its][i]);
				e += da*da;
				ne++;
			}
		
			err += e;

			//! ### Retreive the new output from the training set ###
#ifdef DEBUG
printf("e = %g, err = %g\n",e,err);
printf("Assigning the output %d:\n",its);
for (int i=0; i<nno; i++) {
	printf("%g\t->\t%g\n",V[nll-1][i],O[its][i]);
}
#endif

			for (int i=0; i<nno; i++) {
				V[nll-1][i] = O[its][i];
			}

			//! ### Back propagation ###
#ifdef DEBUG
printf("Back propagation\n");
#endif

			for (int l=nll-2; l>=0; l--) {
				for (int i=0; i<N[l+1]; i++) {
					REAL p = 0.0;
					for (int j=0; j<N[l]; j++) {
						p += (*act)(V[l][j]) * W[l][i][j] + B[l][i];
					}
					REAL d = rel * ((*act)(V[l+1][i]) - (*act)(p)) * (*act1)(p);
//?					B[l][i] += d;
					for (int j=0; j<N[l]; j++) {
						W[l][i][j] += d * (*act)(V[l][j]);
#ifdef DEBUG
printf("l=%d,i=%d,j=%d,W=%f\n",l,i,j,W[l][i][j]);
#endif
					}
#ifdef DEBUG
printf("W[l=%d][i=%d]:",l,i);
for (int j=0;j<N[l];j++) {
	printf(" %g",W[l][i][j]);
}
printf("\n");
#endif
				}
			}	// endfor l: back-propagation

		} // endfor its: training set
#ifdef DEBUG
printf("Iter %d: Error=%g, Normalized error=%g\n",iter,err,sqrt(err)/ne);
#endif
		err = sqrt(err)/ne;
	} // endfor MAIN LOOP


	//! ## Clean-up ##
#ifdef DEBUG
printf("Cleanup\n");fflush(stdout);
#endif
	for (int l=0; l<nll-1; l++) {
		int n = N[l+1];
		for (int i=0; i<n; i++) 
			delete (W[l][i]);
		delete (W[l]);
		delete (B[l]);
	}
	delete(W);
	delete(B);
	for (int l=0; l<nll; l++) delete(V[l]);
	delete(V);
	delete(N);

#ifdef DEBUG
printf("End.\n");fflush(stdout);
#endif
	return (0);
}

